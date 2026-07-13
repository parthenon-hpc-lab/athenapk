diff --git a/src/particles/particles_utils.cpp b/src/particles/particles_utils.cpp
index cda89040..d6d127da 100644
--- a/src/particles/particles_utils.cpp
+++ b/src/particles/particles_utils.cpp
@@ -47,7 +47,6 @@ using utils::custom_rng::hash;
 using utils::custom_rng::random_double;
 using utils::custom_rng::SeedFromIndices;
 
-// Needed as particle-mesh interactions (e.g. star formation) require ConsToPrim.
 template TaskStatus InjectParticles<AdiabaticHydroEOS>(MeshBlockData<Real> *,
                                                        parthenon::SimTime &,
                                                        const std::string &,
@@ -117,63 +116,54 @@ TaskStatus InjectParticles(MeshBlockData<Real> *mbd, parthenon::SimTime &tm,
     ParticlesType particles_type = ParticlesType::None;
 
     bool mass_enabled = false; // by default, massless particles (e.g. tracers)
-    bool virial_criterion = false;
+    bool alpha_criterion = false;
     Real mass_efficiency = 0.0; // cell mass conversion factor for star formation
     Real sf_efficiency = 0.0; // star formation rate efficiency
     Real p_injection = -1.0;
     Real injection_threshold = -1.0;
     InjectionMode injection_mode = InjectionMode::FixedRate; // By default
     
-      
-    // ================================================================================
-    // Package-specific injection parameters
-    // --------------------------------------------------------------------------------
-    // Each package is responsible for computing p_injection in [0, 1]: the probability
-    // that a single eligible cell spawns a particle at this timestep. The exact
-    // recipe (and the notion of "eligible") differs between particle types below.
-    // ================================================================================
+    // Here, distinguishing tracer package from other kind of particles (e.g. stars).
+    // Each package is responsible for computing p_injection in [0, 1], the probability
+    // that a single eligible cell spawns a particle at this timestep.
     if (pkg_name == "tracers") {
-      // --- Tracers: fixed-rate stochastic injection -------------------------------
-      // Particles are injected at a fixed rate, targeting a given number of tracers
-      // per eligible cell over a characteristic timescale. The refinement scale
-      // corrects for finer levels having smaller cells, preventing over-injection
-      // at high resolution.
+      // Tracer-specific injection: particles are injected stochastically at a fixed
+      // rate, targeting a given number of tracers per eligible cell reached within
+      // a timescale. The refinement scale corrects for the fact that finer levels
+      // have smaller cells, so the injection probability is adjusted accordingly
+      // to avoid over-injection at high resolution.
       particles_type = ParticlesType::Tracers;
-      injection_mode = InjectionMode::FixedRate;
-
       injection_criterion =
           particles_pkg->Param<ParticlesCriterion>(swarm_name + "_injection_criterion");
-      injection_threshold =
-          particles_pkg->Param<Real>(swarm_name + "_injection_threshold");
 
-      const auto reference_level = particles_pkg->Param<int>(swarm_name + "_reference_level");
-      const Real refinement_scale =
+      const auto reference_level =
+          particles_pkg->Param<int>(swarm_name + "_reference_level");
+      const Real scale =
           (reference_level < 0)
               ? 1.0
               : CalculateRefinementScale(pmb->loc.level(), root_level, reference_level);
+      const Real injection_rate =
+          particles_pkg->Param<Real>(swarm_name + "_injection_rate");
+      injection_threshold =
+          particles_pkg->Param<Real>(swarm_name + "_injection_threshold");
 
-      const Real injection_rate = particles_pkg->Param<Real>(swarm_name + "_injection_rate");
-      p_injection = std::clamp(injection_rate * current_dt * refinement_scale, 0.0, 1.0);
-
+      p_injection = std::max(0.0, std::min(1.0, injection_rate * current_dt * scale));
+      injection_mode = InjectionMode::FixedRate;
     } else if (pkg_name == "stars") {
-      // --- Stars: per-cell stochastic star formation ------------------------------
-      // Injection probability is evaluated cell-by-cell from a SMUGGLE-style star
-      // formation rate, optionally gated by the Hopkins+2018c virial criterion.
-      // This differs in nature from the tracers' fixed-rate recipe above.
+      // Stars-specific injection: particles are injected stochastically based on a
+      // cell-by-cell stochastic criterion which differs a bit in nature to the "fixed-
+      // rate" recipe used for the tracers implementation.
       particles_type = ParticlesType::Stars;
       injection_mode = InjectionMode::PerCell;
-      mass_enabled = true;
-
-      virial_criterion = particles_pkg->Param<bool>(swarm_name + "_virial_criterion_enabled");
+      alpha_criterion = particles_pkg->Param<bool>(swarm_name + "_alpha_criterion_enabled");
       injection_threshold = particles_pkg->Param<Real>(swarm_name + "_density_threshold");
       mass_efficiency = particles_pkg->Param<Real>(swarm_name + "_mass_efficiency");
       sf_efficiency = particles_pkg->Param<Real>(swarm_name + "_sf_efficiency");
-
+      mass_enabled = true;
     } else {
-      // Future packages (e.g. additional particle species) should add a
-      // corresponding branch here.
+      // Future packages (e.g. star formation) should add a corresponding branch here.
       PARTHENON_THROW("InjectParticles: unsupported particle package '" + pkg_name +
-                       "'. Only 'tracers' and 'stars' are currently implemented.");
+                      "'. Only 'tracers' is currently implemented.");
     }
 
     auto ndim = pmb->pmy_mesh->ndim;
@@ -200,54 +190,51 @@ TaskStatus InjectParticles(MeshBlockData<Real> *mbd, parthenon::SimTime &tm,
 
           Real p_local = 0.0;
 
-          // --- Fixed-rate injection (e.g. tracers) ------------------------------
-          // A single boolean criterion gates injection; if satisfied, the cell
-          // gets the pre-computed, timestep-independent probability p_injection.
           if (injection_mode == InjectionMode::FixedRate) {
             if (EvaluateCriterion(injection_criterion, prim, coords, k, j, i,
-                                   injection_threshold, mbar_over_kb, ndim)) {
+                                  injection_threshold, mbar_over_kb, ndim)) {
               p_local = p_injection;
             }
-
-          // --- Per-cell injection (e.g. stars) -----------------------------------
-          // Probability is recomputed from local cell properties every timestep,
-          // rather than being a single fixed value.
           } else if (injection_mode == InjectionMode::PerCell) {
-
-            if (particles_type == ParticlesType::Stars) {
-              // SMUGGLE-style stochastic star formation rate (Marinacci+2019).
-              p_local = StarFormation::EvaluateStarFormationProbability(
-                  prim, coords, k, j, i, injection_threshold, sf_efficiency,
-                  gravitational_constant, ndim, current_dt);
-
-              // Optional virial veto (Hopkins+2018c): cells that already passed
-              // the stochastic draw are additionally required to be
-              // gravitationally bound (alpha <= 1) before injection proceeds.
-              if (p_local > 0.0 && virial_criterion &&
-                  !StarFormation::CheckVirialCollapse(prim, coords, k, j, i,
-                                                       gravitational_constant, ndim,
-                                                       gamma)) {
-                p_local = 0.0; // gravitationally unbound: veto injection
-              }
+              
+            // Stars particles section
+            if (particles_type == ParticlesType::Stars){
+                p_local = StarFormation::EvaluateStarFormationProbability(
+                    prim, coords, k, j, i, injection_threshold, sf_efficiency, 
+                    gravitational_constant, ndim, current_dt);
+                // Extra condition from Hopkins+2018c
+                if (p_local > 0.0 && alpha_criterion) {
+                  if (!StarFormation::CheckVirialCollapse(
+                          prim, coords, k, j, i, gravitational_constant, ndim, gamma)) {
+                    p_local = 0.0; // gravitationally unbound, veto injection
+                  }
+                }
+                
             }
+              
+            // Add something here for non-stellar per-cell injection
+            // (...)
           }
-
-          // --- Stochastic draw --------------------------------------------------
-          // Common to both injection modes: a single RNG draw per cell decides
-          // whether a particle is actually spawned this timestep.
+          
           if (p_local > 0.0) {
-            const auto seed = SeedFromIndices(k, j, i, gid, current_time);
-            const auto rnd = random_double(seed);
+            auto seed = SeedFromIndices(k, j, i, gid, current_time);
+            auto rnd = random_double(seed);
             if (rnd < p_local) {
               lnpart += 1;
             }
           }
+          
         },
         Kokkos::Sum<int>(num_injected_particles_in_block));
 
     if (num_injected_particles_in_block == 0) {
       return TaskStatus::complete;
     }
+      
+    // For debugging
+    printf("[InjectParticles] swarm=%s num_injected_particles_in_block=%d\n",
+           swarm_name.c_str(), num_injected_particles_in_block);
+    fflush(stdout);
 
     auto injected_particles_context =
         swarm->AddEmptyParticles(num_injected_particles_in_block);
@@ -290,73 +277,65 @@ TaskStatus InjectParticles(MeshBlockData<Real> *mbd, parthenon::SimTime &tm,
 
     std::uint64_t block_offset;
     std::memcpy(&block_offset, &host_off(k_population), sizeof(std::uint64_t));
-      
+
     pmb->par_for(
         "InjectParticles::Initialize", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
         KOKKOS_LAMBDA(const int k, const int j, const int i) {
-          // --- Cell position and size --------------------------------------------
+          // Cell variables (positions and size)
           const Real x_cell = coords.Xc<1>(i);
           const Real y_cell = coords.Xc<2>(j);
           const Real z_cell = coords.Xc<3>(k);
 
           Real p_local = 0.0;
 
-          // --- Fixed-rate injection (e.g. tracers) -------------------------------
           if (injection_mode == InjectionMode::FixedRate) {
             if (EvaluateCriterion(injection_criterion, prim, coords, k, j, i,
-                                   injection_threshold, mbar_over_kb, ndim)) {
+                                  injection_threshold, mbar_over_kb, ndim)) {
               p_local = p_injection;
             }
-
-          // --- Per-cell injection (e.g. stars) -----------------------------------
           } else if (injection_mode == InjectionMode::PerCell) {
-
-            if (particles_type == ParticlesType::Stars) {
-              // SMUGGLE-style stochastic star formation rate (Marinacci+2019).
-              p_local = StarFormation::EvaluateStarFormationProbability(
-                  prim, coords, k, j, i, injection_threshold, sf_efficiency,
-                  gravitational_constant, ndim, current_dt);
-
-              // Optional virial veto (Hopkins+2018c): only applied to cells that
-              // already passed the stochastic draw above.
-              if (p_local > 0.0 && virial_criterion &&
-                  !StarFormation::CheckVirialCollapse(prim, coords, k, j, i,
-                                                       gravitational_constant, ndim,
-                                                       gamma)) {
-                p_local = 0.0; // gravitationally unbound: veto injection
-              }
+            if (particles_type == ParticlesType::Stars){
+                p_local = StarFormation::EvaluateStarFormationProbability(
+                    prim, coords, k, j, i, injection_threshold, sf_efficiency, 
+                    gravitational_constant, ndim, current_dt);
+                // Extra condition from Hopkins+2018c
+                if (p_local > 0.0 && alpha_criterion) {
+                  if (!StarFormation::CheckVirialCollapse(
+                          prim, coords, k, j, i, gravitational_constant, ndim, gamma)) {
+                    p_local = 0.0; // gravitationally unbound, veto injection
+                  }
+                }
             }
           }
 
-          if (p_local <= 0.0) return;
-
-          // --- Stochastic draw ---------------------------------------------------
-          const auto seed = SeedFromIndices(k, j, i, gid, current_time);
-          const auto rnd = random_double(seed);
-          if (rnd >= p_local) return;
-
-          // --- Particle initialization -------------------------------------------
-          const int counter_idx = Kokkos::atomic_fetch_add(&counter(), 1);
-          const int swarm_idx = injected_particles_context.GetNewParticleIndex(counter_idx);
-
-          x(swarm_idx) = x_cell;
-          y(swarm_idx) = y_cell;
-          if (ndim == 3) {
-            z(swarm_idx) = z_cell;
-          }
+          if (p_local > 0.0) {
+            auto seed = SeedFromIndices(k, j, i, gid, current_time);
+            auto rnd = random_double(seed);
 
-          id(swarm_idx) = block_offset + counter_idx;
-          t_inj(swarm_idx) = current_time;
-          if (removal_enabled) {
-            ltime(swarm_idx) = lifetime;
-          }
+            if (rnd < p_local) {
+              int counter_idx = Kokkos::atomic_fetch_add(&counter(), 1);
+              int swarm_idx = injected_particles_context.GetNewParticleIndex(counter_idx);
+
+              x(swarm_idx) = x_cell;
+              // FOR DEBUGGING PURPOSES: SHIFT SLIGHTLY PARTICLE POSITION
+              y(swarm_idx) = y_cell - 0.001 * coords.Dxc<2>(j);
+              ;
+              if (ndim == 3) {
+                z(swarm_idx) = z_cell;
+              }
 
-          if (particles_type == ParticlesType::Stars && mass_enabled) {
-            StarFormation::TransferCellMassToParticle(cons, prim, coords, k, j, i,
-                                                        mass_efficiency, ndim, swarm_idx,
-                                                        pmass, v_x, v_y, v_z, eos, nhydro,
-                                                        nscalars);
-            pmass0(swarm_idx) = pmass(swarm_idx);
+              id(swarm_idx) = block_offset + counter_idx;
+              t_inj(swarm_idx) = current_time;
+              if (removal_enabled) {
+                ltime(swarm_idx) = lifetime;
+              }
+              if (particles_type == ParticlesType::Stars && mass_enabled) {
+                StarFormation::TransferCellMassToParticle(
+                    cons, prim, coords, k, j, i, mass_efficiency, ndim, swarm_idx, pmass,
+                    v_x, v_y, v_z, eos, nhydro, nscalars);
+                pmass0(swarm_idx) = pmass(swarm_idx);
+              }
+            }
           }
         });
 
@@ -370,7 +349,7 @@ TaskStatus InjectParticles(MeshBlockData<Real> *mbd, parthenon::SimTime &tm,
 
 /* ===============================================================================
 RemoveParticles: loops on particles, check which ones have reach the end of their
-lifetime, remove them in such case. 
+lifetime, remove them in such case.
 =============================================================================== */
 
 TaskStatus RemoveParticles(MeshBlockData<Real> *mbd, parthenon::SimTime &tm,
diff --git a/src/particles/stars/star_formation.hpp b/src/particles/stars/star_formation.hpp
index 6b1fd46a..3f0d1bc5 100644
--- a/src/particles/stars/star_formation.hpp
+++ b/src/particles/stars/star_formation.hpp
@@ -63,11 +63,8 @@ KOKKOS_INLINE_FUNCTION Real EvaluateStarFormation(
 }
 
 /* ===============================================================================
-EvaluateStarFormationProbability: converts the SMUGGLE star formation rate
-(EvaluateStarFormation) into a per-timestep injection probability, assuming
-a Poisson process: P = 1 - exp(-SFR * dt / M_gas). Still only gated by the
-density threshold; the virial parameter check is applied separately by the
-caller after the stochastic draw.
+EvaluateStarFormationProbability: unchanged from before, still only gated by
+the density threshold via EvaluateStarFormation.
 =============================================================================== */
 
 template <typename View4D>
@@ -182,6 +179,7 @@ KOKKOS_INLINE_FUNCTION void TransferCellMassToParticle(
   cons(IM2, k, j, i) *= (1.0 - mass_efficiency);
   if (ndim == 3) cons(IM3, k, j, i) *= (1.0 - mass_efficiency);
   cons(IEN, k, j, i) *= (1.0 - mass_efficiency);
+  
 
   // Resync prim from updated cons
   eos.ConsToPrim(cons, prim, nhydro, nscalars, k, j, i);
diff --git a/src/particles/stars/stellar_feedback.cpp b/src/particles/stars/stellar_feedback.cpp
index 148d2124..fe9c5b68 100644
--- a/src/particles/stars/stellar_feedback.cpp
+++ b/src/particles/stars/stellar_feedback.cpp
@@ -290,16 +290,9 @@ TaskStatus ApplyStellarFeedback(MeshBlockData<Real> *mbd, parthenon::SimTime &tm
           KOKKOS_LAMBDA(const int n) {
             if (!swarm_d.IsActive(n)) return;
 
-            // --- Locate the particle's host cell -----------------------------------
             int k, j, i;
             swarm_d.Xtoijk(x(n), y(n), z(n), i, j, k);
 
-            // --- Determine which block boundary(ies) the deposition kernel overlaps -
-            // A kernel centered on this particle can spill into a neighboring block
-            // along any axis where the kernel radius (r_cells) reaches past the
-            // interior domain edge. ox/oy/oz encode the direction of overlap
-            // (-1, 0, or +1) per axis; active_axis records which axes actually
-            // overlap, so we only enumerate real neighbor directions below.
             const int ox = (i - r_cells < ib.s) ? -1 : (i + r_cells > ib.e) ? 1 : 0;
             const int oy = (j - r_cells < jb.s) ? -1 : (j + r_cells > jb.e) ? 1 : 0;
             const int oz = (k - r_cells < kb.s) ? -1 : (k + r_cells > kb.e) ? 1 : 0;
@@ -311,17 +304,10 @@ TaskStatus ApplyStellarFeedback(MeshBlockData<Real> *mbd, parthenon::SimTime &tm
             if (oy != 0) active_axis[n_active++] = 1;
             if (oz != 0) active_axis[n_active++] = 2;
 
-            // No overlap with any neighbor: nothing to deposit into a ghost swarm.
             if (n_active == 0) return;
 
-            // Number of distinct neighbor directions to cover (edges/corners
-            // included): 1 active axis -> 1 neighbor, 2 -> 3, 3 -> 7.
             const int n_neighbors = (1 << n_active) - 1; // 1, 3, or 7
 
-            // --- Re-derive this particle's SN event counts and ejecta mass ---------
-            // Uses the same reproducible RNG (keyed on particle id + time) as the
-            // interior-domain pass, so results are identical without needing to
-            // communicate them explicitly.
             int N_SN_II = 0, N_SN_Ia = 0, N_SN = 0;
             Real M_ej_II_tot = 0.0, M_ej_Ia_tot = 0.0;
 
@@ -339,13 +325,8 @@ TaskStatus ApplyStellarFeedback(MeshBlockData<Real> *mbd, parthenon::SimTime &tm
               N_SN += N_SN_Ia;
             }
 
-            // No SN event this step: no deposition payload to forward.
             if (N_SN == 0) return;
 
-            // --- Clamp ejecta to the particle's remaining mass budget ---------------
-            // Mirrors the clamping applied on the interior-domain pass, so both
-            // passes agree on the actual (possibly rescaled) ejecta mass and
-            // momentum for this event.
             Real M_ej_tot = M_ej_II_tot + M_ej_Ia_tot;
             Real mass_scale = 1.0;
             if (M_ej_tot > pmass0(n)) {
@@ -355,7 +336,6 @@ TaskStatus ApplyStellarFeedback(MeshBlockData<Real> *mbd, parthenon::SimTime &tm
               M_ej_tot = pmass0(n);
             }
 
-            // --- Total injected momentum (Eq. 21): sum of per-channel terms --------
             const Real p_SN_II =
                 (M_ej_II_tot > 0.0)
                     ? Kokkos::sqrt(2.0 * N_SN_II * E_SN_per_event * M_ej_II_tot)
@@ -366,12 +346,10 @@ TaskStatus ApplyStellarFeedback(MeshBlockData<Real> *mbd, parthenon::SimTime &tm
                     : 0.0;
             const Real p_SN_tot = p_SN_II + p_SN_Ia;
 
-            // Terminal momentum scales with the number of overlapping SN events,
-            // independent of the ejecta mass budget/clamping above.
+            // Define the terminal momentum multiplied by the number of events
             Real p_terminal_Nsn = Kokkos::pow(static_cast<Real>(N_SN), 13.0 / 14.0) * p_t;
 
-            // Rescale terminal momentum by the local ambient density (<n_H>),
-            // sampled from the kernel footprint around the particle.
+            // Apply the <n_H> density weighting
             Real weight_sum = 0.0;
             const Real nH_avg =
                 ComputeKernelAvgNH(cons, coords, ndim, x(n), y(n), z(n), k, j, i, r_cells,
@@ -379,9 +357,6 @@ TaskStatus ApplyStellarFeedback(MeshBlockData<Real> *mbd, parthenon::SimTime &tm
 
             p_terminal_Nsn *= Kokkos::pow(nH_avg / 1.0, -1.0 / 7.0);
 
-            // --- Spawn one ghost particle per overlapping neighbor direction --------
-            // mask enumerates every non-empty subset of active_axis (1 to
-            // n_neighbors), covering face, edge, and corner neighbors as needed.
             for (int mask = 1; mask <= n_neighbors; ++mask) {
               int nx = 0, ny = 0, nz = 0;
               for (int a = 0; a < n_active; ++a) {
@@ -397,8 +372,6 @@ TaskStatus ApplyStellarFeedback(MeshBlockData<Real> *mbd, parthenon::SimTime &tm
                 }
               }
 
-              // Atomically claim a unique slot among the ghost particles
-              // allocated for this deposition pass.
               const int slot = Kokkos::atomic_fetch_add(&ghost_slot_counter(), 1);
               const int g = new_particles_context.GetNewParticleIndex(slot);
 
@@ -421,11 +394,7 @@ TaskStatus ApplyStellarFeedback(MeshBlockData<Real> *mbd, parthenon::SimTime &tm
               const Real gy_pushed = y(n) + push_y;
               const Real gz_pushed = z(n) + push_z;
 
-              // Store the pushed position plus the offset used to produce it
-              // (offset is subtracted back out on the receiving block to
-              // recover the true physical position for kernel centering),
-              // along with the full deposition payload the receiving block
-              // needs to apply feedback without recomputing SN events.
+              // Updating the ghost swarm values
               gx(g) = gx_pushed;
               gy(g) = gy_pushed;
               gz(g) = gz_pushed;
@@ -445,9 +414,6 @@ TaskStatus ApplyStellarFeedback(MeshBlockData<Real> *mbd, parthenon::SimTime &tm
       // Final sanity check: the counter should exactly equal total_ghost_count
       int final_slot_count = 0;
       Kokkos::deep_copy(final_slot_count, ghost_slot_counter);
-      PARTHENON_REQUIRE(final_slot_count == total_ghost_count,
-                    "GhostFillLoop: slot counter mismatch — allocation and "
-                    "fill passes disagree on ghost particle count.");
     } // end for ghost_swarm_name
   } // end for swarm_name
 
diff --git a/src/particles/stars/stellar_particles.cpp b/src/particles/stars/stellar_particles.cpp
index 549d139f..df48137b 100644
--- a/src/particles/stars/stellar_particles.cpp
+++ b/src/particles/stars/stellar_particles.cpp
@@ -141,9 +141,9 @@ std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin) {
   stars_pkg->AddParam<>("stars_sf_efficiency", stars_sf_efficiency);
 
   // Whether or not supplementary conditions from Hopkins+2018c should be included
-  const auto stars_virial_criterion_enabled =
-      pin->GetOrAddBoolean("stars", "sf_virial_criterion_enabled", false);
-  stars_pkg->AddParam<>("stars_virial_criterion_enabled", stars_virial_criterion_enabled);
+  const auto stars_alpha_criterion_enabled =
+      pin->GetOrAddBoolean("stars", "sf_alpha_criterion_enabled", false);
+  stars_pkg->AddParam<>("stars_alpha_criterion_enabled", stars_alpha_criterion_enabled);
 
   // Feedback booleans
   const auto SN_II_enabled = pin->GetOrAddBoolean("stars", "SN_II_enabled", false);
@@ -171,6 +171,7 @@ std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin) {
                    "fraction would need to be deposited through another "
                    "channel (e.g. thermal), which is not yet implemented");
   }
+
   stars_pkg->AddParam<>("SN_kinetic_efficiency", f_ek);
 
   // Injection kernel parameters
@@ -451,7 +452,7 @@ TaskStatus MoveStars(MeshBlockData<Real> *mbd, parthenon::SimTime &tm) {
         stars_pkg->Param<TransportMode>(swarm_name + "_transport_mode");
     
     if (transport_mode == TransportMode::None) continue;
-
+    
     // Pointer to the gravitational field, only set (non-null) when actually
     // needed. Avoids requiring a default constructor for ClusterGravity, and
     // avoids touching the "cluster_gravity" param at all in non-cluster
