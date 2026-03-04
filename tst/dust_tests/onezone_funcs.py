import numpy as np
import matplotlib.pyplot as plt
import os
import shutil
import glob
from matplotlib.pyplot import cm
import re
import yt
from pprint import pprint
from yt.utilities.physical_constants import mp as unyt_mp
from multiprocessing import Pool, cpu_count
from itertools import groupby
import random
import gc
import matplotlib as mpl 
import statistics
import random


# mpl.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams["text.usetex"] = True 
mpl.rcParams["font.family"] = "serif"
plt.rcParams.update({
    'font.size': 14,        # default text size
    'axes.titlesize': 16,   # title size
    'axes.labelsize': 14,   # x and y labels
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
})


## Global vars
Myr_in_s = 	3.1536 * 10**13 
Gyr_in_sec = 1000 * Myr_in_s
cm_to_microM = 1e4
mp_g = 1.673 * 1e-24 

T_sput = 2e6
f_sput = 1
Z_on_Z_solar = 0.33
S_acc = 0.3
grain_density_carbonaceous_gcm3 = 2.2 ## Hydro/dust_single_grain_densities
grain_density_carbonaceous_gmicroM3 = grain_density_carbonaceous_gcm3 * 1e-12
grain_density_silicates_gcm3 = 3.3 ## Hydro/dust_single_grain_densities
grain_density_silicates_gmicroM3 = grain_density_silicates_gcm3 * 1e-12




def mysavefig(s):
    os.makedirs(os.path.dirname(s), exist_ok = True)
    plt.savefig(s, dpi=700)
    print(f"=== Saved figure: {s}")

def round_sf(x, sf=3):
    x = np.asarray(x)
    # Handle zeros safely (log10(0) = -inf)
    mags = np.floor(np.log10(np.abs(x), where=(x!=0), out=np.zeros_like(x))) + 1
    decimals = sf - mags
    factor = np.power(10.0, decimals)
    return np.round(x * factor) / factor
    
def power_law_sample(alpha, a_min, a_max, size):
    # Invert the CDF (inverse transform sampling) https://en.wikipedia.org/wiki/Inverse_transform_sampling
    r = np.random.random(size)
    exponent = 1.0 + alpha
    return ( ((a_max**exponent - a_min**exponent) * r) + a_min**exponent) ** (1.0 / exponent)



def run_onezone(parent_dir, N_particles, save_tag, t_end_Myr, dt_coarse = "auto", min_dt_Myr = 1e-4, dt_Myr_0 = 0.01, RK45_tol = 1e-9, onezone_integration="heun"):
    
    

    current_t_Myr = 0.

    fid_ds = f"{parent_dir}/" + "parthenon.vars.00000.phdf"
    fid_ds_p1 = f"{parent_dir}/" + "parthenon.vars.00001.phdf"

    onezone_times_arr_Myr = []
    total_mass_g_arr    = []
    summed_cooling_rates_per_grain_arr = []
    onezone_temperatures = []
    onezone_total_masses = []
    onezone_total_numgrains = []


    '''
    in athenaPK there is 
    const parthenon::Real mu =
        1 / (He_mass_fraction * 3. / 4. + (1 - He_mass_fraction) * 2);
    mean_molecular_mass_ = mu * units.atomic_mass_unit();
    give 0.593 if He frac = 0.25
    '''
    ### Let's grab some initial params from the dataset
    ds = yt.load(fid_ds)
    # ad = ds.all_data()
    # for k in list(ds.parameters.keys()):
    #     print(k)

    prim_carbonaceous_dust_fields_Ni = [f for f in ds.derived_field_list if "prim_dust" in f[1] and "number_density" in f[1] and "carbonaceous" in f[1]]
    prim_carbonaceous_dust_fields_Ni = sorted(prim_carbonaceous_dust_fields_Ni, key=lambda f: int(re.search(r"(\d+)$", f[1]).group(1))) ## sort numerically             
    prim_carbonaceous_dust_fields_Mi = [f for f in ds.derived_field_list if "prim_dust" in f[1] and "mass_density" in f[1] and "carbonaceous" in f[1]]
    prim_carbonaceous_dust_fields_Mi = sorted(prim_carbonaceous_dust_fields_Mi, key=lambda f: int(re.search(r"(\d+)$", f[1]).group(1))) ## sort numerically         
    cons_carbonaceous_dust_fields_Ni = [f for f in ds.derived_field_list if "cons_dust" in f[1] and "number_density" in f[1] and "carbonaceous" in f[1]]
    cons_carbonaceous_dust_fields_Ni = sorted(cons_carbonaceous_dust_fields_Ni, key=lambda f: int(re.search(r"(\d+)$", f[1]).group(1))) ## sort numerically             
    cons_carbonaceous_dust_fields_Mi = [f for f in ds.derived_field_list if "cons_dust" in f[1] and "mass_density" in f[1] and "carbonaceous" in f[1]]
    cons_carbonaceous_dust_fields_Mi = sorted(cons_carbonaceous_dust_fields_Mi, key=lambda f: int(re.search(r"(\d+)$", f[1]).group(1))) ## sort numerically         


    prim_silicate_dust_fields_Ni = [f for f in ds.derived_field_list if "prim_dust" in f[1] and "number_density" in f[1] and "silicate" in f[1]]
    prim_silicate_dust_fields_Ni = sorted(prim_silicate_dust_fields_Ni, key=lambda f: int(re.search(r"(\d+)$", f[1]).group(1))) ## sort numerically             
    prim_silicate_dust_fields_Mi = [f for f in ds.derived_field_list if "prim_dust" in f[1] and "mass_density" in f[1] and "silicate" in f[1]]
    prim_silicate_dust_fields_Mi = sorted(prim_silicate_dust_fields_Mi, key=lambda f: int(re.search(r"(\d+)$", f[1]).group(1))) ## sort numerically         
    cons_silicate_dust_fields_Ni = [f for f in ds.derived_field_list if "cons_dust" in f[1] and "number_density" in f[1] and "silicate" in f[1]]
    cons_silicate_dust_fields_Ni = sorted(cons_silicate_dust_fields_Ni, key=lambda f: int(re.search(r"(\d+)$", f[1]).group(1))) ## sort numerically             
    cons_silicate_dust_fields_Mi = [f for f in ds.derived_field_list if "cons_dust" in f[1] and "mass_density" in f[1] and "silicate" in f[1]]
    cons_silicate_dust_fields_Mi = sorted(cons_silicate_dust_fields_Mi, key=lambda f: int(re.search(r"(\d+)$", f[1]).group(1))) ## sort numerically         






    assert ds.parameters["Hydro/dust_on"] == True
    num_grain_size = ds.parameters["Hydro/dust_num_grains_sizes"]
    init_uniform_gas = ds.parameters["Hydro/init_uniform_gas"]
    uniform_gas_rho  = ds.parameters["Hydro/uniform_gas_rho"]
    uniform_gas_pres = ds.parameters["Hydro/uniform_gas_pres"]
    uniform_gas_dust_to_gas = ds.parameters["Hydro/uniform_gas_dust_to_gas"]
    mbar_over_kb = ds.parameters["Hydro/mbar_over_kb"]
    He_mass_fraction = ds.parameters["Hydro/He_mass_fraction"]
    dust_grainsize_bin_edges_microM = ds.parameters["Hydro/dust_grainsize_bin_edges_microM"] 
    dust_grain_midbin_sizes_microM = ds.parameters["Hydro/dust_grain_midbin_sizes_microM"]
    sputtering_on = ds.parameters["Hydro/dust_sputtering_on"]
    dust_metal_accretion_on = ds.parameters["Hydro/dust_metal_accretion_on"]
    dust_silicate_grains_on = ds.parameters["Hydro/dust_silicate_grains_on"]
    dust_carbonaceous_grains_on = ds.parameters["Hydro/dust_carbonaceous_grains_on"]
    nH_to_ne = ds.parameters["Hydro/nH_to_ne"]
    dust_cooling = ds.parameters["Hydro/dust_cooling"]
    init_dist = ds.parameters["Hydro/init_grainsize_distribution"]
    gamma = ds.parameters["Hydro/AdiabaticIndex"]
    mbar_gm1_over_kb_code = mbar_over_kb * (gamma-1)


    abins = dust_grainsize_bin_edges_microM
    X = 1 - He_mass_fraction


    a_min = np.min(dust_grainsize_bin_edges_microM)
    a_max = np.max(dust_grainsize_bin_edges_microM)




    if not init_uniform_gas == 1:
        print("Simulation was not uniform gas!")
        exit(1)


    code_vol_in_cgs = (ds.length_unit**3).to("cm**3").value

    rho_gas = uniform_gas_rho * ds.mass_unit /(ds.length_unit*ds.length_unit*ds.length_unit)
    rho_gas_gcm3 = rho_gas.to("g/cm**3").value
    rho_gas_code = rho_gas.to("code_mass/code_length**3").value
    pres_gas = uniform_gas_pres * ds.mass_unit / (ds.length_unit * ds.time_unit*ds.time_unit)
    pres_gas_cgs = pres_gas.to("g/(cm*s**2)").value
    # Real temp = mbar_over_kb * prim(IPR, k, j, i) / cons(IDN, k, j, i);
    T_gas_current = mbar_over_kb *  uniform_gas_pres / uniform_gas_rho
    nH_cm3 = rho_gas_gcm3 * X / mp_g ## athenaPK dust.cpp has Real n_H = rho * x_H / units_mh;




    code_vol = ds.length_unit**3
    code_mass = ds.mass_unit




    if init_dist == "MRN":
        a_array = power_law_sample(alpha=-4.5, a_min=a_min, a_max=a_max, size=N_particles)
    if init_dist == "MRN_inverse":
        a_array = power_law_sample(alpha=4.5, a_min=a_min, a_max=a_max, size=N_particles)
    if init_dist == "flat":
        a_array = power_law_sample(alpha=0, a_min=a_min, a_max=a_max, size=N_particles)
    if init_dist == "flat_in_range":
        ### dont overwrite a_min and a_max here since these determine the rebinning/dust removal edges
        flat_graindist_in_range_amin = ds.parameters["Hydro/flat_graindist_in_range_amin"]
        flat_graindist_in_range_amax = ds.parameters["Hydro/flat_graindist_in_range_amax"]
        print("flat_graindist_in_range_amin", flat_graindist_in_range_amin)
        print("flat_graindist_in_range_amax", flat_graindist_in_range_amax)
        a_array = power_law_sample(alpha=0, a_min=flat_graindist_in_range_amin, a_max=flat_graindist_in_range_amax, size=N_particles)


    random.shuffle(a_array) ## Randomise, especially important if we have 2 compositions since we will split the list in 2
    print("init_dist", init_dist)


    y_C = []
    y2_C = []
    
    y_S = []
    y2_S = []
    
    if dust_carbonaceous_grains_on:
        for j in range(len( prim_carbonaceous_dust_fields_Ni)):
            ### y_C and y2_C should be identical
            thisfield = prim_carbonaceous_dust_fields_Ni[j]
            y_C.append(np.sum(ds.r[thisfield][:].value * ds.r["gas","mass"].to(code_mass).value))
            thisfield = cons_carbonaceous_dust_fields_Ni[j]
            y2_C.append(np.sum(ds.r[thisfield][:].value * ds.r["gas","volume"].to(code_vol).value))

    if dust_silicate_grains_on:
        for j in range(len( prim_silicate_dust_fields_Ni)):
            thisfield = prim_silicate_dust_fields_Ni[j]
            y_S.append(np.sum(ds.r[thisfield][:].value * ds.r["gas","mass"].to(code_mass).value))
            thisfield = cons_silicate_dust_fields_Ni[j]
            y2_S.append(np.sum(ds.r[thisfield][:].value * ds.r["gas","volume"].to(code_vol).value))

    # y  = np.sum(np.asarray(y_S)) + np.sum(np.asarray(y_C)) 
    # y2  = np.sum(np.asarray(y2_S)) + np.sum(np.asarray(y2_C)) 
    
    y = y_C + y_S
    
    hist_values, bin_edges, patches =  plt.hist(a_array, bins = abins, histtype = "step", density = False)
    max_bin = np.argmax(hist_values)
    norm_factor_Ni = hist_values[max_bin] / y[max_bin] ### need time-constant norm to normalise Ni and Mi at time=0 between one-zone model and simulation
    one_zone_to_simulation_grain_number_ratio = np.size(a_array) / np.sum(np.asarray(y)) ### should basically equal norm_factor_Ni if the initial distributions match
    total_num_grains_in_simulation = np.sum(np.asarray(y))
    
    C_end_idx = int(len(a_array) * np.sum(np.asarray(y_C))/(np.sum(np.asarray(y_C))+np.sum(np.asarray(y_S))))
    print("C_end_idx 1", C_end_idx)
    
    comp_types = np.zeros_like(a_array)
    comp_types[C_end_idx:] = 1
    


    domain_width = ds.domain_width.to("cm").value
    simulation_volume_cgs = np.prod(domain_width)

    volume_from_sum = np.sum(ds.r['parthenon', 'volume'].to("cm**3").value)
    volume_from_domain = simulation_volume_cgs  # computed from domain_width
    assert np.isclose(volume_from_sum, volume_from_domain, rtol=1e-8, atol=0), \
        f"Volumes differ! sum={volume_from_sum}, domain={volume_from_domain}"


    ### Make the dust densities = N/V equal
    one_zone_effective_volume_cgs = simulation_volume_cgs * one_zone_to_simulation_grain_number_ratio    
    thermal_E_per_vol_code = rho_gas_code * T_gas_current / mbar_gm1_over_kb_code
    code_energy = ds.mass_unit * (ds.length_unit**2) / (ds.time_unit**2)
    code_vol = ds.length_unit**3



    code_energy_in_erg = code_energy.to("erg").value
    code_vol_in_cm3 = code_vol.to("cm**3").value

    median_code_energy_over_mass = np.median((ds.r[('gas', 'specific_thermal_energy')]).to(code_energy/code_mass).value)
    median_code_energy_over_volume = np.median((ds.r[('gas', 'specific_thermal_energy')]*ds.r[('gas', 'density')] ).to(code_energy/code_vol).value)
    median_code_energy_over_volume_ergs_per_cm3 = np.median((ds.r[('gas', 'specific_thermal_energy')]*ds.r[('gas', 'density')] ).to("erg/cm**3").value)
    dust_number_density_cgs = np.size(a_array) / one_zone_effective_volume_cgs
    electron_number_density_cgs = nH_to_ne * nH_cm3



    code_energy_per_vol_conversion_1 = np.median((ds.r[('gas', 'specific_thermal_energy')]*ds.r[('gas', 'density')]).to(code_energy/code_vol).value) / np.median((ds.r[('gas', 'specific_thermal_energy')]*ds.r[('gas', 'density')]).to("erg/cm**3").value)
    code_energy_per_vol_conversion_2 = 1./ (code_energy_in_erg / code_vol_in_cm3)


    print("[FJJ DEBUG] code_energy_per_vol_conversion_1 vs code_energy_per_vol_conversion_2", code_energy_per_vol_conversion_1, code_energy_per_vol_conversion_2)
    print("[FJJ DEBUG] Total Simulation dust number = ", np.sum(y))
    print("[FJJ DEBUG] one_zone_to_simulation_grain_number_ratio", one_zone_to_simulation_grain_number_ratio, norm_factor_Ni)
    print("[FJJ DEBUG] simulation_volume_cgs = ", simulation_volume_cgs)
    print("[FJJ DEBUG] one_zone_effective_volume_cgs =", one_zone_effective_volume_cgs)
    print("[FJJ DEBUG] dust_number_density_cgs", dust_number_density_cgs, "total_num_grains_in_simulation / vol_cgs", total_num_grains_in_simulation/ simulation_volume_cgs )
    print("[FJJ DEBUG] should match: median_code_energy_over_volume", median_code_energy_over_volume, "thermal_E_per_vol_code", thermal_E_per_vol_code)
    print("[FJJ DEBUG] median_code_energy_over_mass * mbar_gm1_over_kb_code should = T", median_code_energy_over_mass * mbar_gm1_over_kb_code, " actual T in K = ", np.median(ds.r[('gas', 'temperature')].to("K").value))
    print("[FJJ DEBUG] uniform_gas_rho = ", uniform_gas_rho)
    print("[FJJ DEBUG] uniform_gas_dust_to_gas = ", uniform_gas_dust_to_gas)
    print("[FJJ DEBUG] rho_gas_gcm3=", rho_gas_gcm3)
    print("[FJJ DEBUG] pres_gas_cgs", pres_gas_cgs)
    print("[FJJ DEBUG] Log onezone Temperature", np.log10(T_gas_current))
    print("[FJJ DEBUG] Log simulation Temperature", np.log10(np.median(ds.all_data()["gas","temperature"].to("K").value)))

    if dt_coarse == "auto":
        ds_p1 = yt.load(fid_ds_p1)
        dt_between_snaps = ds_p1.current_time.to("Myr").value - ds.current_time.to("Myr").value

        # dt_between_snaps = 10
        dt_coarse = dt_between_snaps / 2
    onezone_snap_times = np.arange(0, dt_coarse * (1+int((t_end_Myr//dt_coarse))), step = dt_coarse) 







    def do_sputtering_vec(a, T_gas_current):
        da_dt = - f_sput * 3.2e-18 * (rho_gas_gcm3/mp_g) / (((T_sput/T_gas_current)**2.5) + 1.)
        da_dt *= cm_to_microM
        return da_dt
        # return a + (da_dt * dt_in_s)

    def do_metal_accretion_vec(a, T_gas_current):
        da_dt = Z_on_Z_solar * (nH_cm3 / 1000) * np.sqrt(T_gas_current/10) * S_acc / 0.3  # microM per Gyr
        return da_dt / Gyr_in_sec
        # return a + (da_dt * dt_in_s / Gyr_in_sec)

    def do_rebinning(arr):
        arr = np.minimum(arr, a_max)
        return arr


    def cooling_rates_per_grain_func(arr, T_gas_current):
        cooling_rates_per_grain = np.zeros_like(arr)
        chi = 1.71 * 1e8 * np.power(arr, 2/3) / T_gas_current

        cooling_rates_per_grain[chi>=4.5]                 = 5.38 * 1e-18 * arr[chi>=4.5]**2 * T_gas_current**1.5
        cooling_rates_per_grain[(chi<4.5)&(chi>=1.5)]     = 3.37 * 1e-13 * arr[(chi<4.5)&(chi>=1.5)]**2.41 * T_gas_current**0.88
        cooling_rates_per_grain[chi<1.5]                  = 6.48 * 1e-6  * arr[chi<1.5]**3 
        return cooling_rates_per_grain


    def get_T_from_thermal_E_per_vol_erg_cm3(thermal_E_per_vol_erg_cm3):
        thermal_E_per_vol_code = thermal_E_per_vol_erg_cm3 * code_vol_in_cm3 / code_energy_in_erg
        # print("thermal_E_per_vol_code = thermal_E_per_vol_erg_cm3 * code_vol_in_cm3 / code_energy_in_erg", np.shape(thermal_E_per_vol_code),  np.shape(thermal_E_per_vol_erg_cm3),  np.shape(code_vol_in_cm3),  np.shape(code_energy_in_erg),)
        return thermal_E_per_vol_code * mbar_gm1_over_kb_code / rho_gas_code



    def cooling_rates_per_grain_func_RK(dummy_t, a_array, thermal_E_per_vol_erg_cm3):
        ### Function defined to keep main block free of conversions between E and T
        T_gas_current = get_T_from_thermal_E_per_vol_erg_cm3(thermal_E_per_vol_erg_cm3)
        cooling_rates_per_grain = cooling_rates_per_grain_func(a_array, T_gas_current)
        assert np.sum(cooling_rates_per_grain == 0) == 0
        summed_cooling_rates_per_grain = np.sum(cooling_rates_per_grain)
        volumetric_cooling_erg_per_s_per_cm3 =  -1. * electron_number_density_cgs * summed_cooling_rates_per_grain / one_zone_effective_volume_cgs
        return volumetric_cooling_erg_per_s_per_cm3



    def do_cooling_evolution_RK4(a_array, T_gas_current, dt_Myr):
        
        thermal_E_per_vol_code =  rho_gas_code * T_gas_current / mbar_gm1_over_kb_code
        thermal_E_per_vol_erg_cm3 = thermal_E_per_vol_code * code_energy_in_erg / code_vol_in_cm3
        T_gas_current = get_T_from_thermal_E_per_vol_erg_cm3(thermal_E_per_vol_erg_cm3)    
        ### Do RK4 method 
        h = dt_Myr*Myr_in_s
        y0 = thermal_E_per_vol_erg_cm3
        yn = y0
        t = 0 ### our func does not depend on time, so use dummy var here
        k1 = cooling_rates_per_grain_func_RK(t, a_array, yn)
        yn_hk12 = yn + h*k1/2.
        k2 = cooling_rates_per_grain_func_RK(t, a_array, yn_hk12)
        yn_hk22 = yn + h*k2/2.
        k3 = cooling_rates_per_grain_func_RK(t, a_array, yn_hk22) 
        yn_hk3 = yn + h*k3
        k4 = cooling_rates_per_grain_func_RK(t, a_array, yn_hk3) 
        yn1 = yn + (h/6.)*(k1+2*k2+2*k3+k4)
        thermal_E_per_vol_erg_cm3 = yn1
        T_gas_current = get_T_from_thermal_E_per_vol_erg_cm3(thermal_E_per_vol_erg_cm3)
        summed_cooling_rates_per_grain = yn - yn1
        return summed_cooling_rates_per_grain, T_gas_current
        
        
        
    def AthenaPK_RK45_Stepper(a_array, T_gas_current, dt_Myr):
        thermal_E_per_vol_code =  rho_gas_code * T_gas_current / mbar_gm1_over_kb_code
        thermal_E_per_vol_erg_cm3 = thermal_E_per_vol_code * code_energy_in_erg / code_vol_in_cm3
        T_gas_current = get_T_from_thermal_E_per_vol_erg_cm3(thermal_E_per_vol_erg_cm3)    
        ### Do RK4 method 
        
        accepted_error = False
        y0 = thermal_E_per_vol_erg_cm3
        h = dt_Myr*Myr_in_s
        
        while(accepted_error == False):
            
            yn = y0
            t0 = 0 ### our func does not depend on time, so use dummy var here
            ### This is literally copy-pasted and made pythonic from the athenaPK code from P. Grete
            # - so we should have best match possible to the cooling evolution
            k1 = h * cooling_rates_per_grain_func_RK(t0, a_array, y0)
            k2 = h * cooling_rates_per_grain_func_RK(t0 + 1. / 4. * h, a_array, y0 + 1. / 4. * k1)
            k3 = h * cooling_rates_per_grain_func_RK(t0 + 3. / 8. * h, a_array, y0 + 3. / 32. * k1 + 9. / 32. * k2)
            k4 = h * cooling_rates_per_grain_func_RK(t0 + 12. / 13. * h, a_array, y0 + 1932. / 2197. * k1 - 7200. / 2197. * k2 + 7296. / 2197. * k3)
            k5 = h * cooling_rates_per_grain_func_RK(t0 + h, a_array, y0 + 439. / 216. * k1 - 8. * k2 + 3680. / 513. * k3 - 845. / 4104. * k4)
            k6 = h * cooling_rates_per_grain_func_RK(t0 + 1. / 2. * h, a_array, y0 - 8. / 27. * k1 + 2. * k2 - 3544. / 2565. * k3 + 1859. / 4104. * k4 - 11. / 40. * k5); ## TODO(forrestglines): Check k2?
            y1_l = y0 + 25. / 216. * k1 + 1408. / 2565. * k3 + 2197. / 4104. * k4 - 1. / 5. * k5 ## 4th order
            y1_h = y0 + 16. / 135. * k1 + 6656. / 12825. * k3 + 28561. / 56430. * k4 - 9. / 50. * k5 + 2. / 55. * k6 ## 5th order
            err = abs((y1_h - y1_l) / y1_h)
            
            # print("err", err)
            if err < (RK45_tol) or h < min_dt_Myr*Myr_in_s:
                accepted_error = True
            
            else:
                h = 0.95 * h * np.power(RK45_tol / err, 2)
                
        dt_Myr = h / Myr_in_s
        thermal_E_per_vol_erg_cm3 = y1_h
        T_gas_current = get_T_from_thermal_E_per_vol_erg_cm3(thermal_E_per_vol_erg_cm3)
        summed_cooling_rates_per_grain = y0 - y1_h
        return summed_cooling_rates_per_grain, T_gas_current, dt_Myr
            
        
        



    def evolve_onezone_to_time_w_subcycling(target_t_Myr, a_array, comp_types, T_gas_current, current_t_Myr, total_times_arr_Myr, total_mass_g_arr, summed_cooling_rates_per_grain_arr, onezone_temperatures, onezone_total_masses, onezone_total_numgrains):
        
        # print("T_gas_current = ", T_gas_current)
        original_t = current_t_Myr
        while(current_t_Myr < target_t_Myr):
        
            
            a_array = a_array[a_array>a_min]
            if "Dwek_Werner1981" in dust_cooling: 
                summed_cooling_rates_per_grain, T_gas_current, dt_Myr_this_step = AthenaPK_RK45_Stepper(a_array, T_gas_current, dt_Myr_0)
                summed_cooling_rates_per_grain_arr.append(summed_cooling_rates_per_grain)
                
            dt_in_s_this_step = dt_Myr_this_step*Myr_in_s
                
            progress =  100*(current_t_Myr-original_t)/(target_t_Myr-original_t)
            if progress % 0.5 < 0.25:
                print(f"At {round(progress,3)} % of current timejump evolution. Evolving from {round(original_t ,3)} to {round(target_t_Myr,3)} Myr")
            
            particle_masses_C = 4. * np.pi * (1/3.) * a_array[comp_types==0]**3 * grain_density_carbonaceous_gmicroM3   
            particle_masses_S = 4. * np.pi * (1/3.) * a_array[comp_types==1]**3 * grain_density_silicates_gmicroM3   
            particle_masses = np.concatenate((particle_masses_C , particle_masses_S))
            
            total_mass_g = np.sum(particle_masses)
            
            total_times_arr_Myr.append(current_t_Myr)
            total_mass_g_arr.append(total_mass_g)
            onezone_temperatures.append(T_gas_current)

            da_dt = np.zeros_like(a_array)
            
            
            if onezone_integration == "euler":
                if sputtering_on: da_dt += do_sputtering_vec(a_array, T_gas_current)
                if dust_metal_accretion_on: da_dt+= do_metal_accretion_vec(a_array, T_gas_current)
                a_array = a_array + (da_dt * dt_in_s_this_step)
            elif onezone_integration == "heun":
                
                da_dt_i = np.zeros_like(a_array)
                if sputtering_on: da_dt_i += do_sputtering_vec(a_array, T_gas_current)
                if dust_metal_accretion_on: da_dt_i+= do_metal_accretion_vec(a_array, T_gas_current)
                heun_ip1 = a_array + (da_dt_i * dt_in_s_this_step)
                
                da_dt_ip1 = np.zeros_like(a_array)
                if sputtering_on: da_dt_ip1 += do_sputtering_vec(heun_ip1, T_gas_current)
                if dust_metal_accretion_on: da_dt_ip1+= do_metal_accretion_vec(heun_ip1, T_gas_current)
                a_array = a_array + ( (dt_in_s_this_step/2)*(da_dt_i+da_dt_ip1) )
                
            
            a_array = do_rebinning(a_array)
            ### remove too-small particles
            # print("a_min pre clean", a_min, np.min(a_array))
            comp_types = comp_types[a_array>a_min]
            a_array = a_array[a_array>a_min]
            # print("a_min post clean", a_min, np.min(a_array))
            
            current_t_Myr += dt_Myr_this_step
            
            # print("Appending to onezone_total_masses")
            onezone_total_masses.append(np.sum(particle_masses))
            onezone_total_numgrains.append(np.size(a_array))
            
            # print("current_t_Myr = ", current_t_Myr)
            
        particle_masses_C = 4. * np.pi * (1/3.) * a_array[comp_types==0]**3 * grain_density_carbonaceous_gmicroM3   
        particle_masses_S = 4. * np.pi * (1/3.) * a_array[comp_types==1]**3 * grain_density_silicates_gmicroM3   
        particle_masses = np.concatenate((particle_masses_C , particle_masses_S)) 
        return a_array, comp_types, T_gas_current, current_t_Myr, total_times_arr_Myr, total_mass_g_arr, summed_cooling_rates_per_grain_arr, particle_masses, onezone_temperatures, onezone_total_masses, onezone_total_numgrains
        
        
        
    current_time_snap = 0.
    for snap_time in onezone_snap_times:
        # if current_time_snap < time_first_plots
        current_time_string = str("%.2f" % snap_time + " Myr")
        a_array, comp_types, T_gas_current, current_t_Myr, onezone_times_arr_Myr, total_mass_g_arr, summed_cooling_rates_per_grain_arr, particle_masses, onezone_temperatures, onezone_total_masses, onezone_total_numgrains = evolve_onezone_to_time_w_subcycling(snap_time,                                                                                                                                                              
                                                a_array, comp_types, T_gas_current, current_t_Myr, onezone_times_arr_Myr, total_mass_g_arr, summed_cooling_rates_per_grain_arr, 
                                                onezone_temperatures, onezone_total_masses, onezone_total_numgrains)
        
        os.makedirs(f"{parent_dir}/Testing_Dust_Model_vs_OneZone/Onezone_data/", exist_ok = True)
        np.savez(
            f"{parent_dir}/Testing_Dust_Model_vs_OneZone/Onezone_data/data_{save_tag}_{'%.2f' % snap_time}.npz",
            a_array=a_array,
            comp_types = comp_types,
            T_gas_current=T_gas_current,
            current_t_Myr=current_t_Myr,
            onezone_times_arr_Myr=onezone_times_arr_Myr,
            total_mass_g_arr=total_mass_g_arr,
            summed_cooling_rates_per_grain_arr=summed_cooling_rates_per_grain_arr,
            particle_masses=particle_masses,
            onezone_temperatures=onezone_temperatures,
            onezone_total_masses=onezone_total_masses,
            onezone_total_numgrains=onezone_total_numgrains,
            save_tag=save_tag,
        )







##########################



def getNorms(fiducial_ds, fiducial_one_zone):
    ds = yt.load(fiducial_ds)

    ad = ds.all_data()
    # print("fiducial_ds", fiducial_ds)
    # for k in list(ds.parameters.keys()):
    #     print(k, ds.parameters[k])

    code_vol = ds.length_unit**3
    code_mass = ds.mass_unit
    
    dust_grain_midbin_sizes_microM = ds.parameters["Hydro/dust_grain_midbin_sizes_microM"]
    dust_silicate_grains_on = ds.parameters["Hydro/dust_silicate_grains_on"]
    dust_carbonaceous_grains_on = ds.parameters["Hydro/dust_carbonaceous_grains_on"]

    dust_grain_midbin_sizes_microM = ds.parameters["Hydro/dust_grain_midbin_sizes_microM"]

    data = np.load(fiducial_one_zone)  # remove mmap_mode
    a_array = np.array(data["a_array"])  # force into RAM
    comp_types = np.array(data["comp_types"])
    particle_masses = data["particle_masses"]



    prim_carbonaceous_dust_fields_Ni = [f for f in ds.derived_field_list if "prim_dust" in f[1] and "number_density" in f[1] and "carbonaceous" in f[1]]
    prim_carbonaceous_dust_fields_Ni = sorted(prim_carbonaceous_dust_fields_Ni, key=lambda f: int(re.search(r"(\d+)$", f[1]).group(1))) ## sort numerically             
    prim_carbonaceous_dust_fields_Mi = [f for f in ds.derived_field_list if "prim_dust" in f[1] and "mass_density" in f[1] and "carbonaceous" in f[1]]
    prim_carbonaceous_dust_fields_Mi = sorted(prim_carbonaceous_dust_fields_Mi, key=lambda f: int(re.search(r"(\d+)$", f[1]).group(1))) ## sort numerically         
    cons_carbonaceous_dust_fields_Ni = [f for f in ds.derived_field_list if "cons_dust" in f[1] and "number_density" in f[1] and "carbonaceous" in f[1]]
    cons_carbonaceous_dust_fields_Ni = sorted(cons_carbonaceous_dust_fields_Ni, key=lambda f: int(re.search(r"(\d+)$", f[1]).group(1))) ## sort numerically             
    cons_carbonaceous_dust_fields_Mi = [f for f in ds.derived_field_list if "cons_dust" in f[1] and "mass_density" in f[1] and "carbonaceous" in f[1]]
    cons_carbonaceous_dust_fields_Mi = sorted(cons_carbonaceous_dust_fields_Mi, key=lambda f: int(re.search(r"(\d+)$", f[1]).group(1))) ## sort numerically         

    prim_silicate_dust_fields_Ni = [f for f in ds.derived_field_list if "prim_dust" in f[1] and "number_density" in f[1] and "silicate" in f[1]]
    prim_silicate_dust_fields_Ni = sorted(prim_silicate_dust_fields_Ni, key=lambda f: int(re.search(r"(\d+)$", f[1]).group(1))) ## sort numerically             
    prim_silicate_dust_fields_Mi = [f for f in ds.derived_field_list if "prim_dust" in f[1] and "mass_density" in f[1] and "silicate" in f[1]]
    prim_silicate_dust_fields_Mi = sorted(prim_silicate_dust_fields_Mi, key=lambda f: int(re.search(r"(\d+)$", f[1]).group(1))) ## sort numerically         
    cons_silicate_dust_fields_Ni = [f for f in ds.derived_field_list if "cons_dust" in f[1] and "number_density" in f[1] and "silicate" in f[1]]
    cons_silicate_dust_fields_Ni = sorted(cons_silicate_dust_fields_Ni, key=lambda f: int(re.search(r"(\d+)$", f[1]).group(1))) ## sort numerically             
    cons_silicate_dust_fields_Mi = [f for f in ds.derived_field_list if "cons_dust" in f[1] and "mass_density" in f[1] and "silicate" in f[1]]
    cons_silicate_dust_fields_Mi = sorted(cons_silicate_dust_fields_Mi, key=lambda f: int(re.search(r"(\d+)$", f[1]).group(1))) ## sort numerically         

    # for field in ds.derived_field_list:
    #     print(field)
    # y = []
    # y2 = []
    # for j in range(len(prim_carbonaceous_dust_fields_Ni)):
    #     thisfield = prim_carbonaceous_dust_fields_Ni[j]
    #     y.append(np.sum(ad[thisfield][:].value * ad["gas","mass"].to(code_mass).value))
    #     thisfield = cons_carbonaceous_dust_fields_Ni[j]
    #     y2.append(np.sum(ad[thisfield][:].value * ad["gas","volume"].to(code_vol).value))

    x = dust_grain_midbin_sizes_microM
    y_C = []
    y2_C = []
    
    y_S = []
    y2_S = []
    
    if dust_carbonaceous_grains_on:
        ### y_C and y2_C should be identical
        for j in range(len( prim_carbonaceous_dust_fields_Ni)):
            thisfield = prim_carbonaceous_dust_fields_Ni[j]
            y_C.append(np.sum(ds.r[thisfield][:].value * ds.r["gas","mass"].to(code_mass).value))
            thisfield = cons_carbonaceous_dust_fields_Ni[j]
            y2_C.append(np.sum(ds.r[thisfield][:].value * ds.r["gas","volume"].to(code_vol).value))

    if dust_silicate_grains_on:
        for j in range(len( prim_silicate_dust_fields_Ni)):
            thisfield = prim_silicate_dust_fields_Ni[j]
            y_S.append(np.sum(ds.r[thisfield][:].value * ds.r["gas","mass"].to(code_mass).value))
            thisfield = prim_silicate_dust_fields_Ni[j]
            y2_S.append(np.sum(ds.r[thisfield][:].value * ds.r["gas","volume"].to(code_vol).value))

    y_S = np.sum(np.asarray(y_S))
    y_C = np.sum(np.asarray(y_C))
    y2_S = np.sum(np.asarray(y2_S))
    y2_C = np.sum(np.asarray(y2_C))
    y  = y_S + y_C
    # exit()
    
    
    norm_factor_Ni = np.size(particle_masses)/np.sum(np.asarray(y)) ## hist_values[max_bin] / y[max_bin] ### need time-constant norm to normalise Ni and Mi at time=0 between one-zone model and simulation


    particle_masses_C = 4. * np.pi * (1/3.) * a_array[comp_types==0]**3 * grain_density_carbonaceous_gmicroM3   
    particle_masses_S = 4. * np.pi * (1/3.) * a_array[comp_types==1]**3 * grain_density_silicates_gmicroM3   
    particle_masses = np.concatenate((particle_masses_C , particle_masses_S))
    
    ### quick check
    norm_factor_Mi = np.sum(particle_masses)/np.sum(np.asarray(y))
    print(norm_factor_Ni, norm_factor_Mi, np.sum(particle_masses), ds.mass_unit.to("g").value)
    
    
    norm_factor_Mi = norm_factor_Ni * ds.mass_unit.to("g").value  ### make exactly consistent with Ni norm
    
    # exit()

    # norm_factor_Mi_2 = 
    # ## doing norm_factor_Mi = np.sum(particle_masses)/np.sum(np.asarray(y))  tends to result in biases due to poor stats in low-number, largegrainsize bins
    # abins = ds.parameters["Hydro/dust_grainsize_bin_edges_microM"] 
    # hist_values, bin_edges =  np.histogram(a_array, bins = abins, weights = particle_masses, density = False)
    # median_value = statistics.median(hist_values)
    # median_index = list(hist_values).index(median_value)
    # norm_factor_Mi = median_value/np.asarray(y)[median_index]
    
    
    
    del ds
    del ad
    del data
    del a_array
    del particle_masses
    gc.collect()
    return norm_factor_Ni, norm_factor_Mi









def get_fractional_errors(onezone_data_path_root, onezone_save_tag, sims_parent_dir, snapshot_number, norm_factor_Ni=None, norm_factor_Mi=None, out_file_ext="phdf", do_hists=True, plot_style="line", my_figsize = (4,3) ):
    onezone_data_path = onezone_data_path_root + "/Onezone_data/"
    onezone_data_files = [x for x in os.listdir(onezone_data_path) if ".npz" in x and  onezone_save_tag in x]
    onezone_times = [float(x.split("_")[-1].split(".npz")[0]) for x in onezone_data_files]
    fiducial_one_zone = [x for x in onezone_data_files if "_0.00" in x][0]
    
    all_sim_files = list(sorted([x for x in os.listdir(sims_parent_dir)]))
    all_sim_files = [sims_parent_dir+"/" +x for x in all_sim_files if str(snapshot_number).zfill(5) in x and x.split(".")[-1] == out_file_ext]
    try:
        fid_ds = f"{sims_parent_dir}/" + "parthenon.vars.00000.phdf"
        fiducial_ds = all_sim_files[0].rsplit('.', 2)[0] + ".00000.phdf"
    except Exception as e:
        print(e)
        return np.nan,np.nan,np.nan
    
    frac_err_dict = {}
    
    
    for p in yt.load(fiducial_ds).parameters.items():
        print(p)
    # exit()
    if norm_factor_Ni is None or norm_factor_Mi is None:
        norm_factor_Ni, norm_factor_Mi = getNorms(fiducial_ds, onezone_data_path + "/" + fiducial_one_zone)


    fig_frac_err_vs_onezone = plt.figure(figsize=(4,4))
    fig_frac_err_vs_yardstick = plt.figure(figsize=(4,4))
    plotted_guideline = False
    custom_handles = []
    for sim_file in all_sim_files:
        ds = yt.load(sim_file)
        current_time = float(ds.current_time.to("Myr").value)
        nearest_onezone_time = min(onezone_times, key=lambda x: abs(x - current_time))
        current_time_string = str("%.1f" % round(current_time,1) + "_Myr")
        one_zone_file = [x for x in onezone_data_files if str(nearest_onezone_time) in x][0]
        print("current_time_string=",current_time_string,"nearest_onezone_time=", nearest_onezone_time )
        
        if current_time > 1e-10:
            if not abs(current_time-nearest_onezone_time)/current_time  < 0.3:
                print(nearest_onezone_time, current_time)
            assert abs(current_time-nearest_onezone_time)/current_time  < 0.3
        code_vol = ds.length_unit**3
        code_mass = ds.mass_unit
        
        
        frac_err_dict["simulation_time"]        = current_time
        frac_err_dict["nearest_onezone_time"]   = nearest_onezone_time
        

        num_grain_size   = ds.parameters["Hydro/dust_num_grains_sizes"]
        init_uniform_gas = ds.parameters["Hydro/init_uniform_gas"]
        uniform_gas_rho  = ds.parameters["Hydro/uniform_gas_rho"]
        uniform_gas_pres = ds.parameters["Hydro/uniform_gas_pres"]
        uniform_gas_dust_to_gas = ds.parameters["Hydro/uniform_gas_dust_to_gas"]
        mbar_over_kb = ds.parameters["Hydro/mbar_over_kb"]
        He_mass_fraction = ds.parameters["Hydro/He_mass_fraction"]
        dust_grainsize_bin_edges_microM = ds.parameters["Hydro/dust_grainsize_bin_edges_microM"] 
        dust_grain_midbin_sizes_microM = ds.parameters["Hydro/dust_grain_midbin_sizes_microM"]
        sputtering_on = ds.parameters["Hydro/dust_sputtering_on"]
        dust_metal_accretion_on = ds.parameters["Hydro/dust_metal_accretion_on"]
        dust_silicate_grains_on = ds.parameters["Hydro/dust_silicate_grains_on"]
        dust_carbonaceous_grains_on = ds.parameters["Hydro/dust_carbonaceous_grains_on"]
        nH_to_ne = ds.parameters["Hydro/nH_to_ne"]
        dust_cooling = ds.parameters["Hydro/dust_cooling"]
        init_dist = ds.parameters["Hydro/init_grainsize_distribution"]
        gamma = ds.parameters["Hydro/AdiabaticIndex"]
        frac_err_dict.update(ds.parameters)
        
        
        
        abins = dust_grainsize_bin_edges_microM

        prim_carbonaceous_dust_fields_Ni = [f for f in ds.derived_field_list if "prim_dust" in f[1] and "number_density" in f[1] and "carbonaceous" in f[1]]
        prim_carbonaceous_dust_fields_Ni = sorted(prim_carbonaceous_dust_fields_Ni, key=lambda f: int(re.search(r"(\d+)$", f[1]).group(1))) ## sort numerically             
        prim_carbonaceous_dust_fields_Mi = [f for f in ds.derived_field_list if "prim_dust" in f[1] and "mass_density" in f[1] and "carbonaceous" in f[1]]
        prim_carbonaceous_dust_fields_Mi = sorted(prim_carbonaceous_dust_fields_Mi, key=lambda f: int(re.search(r"(\d+)$", f[1]).group(1))) ## sort numerically         
        cons_carbonaceous_dust_fields_Ni = [f for f in ds.derived_field_list if "cons_dust" in f[1] and "number_density" in f[1] and "carbonaceous" in f[1]]
        cons_carbonaceous_dust_fields_Ni = sorted(cons_carbonaceous_dust_fields_Ni, key=lambda f: int(re.search(r"(\d+)$", f[1]).group(1))) ## sort numerically             
        cons_carbonaceous_dust_fields_Mi = [f for f in ds.derived_field_list if "cons_dust" in f[1] and "mass_density" in f[1] and "carbonaceous" in f[1]]
        cons_carbonaceous_dust_fields_Mi = sorted(cons_carbonaceous_dust_fields_Mi, key=lambda f: int(re.search(r"(\d+)$", f[1]).group(1))) ## sort numerically         

        prim_silicate_dust_fields_Ni = [f for f in ds.derived_field_list if "prim_dust" in f[1] and "number_density" in f[1] and "silicate" in f[1]]
        prim_silicate_dust_fields_Ni = sorted(prim_silicate_dust_fields_Ni, key=lambda f: int(re.search(r"(\d+)$", f[1]).group(1))) ## sort numerically             
        prim_silicate_dust_fields_Mi = [f for f in ds.derived_field_list if "prim_dust" in f[1] and "mass_density" in f[1] and "silicate" in f[1]]
        prim_silicate_dust_fields_Mi = sorted(prim_silicate_dust_fields_Mi, key=lambda f: int(re.search(r"(\d+)$", f[1]).group(1))) ## sort numerically         
        cons_silicate_dust_fields_Ni = [f for f in ds.derived_field_list if "cons_dust" in f[1] and "number_density" in f[1] and "silicate" in f[1]]
        cons_silicate_dust_fields_Ni = sorted(cons_silicate_dust_fields_Ni, key=lambda f: int(re.search(r"(\d+)$", f[1]).group(1))) ## sort numerically             
        cons_silicate_dust_fields_Mi = [f for f in ds.derived_field_list if "cons_dust" in f[1] and "mass_density" in f[1] and "silicate" in f[1]]
        cons_silicate_dust_fields_Mi = sorted(cons_silicate_dust_fields_Mi, key=lambda f: int(re.search(r"(\d+)$", f[1]).group(1))) ## sort numerically         



        dust_grainsize_bin_edges_microM = ds.parameters["Hydro/dust_grainsize_bin_edges_microM"] 
        dust_grain_midbin_sizes_microM = ds.parameters["Hydro/dust_grain_midbin_sizes_microM"]


        simulation_time_integ_idx = 0
        simulation_time_integ_label = None
        
        
        y = []
        x = dust_grain_midbin_sizes_microM
        ad = ds.all_data()
        mass_values = ad[("gas", "mass")].to(code_mass).value
        for j in range(len( prim_carbonaceous_dust_fields_Ni)):
            thisfield = prim_carbonaceous_dust_fields_Ni[j]
            y.append(np.sum(ad[thisfield][:].value * mass_values))
            
        simulation_total_numgrains_C = np.sum(y)

        y = []
        x = dust_grain_midbin_sizes_microM
        for j in range(len( prim_carbonaceous_dust_fields_Mi)):
            thisfield = prim_carbonaceous_dust_fields_Mi[j]
            y.append(np.sum(ad[thisfield][:].value * mass_values))
        simulation_total_mass_C = np.sum(y)

             
        y = []
        ad = ds.all_data()
        mass_values = ad[("gas", "mass")].to(code_mass).value
        for j in range(len( prim_silicate_dust_fields_Ni)):
            thisfield = prim_carbonaceous_dust_fields_Ni[j]
            y.append(np.sum(ad[thisfield][:].value * mass_values))
        simulation_total_numgrains_S = np.sum(y)

        y = []
        x = dust_grain_midbin_sizes_microM
        for j in range(len( prim_silicate_dust_fields_Mi)):
            thisfield = prim_carbonaceous_dust_fields_Mi[j]
            y.append(np.sum(ad[thisfield][:].value * mass_values))
        simulation_total_mass_S = np.sum(y)   
                
                
        mw_temp = np.sum(ad["gas","temperature"].to("K").value * ad["gas","cell_mass"].to("code_mass").value) / np.sum(ad["gas","cell_mass"].to("code_mass").value)
        
        
        if not do_hists:
            del ds
            del ad
            gc.collect()
        
    
        data = np.load(onezone_data_path + "/" + one_zone_file)  # remove mmap_mode
        a_array = np.array(data["a_array"])  # force into RAM
        comp_types = np.array(data["comp_types"])
        # a_array = data["a_array"]
        T_gas_current = data["T_gas_current"]
        current_t_Myr = data["current_t_Myr"]

        # onezone_times_arr_Myr = data["onezone_times_arr_Myr"]
        # total_mass_g_arr = data["total_mass_g_arr"]
        # summed_cooling_rates_per_grain_arr = data["summed_cooling_rates_per_grain_arr"]
        # particle_masses = data["particle_masses"]
        onezone_temperatures = data["onezone_temperatures"]
        onezone_total_masses = data["onezone_total_masses"]
        onezone_total_numgrains = data["onezone_total_numgrains"]
        
        particle_masses_C = 4. * np.pi * (1/3.) * a_array[comp_types==0]**3 * grain_density_carbonaceous_gmicroM3   
        particle_masses_S = 4. * np.pi * (1/3.) * a_array[comp_types==1]**3 * grain_density_silicates_gmicroM3   
        particle_masses = np.concatenate((particle_masses_C , particle_masses_S))

        norm_onezone_total_mass_C    = np.sum(particle_masses_C)
        norm_simulation_total_mass_C = simulation_total_mass_C * norm_factor_Mi
        
        norm_onezone_total_numgrains_C    = np.size(particle_masses_C)
        norm_simulation_total_numgrains_C = simulation_total_numgrains_C * norm_factor_Ni
    
        norm_onezone_total_mass_S    = np.sum(particle_masses_S)
        norm_simulation_total_mass_S = simulation_total_mass_S * norm_factor_Mi
        
        norm_onezone_total_numgrains_S    = np.size(particle_masses_S)
        norm_simulation_total_numgrains_S = simulation_total_numgrains_S * norm_factor_Ni
    
        plt.figure(fig_frac_err_vs_onezone)
        if len(onezone_temperatures) > 0:
            frac_e_T_vs_onezone = np.abs(mw_temp - onezone_temperatures[-1])/onezone_temperatures[-1]
        else: frac_e_T_vs_onezone = np.nan
        frac_e_M_C_vs_onezone = np.abs(norm_onezone_total_mass_C - norm_simulation_total_mass_C)/norm_onezone_total_mass_C
        frac_e_N_C_vs_onezone = np.abs(norm_onezone_total_numgrains_C - norm_simulation_total_numgrains_C)/norm_onezone_total_numgrains_C
        
        frac_e_M_S_vs_onezone = np.abs(norm_onezone_total_mass_S - norm_simulation_total_mass_S)/norm_onezone_total_mass_S
        frac_e_N_S_vs_onezone = np.abs(norm_onezone_total_numgrains_S - norm_simulation_total_numgrains_S)/norm_onezone_total_numgrains_S
        
        
        frac_err_dict["frac_e_M_C_vs_onezone"] = frac_e_M_C_vs_onezone
        frac_err_dict["frac_e_N_C_vs_onezone"] = frac_e_N_C_vs_onezone
        frac_err_dict["frac_e_M_S_vs_onezone"] = frac_e_M_S_vs_onezone
        frac_err_dict["frac_e_N_S_vs_onezone"] = frac_e_N_S_vs_onezone
        frac_err_dict["frac_e_T_vs_onezone"]   = frac_e_T_vs_onezone
        
        
        if do_hists: ### var_i == 1: means only do once
            os.makedirs(sims_parent_dir+"/hists/", exist_ok=True)

            hist_values, bin_edges =  np.histogram(a_array[comp_types==0], bins = abins,density = False)
            y = []
            x = dust_grain_midbin_sizes_microM
            mass_values = ad[("gas", "mass")].to(code_mass).value
            for j in range(len( prim_carbonaceous_dust_fields_Ni)):
                thisfield = prim_carbonaceous_dust_fields_Ni[j]
                y.append(np.sum(ad[thisfield][:].value * mass_values))
            figure_Ni_dist              = plt.figure(figsize = my_figsize)
            
            bin_centers = 0.5 * (bin_edges[:-1] +bin_edges[1:])
            if(plot_style == "hist"):
            
                plt.bar(x, y, width = np.diff(dust_grainsize_bin_edges_microM),     
                facecolor="none",    # transparent fill
                edgecolor="palevioletred",   # black outline
                linewidth=2.5,        # outline thickness
                alpha = 0.9,
                label = r"$\textsc{AthenaPK}$"
                )
                plt.bar(bin_centers, hist_values / norm_factor_Ni, width = np.diff(dust_grainsize_bin_edges_microM),     
                facecolor="none",    # transparent fill
                edgecolor="skyblue",   # black outline
                linewidth=2.5,        # outline thickness
                alpha = 0.5,
                label = "onezone"
                # label = f"1Zone(RKtol={onezone_RK45_tol}_N={onezone_N_particles})"
                )
                
            if(plot_style == "line"):
            
                plt.plot(x, y,     
                color="palevioletred",   # black outline
                linewidth=2.5,        # outline thickness
                alpha = 1,
                label = r"$\textsc{AthenaPK}$"
                )
                plt.plot(bin_centers, hist_values / norm_factor_Ni,    
                color="skyblue",   # black outline
                linewidth=2.5,        # outline thickness
                alpha = 1,
                label = "onezone"
                # label = f"1Zone(RKtol={onezone_RK45_tol}_N={onezone_N_particles})"
                )
            plt.yscale("log")
            plt.ylabel(r"N$_i$")
            plt.xlabel(r"log$_{10}$(a)")
            plt.xscale("log")
            plt.tight_layout()
            plt.legend()
            mysavefig(f"{sims_parent_dir}/hists/hist_{current_time_string}_{snapshot_number}_N_vs_a_carbonaceous_{plot_style}.png")
            plt.close()
            
            
            
            
            hist_values, bin_edges =  np.histogram(a_array[comp_types==1], bins = abins,density = False)
            y = []
            x = dust_grain_midbin_sizes_microM
            mass_values = ad[("gas", "mass")].to(code_mass).value
            for j in range(len( prim_silicate_dust_fields_Ni)):
                thisfield = prim_silicate_dust_fields_Ni[j]
                y.append(np.sum(ad[thisfield][:].value * mass_values))
            figure_Ni_dist              = plt.figure(figsize = my_figsize)

            
            bin_centers = 0.5 * (bin_edges[:-1] +bin_edges[1:])
            
            if(plot_style == "hist"):
            
                plt.bar(x, y, width = np.diff(dust_grainsize_bin_edges_microM),     
                facecolor="none",    # transparent fill
                edgecolor="palevioletred",   # black outline
                linewidth=2.5,        # outline thickness
                alpha = 0.9,
                label = r"$\textsc{AthenaPK}$"
                )
                plt.bar(bin_centers, hist_values / norm_factor_Ni, width = np.diff(dust_grainsize_bin_edges_microM),     
                facecolor="none",    # transparent fill
                edgecolor="skyblue",   # black outline
                linewidth=2.5,        # outline thickness
                alpha = 0.5,
                label = "onezone"
                # label = f"1Zone(RKtol={onezone_RK45_tol}_N={onezone_N_particles})"
                )
                
            if(plot_style == "line"):
            
                plt.plot(x, y,     
                color="palevioletred",   # black outline
                linewidth=2.5,        # outline thickness
                alpha = 1,
                label = r"$\textsc{AthenaPK}$"
                )
                plt.plot(bin_centers, hist_values / norm_factor_Ni,    
                color="skyblue",   # black outline
                linewidth=2.5,        # outline thickness
                alpha = 1,
                label = "onezone"
                # label = f"1Zone(RKtol={onezone_RK45_tol}_N={onezone_N_particles})"
                )

            plt.yscale("log")
            plt.ylabel(r"N$_i$")
            plt.xlabel(r"log$_{10}$(a)")
            plt.xscale("log")
            plt.tight_layout()
            plt.legend()
            mysavefig(f"{sims_parent_dir}/hists/hist_{current_time_string}_{snapshot_number}_N_vs_a_silicates_{plot_style}.png")
            plt.close()
            
            # figure_Ni_dist              = plt.figure(figsize = my_figsize)
            # hist_values, bin_edges =  np.histogram(a_array, bins = abins,density = True)
            # hist_values *= np.size(a_array)
            # plt.bar(x, y/np.diff(dust_grainsize_bin_edges_microM), width = np.diff(dust_grainsize_bin_edges_microM),     
            # facecolor="none",    # transparent fill
            # edgecolor="palevioletred",   # black outline
            # linewidth=2.5,        # outline thickness
            # alpha = 0.9,
            # label = r"$\textsc{AthenaPK}$"
            # )
            
            # bin_centers = 0.5 * (bin_edges[:-1] +bin_edges[1:])
            # plt.bar(bin_centers, hist_values / norm_factor_Ni, width = np.diff(dust_grainsize_bin_edges_microM),     
            # facecolor="none",    # transparent fill
            # edgecolor="skyblue",   # black outline
            # linewidth=2.5,        # outline thickness
            # alpha = 0.5,
            # label = f"1Zone(RKtol={onezone_RK45_tol}_N={onezone_N_particles})"
            # )
            # plt.yscale("log")
            # plt.ylabel(r"N$_i$")
            # plt.xlabel(r"log$_{10}$(a)")
            # plt.xscale("log")
            # plt.tight_layout()
            # plt.legend()
            # print(sim_file)
            # mysavefig(f"./Testing_Dust_Model_vs_OneZone/histogram_test_niceplots_{tag_to_match}/Ni_hist_{current_time_string}_Nbins={num_grain_size}_{sim_file.split('/')[-2]}_{save_tag}_dnda_vs_a.png")
            # plt.close()
            
            
            
            
            
            hist_values, bin_edges =  np.histogram(a_array[comp_types==0], bins = abins, weights = particle_masses[comp_types==0], density = False)
            y = []
            x = dust_grain_midbin_sizes_microM
            for j in range(len( prim_carbonaceous_dust_fields_Mi)):
                thisfield = prim_carbonaceous_dust_fields_Mi[j]
                y.append(np.sum(ad[thisfield][:].value * ad["gas","mass"].to(code_mass).value))
        

            figure_Ni_dist              = plt.figure(figsize = my_figsize)
            bin_centers = 0.5 * (bin_edges[:-1] +bin_edges[1:])
            
            if(plot_style == "hist"):
            
                plt.bar(x, y, width = np.diff(dust_grainsize_bin_edges_microM),     
                facecolor="none",    # transparent fill
                edgecolor="palevioletred",   # black outline
                linewidth=2.5,        # outline thickness
                alpha = 0.9,
                label = r"$\textsc{AthenaPK}$"
                )
                plt.bar(bin_centers, hist_values / norm_factor_Mi, width = np.diff(dust_grainsize_bin_edges_microM),     
                facecolor="none",    # transparent fill
                edgecolor="skyblue",   # black outline
                linewidth=2.5,        # outline thickness
                alpha = 0.5,
                label = "onezone"
                # label = f"1Zone(RKtol={onezone_RK45_tol}_N={onezone_N_particles})"
                )
                
            if(plot_style == "line"):
            
                plt.plot(x, y,     
                color="palevioletred",   # black outline
                linewidth=2.5,        # outline thickness
                alpha = 1,
                label = r"$\textsc{AthenaPK}$"
                )
                plt.plot(bin_centers, hist_values / norm_factor_Mi,    
                color="skyblue",   # black outline
                linewidth=2.5,        # outline thickness
                alpha = 1,
                label = "onezone"
                # label = f"1Zone(RKtol={onezone_RK45_tol}_N={onezone_N_particles})"
                )
                
            plt.yscale("log")
            plt.ylabel(r"M$_i$")
            plt.xlabel(r"log$_{10}$(a)")
            plt.xscale("log")
            plt.tight_layout()
            plt.legend()
            print(sim_file)
            mysavefig(f"{sims_parent_dir}/hists/hist_{current_time_string}_{snapshot_number}_M_vs_a_carbonaceous_{plot_style}.png")
            plt.close()
            
            
            hist_values, bin_edges =  np.histogram(a_array[comp_types==1], bins = abins, weights = particle_masses[comp_types==1], density = False)
            y = []
            x = dust_grain_midbin_sizes_microM
            for j in range(len( prim_silicate_dust_fields_Mi)):
                thisfield = prim_silicate_dust_fields_Mi[j]
                y.append(np.sum(ad[thisfield][:].value * ad["gas","mass"].to(code_mass).value))
        
            figure_Ni_dist              = plt.figure(figsize = my_figsize)
            bin_centers = 0.5 * (bin_edges[:-1] +bin_edges[1:])

            if(plot_style == "hist"):
            
                plt.bar(x, y, width = np.diff(dust_grainsize_bin_edges_microM),     
                facecolor="none",    # transparent fill
                edgecolor="palevioletred",   # black outline
                linewidth=2.5,        # outline thickness
                alpha = 0.9,
                label = r"$\textsc{AthenaPK}$"
                )
                plt.bar(bin_centers, hist_values / norm_factor_Mi, width = np.diff(dust_grainsize_bin_edges_microM),     
                facecolor="none",    # transparent fill
                edgecolor="skyblue",   # black outline
                linewidth=2.5,        # outline thickness
                alpha = 0.5,
                label = "onezone"
                # label = f"1Zone(RKtol={onezone_RK45_tol}_N={onezone_N_particles})"
                )
                
            if(plot_style == "line"):
            
                plt.plot(x, y,     
                color="palevioletred",   # black outline
                linewidth=2.5,        # outline thickness
                alpha = 1,
                label = r"$\textsc{AthenaPK}$"
                )
                plt.plot(bin_centers, hist_values / norm_factor_Mi,    
                color="skyblue",   # black outline
                linewidth=2.5,        # outline thickness
                alpha = 1,
                label = "onezone"
                # label = f"1Zone(RKtol={onezone_RK45_tol}_N={onezone_N_particles})"
                )
                
            plt.yscale("log")
            plt.ylabel(r"M$_i$")
            plt.xlabel(r"log$_{10}$(a)")
            plt.xscale("log")
            plt.tight_layout()
            plt.legend()
            print(sim_file)
            mysavefig(f"{sims_parent_dir}/hists/hist_{current_time_string}_{snapshot_number}_M_vs_a_silicates_{plot_style}.png")
            plt.close()

                # hist_values, bin_edges =  np.histogram(a_array, bins = abins, weights = particle_masses, density = True)
                # hist_values *= np.sum(particle_masses)
                # figure_Ni_dist              = plt.figure(figsize = my_figsize)
                # plt.bar(x, y/np.diff(dust_grainsize_bin_edges_microM), width = np.diff(dust_grainsize_bin_edges_microM),     
                # facecolor="none",    # transparent fill
                # edgecolor="palevioletred",   # black outline
                # linewidth=2.5,        # outline thickness
                # alpha = 1,
                # label = r"$\textsc{AthenaPK}$"
                # )
                # bin_centers = 0.5 * (bin_edges[:-1] +bin_edges[1:])
                # plt.bar(bin_centers, hist_values / norm_factor_Mi, width = np.diff(dust_grainsize_bin_edges_microM),     
                # facecolor="none",    # transparent fill
                # edgecolor="skyblue",   # black outline
                # linewidth=2.5,        # outline thickness
                # alpha = 0.8,
                # label = f"1Zone(RKtol={onezone_RK45_tol}_N={onezone_N_particles})"
                # )
                # plt.yscale("log")
                # plt.ylabel(r"M$_i$")
                # plt.xlabel(r"log$_{10}$(a)")
                # plt.xscale("log")
                # plt.tight_layout()
                # plt.legend()
                # print(sim_file)
                # mysavefig(f"./Testing_Dust_Model_vs_OneZone/histogram_test_niceplots_{tag_to_match}/Mi_hist_{current_time_string}_Nbins={num_grain_size}_{sim_file.split('/')[-2]}_{save_tag}_dnda_vs__a.png")
                # plt.close()
                
                
                
                
    
            

    return norm_factor_Ni, norm_factor_Mi, frac_err_dict