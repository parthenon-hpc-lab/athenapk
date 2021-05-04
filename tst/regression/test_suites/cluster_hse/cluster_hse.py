#========================================================================================
# AthenaPK - a performance portable block structured AMR MHD code
# Copyright (c) 2020-2021, Athena Parthenon Collaboration. All rights reserved.
# Licensed under the 3-clause BSD License, see LICENSE file for details
#========================================================================================
# (C) (or copyright) 2020. Triad National Security, LLC. All rights reserved.
#
# This program was produced under U.S. Government contract 89233218CNA000001 for Los
# Alamos National Laboratory (LANL), which is operated by Triad National Security, LLC
# for the U.S. Department of Energy/National Nuclear Security Administration. All rights
# in the program are reserved by Triad National Security, LLC, and the U.S. Department
# of Energy/National Nuclear Security Administration. The Government is granted for
# itself and others acting on its behalf a nonexclusive, paid-up, irrevocable worldwide
# license in this material to reproduce, prepare derivative works, distribute copies to
# the public, perform publicly and display publicly, and to permit others to do so.
#========================================================================================

# Modules
import math
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pylab as plt
import sys
import os
import utils.test_case
import utils.compare_analytic as compare_analytic
import unyt

""" To prevent littering up imported folders with .pyc files or __pycache_ folder"""
sys.dont_write_bytecode = True


class TestCase(utils.test_case.TestCaseAbs):
    def __init__(self):

        #Define cluster parameters
        #Setup units
        unyt.define_unit("code_length",(1,"Mpc"))
        unyt.define_unit("code_mass",(1e14,"Msun"))
        unyt.define_unit("code_time",(1,"Gyr"))
        self.code_length = unyt.unyt_quantity(1,"code_length")
        self.code_mass = unyt.unyt_quantity(1,"code_mass")
        self.code_time = unyt.unyt_quantity(1,"code_time")

        #Setup constants
        self.k_b = unyt.kb_cgs
        self.G = unyt.G_cgs
        self.m_u =unyt.amu

        self.adiabatic_index = 5./3.

        self.hubble_parameter = unyt.unyt_quantity(70,"km*s**-1*Mpc**-1")

        #Which gravitational fields to include
        self.include_nfw_g = True
        self.which_bcg_g = "HERNQUIST"
        self.include_smbh_g = True

        #NFW parameters
        self.c_nfw = 6.0
        self.M_nfw_200 = unyt.unyt_quantity(1e15,"Msun")

        #BCG parameters
        self.M_bcg_s = unyt.unyt_quantity(1e11,"Msun")
        self.R_bcg_s = unyt.unyt_quantity(4,"kpc")

        #SMBH parameters
        self.M_smbh = unyt.unyt_quantity(1e8,"Msun")

        #Smooth gravity at origin, for numerical reasons
        self.g_smoothing_radius = unyt.unyt_quantity(0.0,"code_length")

        #Entropy profile parameters
        self.K_0 = unyt.unyt_quantity(10,"keV*cm**2")
        self.K_100 = unyt.unyt_quantity(150,"keV*cm**2")
        self.R_K = unyt.unyt_quantity(100,"kpc")
        self.alpha_K = 1.1

        self.He_mass_fraction = 0.25

        #Fix density at radius to close system of equations
        self.R_fix = unyt.unyt_quantity(2e3,"kpc")
        self.rho_fix = unyt.unyt_quantity(1e-28,"g*cm**-3")

        #Building the radii at which to sample initial rho,P
        self.R_sampling = 4.0
        self.max_dR = 0.001

        self.R_min = unyt.unyt_quantity(1e-3,"kpc")
        self.R_max = unyt.unyt_quantity(5e3,"kpc")
        
        self.norm_tol = 1e-3

    def Prepare(self,parameters, step):
        """
        Any preprocessing that is needed before the drive is run can be done in
        this method

        This includes preparing files or any other pre processing steps that
        need to be implemented.  The method also provides access to the
        parameters object which controls which parameters are being used to run
        the driver. 

        It is possible to append arguments to the driver_cmd_line_args if it is
        desired to  override the parthenon input file. Each element in the list
        is simply a string of the form '<block>/<field>=<value>', where the
        contents of the string are exactly what one would type on the command
        line run running a parthenon driver.

        As an example if the following block was uncommented it would overwrite
        any of the parameters that were specified in the parthenon input file
        parameters.driver_cmd_line_args = ['output1/file_type=vtk',
                'output1/variable=cons',
                'output1/dt=0.4',
                'time/tlim=0.4',
                'mesh/nx1=400']
        """

        parameters.driver_cmd_line_args = [
                f"hydro/gamma={self.adiabatic_index}",
                f"problem/code_length_cgs={self.code_length.in_units('cm').v}",
                f"problem/code_mass_cgs={self.code_mass.in_units('g').v}",
                f"problem/code_time_cgs={self.code_time.in_units('s').v}",
                f"problem/hubble_parameter={self.hubble_parameter.in_units('1/code_time').v}",
                f"problem/include_nfw_g={self.include_nfw_g}",
                f"problem/which_bcg_g={self.which_bcg_g}",
                f"problem/include_smbh_g={self.include_smbh_g}",
                f"problem/c_nfw={self.c_nfw}",
                f"problem/M_nfw_200={self.M_nfw_200.in_units('code_mass').v}",
                f"problem/M_bcg_s={self.M_bcg_s.in_units('code_mass').v}",
                f"problem/R_bcg_s={self.R_bcg_s.in_units('code_length').v}",
                f"problem/M_smbh={self.M_smbh.in_units('code_mass').v}",
                f"problem/g_smoothing_radius={self.g_smoothing_radius.in_units('code_length').v}",
                f"problem/K_0={self.K_0.in_units('code_length**4*code_mass/code_time**2').v}",
                f"problem/K_100={self.K_100.in_units('code_length**4*code_mass/code_time**2').v}",
                f"problem/R_K={self.R_K.in_units('code_length').v}",
                f"problem/alpha_K={self.alpha_K}",
                f"problem/He_mass_fraction={self.He_mass_fraction}",
                f"problem/R_fix={self.R_fix.in_units('code_length').v}",
                f"problem/rho_fix={self.rho_fix.in_units('code_mass/code_length**3').v}",
                f"problem/R_sampling={self.R_sampling}",
                f"problem/max_dR={self.max_dR}",
            ]


        return parameters

    def Analyse(self,parameters):
        """
        Analyze the output and determine if the test passes.

        This function is called after the driver has been executed. It is
        responsible for reading whatever data it needs and making a judgment
        about whether or not the test passes. It takes no inputs. Output should
        be True (test passes) or False (test fails).  

        The parameters that are passed in provide the paths to relevant
        locations and commands. Of particular importance is the path to the
        output folder. All files from a drivers run should appear in and output
        folder located in
        parthenon/tst/regression/test_suites/test_name/output.

        It is possible in this function to read any of the output files such as
        hdf5 output and compare them to expected quantities.

        """
        analyze_status = True

        self.Yp = self.He_mass_fraction
        self.mu = 1/(self.Yp*3./4. + (1-self.Yp)*2)
        self.mu_e = 1/(self.Yp*2./4. + (1-self.Yp))

        self.include_gs = []
        if(self.include_nfw_g):
            self.include_gs.append("nfw")
        if(self.which_bcg_g == "HERNQUIST"):
            self.include_gs.append("bcg_hernquist")
        if(self.include_smbh_g):
            self.include_gs.append("smbh")

        self.rho_crit = 3*self.hubble_parameter**2/(8*np.pi*self.G)
        self.rho_nfw_0 = 200./3*self.rho_crit*self.c_nfw**3/(np.log(1 + self.c_nfw) - self.c_nfw/(1 + self.c_nfw))
        self.R_nfw_s = (self.M_nfw_200/(4*np.pi*self.rho_nfw_0*(np.log(1+self.c_nfw)-self.c_nfw/(1+self.c_nfw))))**(1./3.)

        #Thermodynamics+Entropy helpers
        def P_from_rho_K(rho,K):
            return K*(self.mu/self.mu_e)**(2./3.)*(rho/(self.mu*self.m_u))**(5./3.)
        def rho_from_P_K(P,K):
            return (P/K)**(3./5.)*self.mu*self.m_u/(self.mu/self.mu_e)**(2./5.)
        def K_from_r(r):
            return self.K_0 + self.K_100*(r/self.R_K)**self.alpha_K
        def n_from_rho(rho):
            return rho/(self.mu*self.m_u)
        def ne_from_rho(rho):
            return self.mu/self.mu_e*n_from_rho(rho)
        def T_from_rho_P(rho,P):
            return P/(n_from_rho(rho)*self.k_b)

        #Gravity helpers
        def g_nfw_from_r(r):
            return self.G * self.M_nfw_200/( np.log(1 + self.c_nfw) - self.c_nfw/(1+self.c_nfw))*( np.log(1 + r/self.R_nfw_s) - r/(r+self.R_nfw_s))/r**2
        
        def g_bcg_hernquist_from_r(r):
            #M_bcg = 8*self.M_bcg_s*(r/self.R_bcg_s)**2/( 2*( 1 + r/self.R_bcg_s)**2)
            #return G*M_bcg/r**2
            g = self.G*self.M_bcg_s/(self.R_bcg_s**2)/( 2*( 1 + r/self.R_bcg_s)**2)
            return g

        def g_smbh_from_r(r):
            return self.G*self.M_smbh/r**2

        def g_from_r(r,include_gs):
            g = unyt.unyt_array(np.zeros_like(r),"code_length*code_time**-2")

            if "nfw" in include_gs:
                g += g_nfw_from_r(r)
            if "bcg_hernquist" in include_gs:
                g += g_bcg_hernquist_from_r(r)
            if "smbh" in include_gs:
                g += g_smbh_from_r(r)
            return g

        #Pressure function to integrate
        def dP_dr(r,P):
            return -rho_from_P_K(P,K_from_r(r))*g_from_r(r,self.include_gs)

        #RK4 Integrator
        def rk4(f,y0,T):
            y = np.zeros(T.size)*y0
            y[0] = y0
            for i in np.arange(T.size-1):
                h = T[i+1] - T[i]
                k1 = f(T[i]     ,y[i]          )
                k2 = f(T[i]+h/2.,y[i] + h/2.*k1)
                k3 = f(T[i]+h/2.,y[i] + h/2.*k2)
                k4 = f(T[i]+h   ,y[i] + h*k3   )
                y[i+1] = y[i] + h/6.*(k1 + 2*k2 + 2*k3 + k4)
            return y

        

        #Defining the R mesh to integrate along
        R = unyt.unyt_array(np.geomspace(self.R_min.in_units("code_length"),self.R_max.in_units("code_length"),4000),"code_length")

        #Calculate an entropy and pressure at the fixed radius
        K_fix = K_from_r(self.R_fix)
        P_fix = P_from_rho_K(self.rho_fix,K_fix)

        #Prepare two array of R leading inwards and outwards from the virial radius, to integrate 
        R_in  = unyt.unyt_array(np.hstack((R[R < self.R_fix],self.R_fix.in_units("code_length"))),"code_length")
        R_out = unyt.unyt_array(np.hstack((self.R_fix.in_units("code_length"),R[R > self.R_fix])),"code_length")

        #Integrate in from self.R_fix
        P_in = rk4(dP_dr,P_fix,R_in[::-1])
        P_in = P_in[::-1]#Flip P to match R

        #Integrate out from P
        P_out = rk4(dP_dr,P_fix,R_out)

        #Put the two pieces of P together
        P = unyt.unyt_array(np.hstack((P_in[:-1].in_units("dyne/cm**2"),P_out[1:].in_units("dyne/cm**2"))),"dyne/cm**2")

        K = K_from_r(R)
        rho = rho_from_P_K(P,K)

        T = T_from_rho_P(rho,P)

        g = g_from_r(R,self.include_gs)

        gs = {include_g: g_from_r(R,(include_g,)) for include_g in self.include_gs}
        Ms = {include_g: gs[include_g]*R**2/self.G for include_g in self.include_gs}

        #The analytic pressure and density profiles
        analytic_R = R.in_units("code_length")
        analytic_P = P.in_units("code_mass/(code_length*code_time**2)")
        analytic_K = K.in_units("code_length**4*code_mass*code_time**-2")
        analytic_rho =rho.in_units("code_mass/code_length**3")
        analytic_g = g.in_units("code_length*code_time**-2")

        #Check that test_he_sphere.dat matches python model
        try:
            he_sphere_data = np.loadtxt(f"{parameters.output_path}/test_he_sphere.dat")
            he_sphere_R     = unyt.unyt_array(he_sphere_data[1:,0],"code_length")
            he_sphere_P     = unyt.unyt_array(he_sphere_data[1:,1],"code_mass/(code_length*code_time**2)")
            he_sphere_K     = unyt.unyt_array(he_sphere_data[1:,2],"code_length**4*code_mass*code_time**-2")
            he_sphere_rho   = unyt.unyt_array(he_sphere_data[1:,3],"code_mass*code_length**-3")
            he_sphere_n     = unyt.unyt_array(he_sphere_data[1:,4],"code_length**-3")
            he_sphere_ne    = unyt.unyt_array(he_sphere_data[1:,5],"code_length**-3")
            he_sphere_T     = unyt.unyt_array(he_sphere_data[1:,6],"K")
            he_sphere_g     = unyt.unyt_array(he_sphere_data[1:,7],"code_length*code_time**-2")
            he_sphere_dP_dr = unyt.unyt_array(he_sphere_data[1:,8],"code_mass/(code_length**2*code_time**2)")
        except IOError:
            print("test_he_sphere.dat file not accessible")
            return False

        profile_comparison_vars = ((analytic_P,   he_sphere_P,   "Pressure"),
                                   (analytic_K,   he_sphere_K,   "Entropy"),
                                   (analytic_rho, he_sphere_rho, "Density"),
                                   (analytic_g,   he_sphere_g,   "Gravity") )

        fig,axes = plt.subplots(2,2)

        for ax,(analytic_var,he_sphere_var,label) in zip(axes.flatten(),profile_comparison_vars):
            #Interpolate analytic profile to the same radii as he_sphere
            analytic_interp = unyt.unyt_array(np.interp(he_sphere_R,analytic_R,analytic_var),analytic_var.units)

            #Compute relative error at he_sphere_R
            rel_err = np.abs((analytic_interp - he_sphere_var)/analytic_interp)

            norm_rel_err = np.linalg.norm(rel_err[1:-1])/rel_err.size

            if norm_rel_err > self.norm_tol:
                analyze_status = False
                print(f"test_he_sphere.dat field {label} relative error {norm_rel_err} exceeds tolerance {self.norm_tol}")

            #Plot the analytic profile and test_he_sphere.dat profile
            ax.plot(he_sphere_R,he_sphere_var,label="test_he_sphere.dat")
            ax.plot(he_sphere_R,analytic_interp,label="Analytic Interpolation",linestyle="--")

            ax.set_ylabel(label)
            ax.set_xscale("log")
            ax.set_yscale("log")

        ax = axes[1,1]
        for include_g in gs.keys():
            ax.plot(R,gs[include_g].in_units("code_length*code_time**-2"),label=include_g,linestyle=":")

        axes[0,0].legend()
        plt.tight_layout()
        plt.savefig(f"{parameters.output_path}/analytic_comparison.png")

        #Check that the initial and final outputs match
        sys.path.insert(1, parameters.parthenon_path + '/scripts/python')

        try:
            import phdf_diff
        except ModuleNotFoundError:
            print("Couldn't find module to compare Parthenon hdf5 files.")
            return False

        files = [f"{parameters.output_path}/parthenon.prim.{i:05d}.phdf" for i in range(2)]

        #Compare the initial output to the analytic model
        def analytic_gold(X,Y,Z,analytic_var):
            r = np.sqrt(X**2 + Y**2 + Z**2)
            analytic_interp = unyt.unyt_array(np.interp(r,analytic_R,analytic_var),
                                                analytic_var.units)
            return analytic_interp

        analytic_components = {
                "Pressure":lambda X,Y,Z,time : analytic_gold(X,Y,Z,analytic_P).v,
                "Density":lambda X,Y,Z,time : analytic_gold(X,Y,Z,analytic_rho).v,
                }

        #Use a very loose tolerance, linf relative error
        analytic_status = compare_analytic.compare_analytic(
                files[0], analytic_components,err_func=compare_analytic.linf_rel_err,tol=1e-1)

        print(analytic_status,analyze_status)

        analyze_status &= (analytic_status)

        #Due to HSE, initial and final outputs should match, within a loose tolerance
        compare_status = phdf_diff.compare(files,check_metadata=False,tol=5e-2,relative=True)

        analyze_status &= (compare_status==0)

        return analyze_status
