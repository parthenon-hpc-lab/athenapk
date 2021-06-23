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
import itertools


class PrecessedJetCoords:
    def __init__(self,theta,phi):
        self.theta = theta
        self.phi = phi

        #Axis of the jet
        self.jet_n = np.array((np.cos(self.theta)*np.sin(self.phi),
                    np.sin(self.theta)*np.sin(self.phi),
                    np.cos(self.phi)))
        
    def cart_to_rho_h(self,pos_cart):
        """
        Convert from cartesian coordinates to jet coordinates
        """

        pos_h = np.sum(pos_cart*self.jet_n[:,None],axis=0)
        pos_rho = np.linalg.norm( pos_cart - pos_h*self.jet_n[:,None],axis=0)
        return (pos_rho,pos_h)

class ZJetCoords:
    def __init__(self):
        self.jet_n = np.array((0,0,1.0))

    def cart_to_rho_h(self,pos_cart):
        """
        Convert from cartesian coordinates to jet coordinates
        """

        pos_rho = np.linalg.norm(pos_cart[:2])
        pos_h = pos_cart[2]

        return pos_rho,pos_h

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

        self.tlim = unyt.unyt_quantity(5e-3,"code_time")

        #Setup constants
        self.k_b = unyt.kb_cgs
        self.G = unyt.G_cgs
        self.m_u =unyt.amu

        self.adiabatic_index = 5./3.
        self.He_mass_fraction = 0.25

        #Define the initial uniform gas
        self.uniform_gas_rho = unyt.unyt_quantity(1e-24,"g/cm**3")
        self.uniform_gas_ux = unyt.unyt_quantity(0,"cm/s")
        self.uniform_gas_uy = unyt.unyt_quantity(0,"cm/s")
        self.uniform_gas_uz = unyt.unyt_quantity(0,"cm/s")
        self.uniform_gas_pres = unyt.unyt_quantity(1e-10,"dyne/cm**2")

        self.uniform_gas_Mx =  self.uniform_gas_rho*self.uniform_gas_ux
        self.uniform_gas_My =  self.uniform_gas_rho*self.uniform_gas_uy
        self.uniform_gas_Mz =  self.uniform_gas_rho*self.uniform_gas_uz
        self.uniform_gas_energy_density = \
            1./2.*self.uniform_gas_rho*(self.uniform_gas_ux**2 + self.uniform_gas_uy**2 + self.uniform_gas_uz**2) \
            + self.uniform_gas_pres/(self.adiabatic_index - 1.)

        #The precessing jet
        self.jet_phi = 0.2
        self.jet_theta_dot = 0
        self.jet_theta0 = 1
        self.precessed_jet_coords = PrecessedJetCoords(self.jet_theta0,self.jet_phi)
        self.zjet_coords = ZJetCoords()

        #Feedback parameters
        self.agn_power = unyt.unyt_quantity(1e44,"erg/s")
        self.agn_thermal_radius = unyt.unyt_quantity(0.5,"kpc")
        self.agn_jet_efficiency = 1.0e-3
        self.agn_jet_radius = unyt.unyt_quantity(0.25,"kpc")
        self.agn_jet_height = unyt.unyt_quantity(1,"kpc")
        
        self.norm_tol = 1e-3

        self.steps = 4
        self.step_params_list = list(itertools.product(
            ("thermal_only","kinetic_only","combined"),(True,False)))

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
        feedback_mode,precessed_jet = self.step_params_list[step-1]
        output_id = f"{feedback_mode}_precessed_{precessed_jet}"

        if feedback_mode == "thermal_only":
            agn_kinetic_fraction = 0.0
            agn_thermal_fraction = 1.0
        elif feedback_mode == "kinetic_only":
            agn_kinetic_fraction = 1.0
            agn_thermal_fraction = 0.0
        elif feedback_mode == "combined":
            agn_kinetic_fraction = 0.5
            agn_thermal_fraction = 0.5
        else:
            raise Exception(f"Feedback mode {feedback_mode} not supported in analysis")



        parameters.driver_cmd_line_args = [
                f"parthenon/output2/id={output_id}",
                f"parthenon/output2/dt={self.tlim.in_units('code_time').v}",
                f"parthenon/time/tlim={self.tlim.in_units('code_time').v}",
                f"hydro/gamma={self.adiabatic_index}",
                f"hydro/He_mass_fraction={self.He_mass_fraction}",

                f"units/code_length_cgs={self.code_length.in_units('cm').v}",
                f"units/code_mass_cgs={self.code_mass.in_units('g').v}",
                f"units/code_time_cgs={self.code_time.in_units('s').v}",

                f"problem/cluster/init_uniform_gas=true",
                f"problem/cluster/uniform_gas_rho={self.uniform_gas_rho.in_units('code_mass*code_length**-3').v}",
                f"problem/cluster/uniform_gas_ux={self.uniform_gas_ux.in_units('code_length*code_time**-1').v}",
                f"problem/cluster/uniform_gas_uy={self.uniform_gas_uy.in_units('code_length*code_time**-1').v}",
                f"problem/cluster/uniform_gas_uz={self.uniform_gas_uz.in_units('code_length*code_time**-1').v}",
                f"problem/cluster/uniform_gas_pres={self.uniform_gas_pres.in_units('code_mass*code_length**-1*code_time**-2').v}",

                f"problem/cluster/jet_phi={self.jet_phi if precessed_jet else 0}",
                f"problem/cluster/jet_theta_dot={self.jet_theta_dot if precessed_jet else 0}",
                f"problem/cluster/jet_theta0={self.jet_theta0 if precessed_jet else 0}",

                f"problem/cluster/agn_power={self.agn_power.in_units('code_mass*code_length**2/code_time**3').v}",
                f"problem/cluster/agn_thermal_fraction={agn_thermal_fraction}",
                f"problem/cluster/agn_kinetic_fraction={agn_kinetic_fraction}",
                f"problem/cluster/agn_jet_efficiency={self.agn_jet_efficiency}",
                f"problem/cluster/agn_thermal_radius={self.agn_thermal_radius.in_units('code_length').v}",
                f"problem/cluster/agn_jet_radius={self.agn_jet_radius.in_units('code_length').v}",
                f"problem/cluster/agn_jet_height={self.agn_jet_height.in_units('code_length').v}",
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
        

        for step in range(1,self.steps+1):
            feedback_mode,precessed_jet = self.step_params_list[step-1]
            output_id = f"{feedback_mode}_precessed_{precessed_jet}"
            step_status = True

            print(f"Testing {output_id}")

            if precessed_jet is True:
                jet_coords = self.precessed_jet_coords
            else:
                jet_coords = self.zjet_coords

            if feedback_mode == "thermal_only":
                agn_kinetic_fraction = 0.0
                agn_thermal_fraction = 1.0
            elif feedback_mode == "kinetic_only":
                agn_kinetic_fraction = 1.0
                agn_thermal_fraction = 0.0
            elif feedback_mode == "combined":
                agn_kinetic_fraction = 0.5
                agn_thermal_fraction = 0.5
            else:
                raise Exception(f"Feedback mode {feedback_mode} not supported in analysis")

            jet_density = (agn_kinetic_fraction*self.agn_power)/(self.agn_jet_efficiency*unyt.c_cgs**2)/( 
                2*np.pi*self.agn_jet_radius**2*self.agn_jet_height)

            jet_velocity = np.sqrt( 2*self.agn_jet_efficiency)*unyt.c_cgs

            def kinetic_feedback(Z,Y,X,time):
                if not hasattr(time,"units"):
                    time = unyt.unyt_quantity(time,"code_time")
                R,H = jet_coords.cart_to_rho_h(np.array((X,Y,Z)))
                R = unyt.unyt_array(R,"code_length")
                H = unyt.unyt_array(H,"code_length")

                sign_jet = np.piecewise(H,[H <=0, H > 0],[-1,1])
                inside_jet = np.piecewise(R,[ R <= self.agn_jet_radius,],[1,0]) \
                            *np.piecewise(H,[ np.abs(H) <= self.agn_jet_height,],[1,0])

                drho = inside_jet*agn_kinetic_fraction*time*jet_density
                dMx  = inside_jet*agn_kinetic_fraction*time*sign_jet*jet_density*jet_velocity*jet_coords.jet_n[0]
                dMy  = inside_jet*agn_kinetic_fraction*time*sign_jet*jet_density*jet_velocity*jet_coords.jet_n[1]
                dMz  = inside_jet*agn_kinetic_fraction*time*sign_jet*jet_density*jet_velocity*jet_coords.jet_n[2]
                dE   = inside_jet*agn_kinetic_fraction*time*0.5*jet_density*jet_velocity**2

                return drho,dMx,dMy,dMz,dE

            def thermal_feedback(Z,Y,X,time):
                if not hasattr(time,"units"):
                    time = unyt.unyt_quantity(time,"code_time")
                R = np.sqrt( X**2 + Y**2 + Z**2)
                inside_sphere = np.piecewise(R,[ R <= self.agn_thermal_radius.in_units("code_length"),],[1,0])
                dE = inside_sphere*time*(self.agn_power*agn_thermal_fraction/(4./3.*np.pi*self.agn_thermal_radius**3))
                return dE


            def agn_feedback(Z,Y,X,dt):

                drho_k,dMx_k,dMy_k,dMz_k,dE_k = kinetic_feedback(Z,Y,X,dt)
                dE_t = thermal_feedback(Z,Y,X,dt)

                drho = drho_k
                dMx  = dMx_k 
                dMy  = dMx_k 
                dMz  = dMx_k 
                dE   = dE_k + dE_t

                return drho,dMx,dMy,dMz,dE


            #Check that the initial and final outputs match the expected tower
            sys.path.insert(1, parameters.parthenon_path + '/scripts/python/packages/parthenon_tools/parthenon_tools')

            try:
                import phdf_diff
            except ModuleNotFoundError:
                print("Couldn't find module to compare Parthenon hdf5 files.")
                return False

            initial_analytic_components = {
                    "Density":lambda Z,Y,X,time :
                        np.ones_like(Z)*self.uniform_gas_rho.in_units("code_mass/code_length**3").v,
                    "MomentumDensity1":lambda Z,Y,X,time :
                        np.ones_like(Z)*self.uniform_gas_Mx.in_units("code_mass*code_length**-2*code_time**-1").v,
                    "MomentumDensity2":lambda Z,Y,X,time :
                        np.ones_like(Z)*self.uniform_gas_My.in_units("code_mass*code_length**-2*code_time**-1").v,
                    "MomentumDensity3":lambda Z,Y,X,time :
                        np.ones_like(Z)*self.uniform_gas_Mz.in_units("code_mass*code_length**-2*code_time**-1").v,
                    "TotalEnergyDensity":lambda Z,Y,X,time :
                        np.ones_like(Z)*self.uniform_gas_energy_density.in_units("code_mass*code_length**-1*code_time**-2").v,
                    }

            final_analytic_components = {
                    "Density":lambda Z,Y,X,time :
                        ( self.uniform_gas_rho + agn_feedback(Z,Y,X,time)[0]).in_units("code_mass/code_length**3").v,
                    "MomentumDensity1":lambda Z,Y,X,time :
                        ( self.uniform_gas_Mx + agn_feedback(Z,Y,X,time)[1]).in_units("code_mass*code_length**-2*code_time**-1").v,
                    "MomentumDensity2":lambda Z,Y,X,time :
                        ( self.uniform_gas_Mx + agn_feedback(Z,Y,X,time)[2]).in_units("code_mass*code_length**-2*code_time**-1").v,
                    "MomentumDensity3":lambda Z,Y,X,time :
                        ( self.uniform_gas_Mx + agn_feedback(Z,Y,X,time)[3]).in_units("code_mass*code_length**-2*code_time**-1").v,
                    "TotalEnergyDensity":lambda Z,Y,X,time :
                        (self.uniform_gas_energy_density + agn_feedback(Z,Y,X,time)[4]).in_units("code_mass*code_length**-1*code_time**-2").v,
                    }

            phdf_files = [f"{parameters.output_path}/parthenon.{output_id}.{i:05d}.phdf" for i in range(2)]

            def zero_corrected_linf_err(gold,test):
                non_zero_linf = np.max(np.abs((gold[gold!=0]-test[gold!=0])/gold[gold!=0]),initial=0)
                zero_linf = np.max(np.abs((gold[gold==0]-test[gold==0])),initial=0)

                return np.max((non_zero_linf,zero_linf))

            #Use a very loose tolerance, linf relative error
            initial_analytic_status,final_analytic_status = [
                compare_analytic.compare_analytic(
                    phdf_file, analytic_components,err_func=zero_corrected_linf_err,tol=1e-3)
                for analytic_components,phdf_file in zip((initial_analytic_components,final_analytic_components),
                                                          phdf_files)]

            print("  Initial analytic status",initial_analytic_status)
            print("  Final analytic status",final_analytic_status)

            analyze_status &= initial_analytic_status & final_analytic_status


        return analyze_status
