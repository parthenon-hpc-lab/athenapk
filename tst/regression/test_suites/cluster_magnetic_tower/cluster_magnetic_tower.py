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
import unyt
import itertools


class PrecessedJetCoords:
    def __init__(self,jet_phi,jet_theta):
        self.jet_phi = jet_phi
        self.jet_theta = jet_theta

        #Axis of the jet
        self.jet_n = np.array((np.cos(self.jet_phi)*np.sin(self.jet_theta),
                    np.sin(self.jet_phi)*np.sin(self.jet_theta),
                    np.cos(self.jet_theta)))
        
        #Axis defining jet_phi_j=0
        self.jet_m = np.array((np.cos(self.jet_phi)*np.sin(self.jet_theta - np.pi/2),
                    np.sin(self.jet_phi)*np.sin(self.jet_theta - np.pi/2),
                    np.cos(self.jet_theta - np.pi/2)))

        #Axis along theta_j=pi/2
        self.jet_o = np.cross(self.jet_n,self.jet_m)

    def cart_to_jet_coords(self,pos_cart):
        """
        Convert from cartesian coordinates to jet coordinates
        """

        pos_h = np.sum(pos_cart*self.jet_n[:,None],axis=0)
        pos_rho = np.linalg.norm( pos_cart - pos_h*self.jet_n[:,None],axis=0)
        pos_theta = np.arctan2(
            np.sum((pos_cart - pos_h*self.jet_n[:,None])*self.jet_o[:,None],axis=0),
            np.sum((pos_cart - pos_h*self.jet_n[:,None])*self.jet_m[:,None],axis=0)
        ) 
        pos_theta[pos_rho == 0] = 0
        return (pos_rho,pos_theta,pos_h)

    def jet_to_cart_vec(self,pos_cart,vec_jet):
        
        vec_rho = vec_jet[0]
        vec_theta = vec_jet[1]
        vec_h = vec_jet[2]
        
        #Compute pos in the jet-cylindrical-axis
        pos_rho,pos_theta,pos_h = self.cart_to_jet_coords(pos_cart)
            
        #Compute vector in cartesian coords
        vec_x = vec_rho*(np.sin(self.jet_phi)*np.sin(pos_theta)*np.cos(self.jet_theta) + np.cos(self.jet_theta)*np.cos(self.jet_phi)*np.cos(pos_theta)) \
            + vec_theta*(np.sin(self.jet_phi)*np.cos(self.jet_theta)*np.cos(pos_theta) - np.sin(pos_theta)*np.cos(self.jet_theta)*np.cos(self.jet_phi)) \
            + vec_h*np.sin(self.jet_theta)

        vec_y = vec_rho*(-np.sin(self.jet_phi)*np.cos(pos_theta) + np.sin(pos_theta)*np.cos(self.jet_phi)) \
            + vec_theta*(np.sin(self.jet_phi)*np.sin(pos_theta) + np.cos(self.jet_phi)*np.cos(pos_theta))

        vec_z = vec_rho*(-np.sin(self.jet_theta)*np.sin(self.jet_phi)*np.sin(pos_theta) - np.sin(self.jet_theta)*np.cos(self.jet_phi)*np.cos(pos_theta)) \
            + vec_theta*(-np.sin(self.jet_theta)*np.sin(self.jet_phi)*np.cos(pos_theta) + np.sin(self.jet_theta)*np.sin(pos_theta)*np.cos(self.jet_phi)) \
            + vec_h*np.cos(self.jet_theta)
        
        return (vec_x,vec_y,vec_z)

class ZJetCoords:
    def __init__(self):
        pass

    def cart_to_jet_coords(self,pos_cart):
        """
        Convert from cartesian coordinates to jet coordinates
        """

        pos_rho = np.linalg.norm(pos_cart[:2])
        pos_theta = np.arctan2(pos_cart[1],pos_cart[0]) 
        pos_theta[pos_rho == 0] = 0
        pos_h = pos_cart[2]

        return (pos_rho,pos_theta,pos_h)

    def jet_to_cart_vec(self,pos_cart,vec_jet):
        
        vec_rho = vec_jet[0]
        vec_theta = vec_jet[1]
        vec_h = vec_jet[2]

        pos_rho,pos_theta,pos_h = self.cart_to_jet_coords(pos_cart)
        
        #Compute vector in cartesian coords
        vec_x = vec_rho*np.cos(pos_theta) - vec_theta*np.sin(pos_theta)
        vec_y =-vec_rho*np.sin(pos_theta) + vec_theta*np.cos(pos_theta)

        vec_z = vec_h
        
        return (vec_x,vec_y,vec_z)

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

        self.tlim = unyt.unyt_quantity(1e-2,"code_time")

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
        self.jet_theta = 0.2
        self.jet_phi_dot = 0
        self.jet_phi0 = 1
        self.precessed_jet_coords = PrecessedJetCoords(self.jet_phi0,self.jet_theta)
        self.zjet_coords = ZJetCoords()

        #Initial and Feedback shared parameters
        self.magnetic_tower_alpha = 20
        self.magnetic_tower_l_scale = unyt.unyt_quantity(10,"kpc")

        #The Initial Tower
        self.initial_magnetic_tower_field = unyt.unyt_quantity(1e-6,"G")

        #The Feedback Tower
        #For const field tests
        self.feedback_magnetic_tower_field = unyt.unyt_quantity(1e-6,"G/Gyr")
        #For const energy tests
        self.feedback_magnetic_tower_power = unyt.unyt_quantity(1e44,"erg/s")
        
        self.norm_tol = 1e-3

        self.steps = 4
        self.step_params_list = list(itertools.product( ("const_field","const_power"),(True,False)))

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

                f"problem/cluster/jet_theta={self.jet_theta if precessed_jet else 0}",
                f"problem/cluster/jet_phi_dot={self.jet_phi_dot if precessed_jet else 0}",
                f"problem/cluster/jet_phi0={self.jet_phi0 if precessed_jet else 0}",

                f"problem/cluster/initial_magnetic_tower_field={self.initial_magnetic_tower_field.in_units('sqrt(code_mass)/sqrt(code_length)/code_time').v}",
                f"problem/cluster/initial_magnetic_tower_alpha={self.magnetic_tower_alpha}",
                f"problem/cluster/initial_magnetic_tower_l_scale={self.magnetic_tower_l_scale.in_units('code_length').v}",

                f"problem/cluster/feedback_magnetic_tower_mode={feedback_mode}",
                f"problem/cluster/feedback_magnetic_tower_field={self.feedback_magnetic_tower_field.in_units('sqrt(code_mass)/sqrt(code_length)/code_time**2').v}",
                f"problem/cluster/feedback_magnetic_tower_power={self.feedback_magnetic_tower_power.in_units('code_mass*code_length**2/code_time**3').v}",
                f"problem/cluster/feedback_magnetic_tower_alpha={self.magnetic_tower_alpha}",
                f"problem/cluster/feedback_magnetic_tower_l_scale={self.magnetic_tower_l_scale.in_units('code_length').v}",
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
        
        magnetic_units = 'sqrt(code_mass)/sqrt(code_length)/code_time'
        l = self.magnetic_tower_l_scale

        def precessed_field(Z,Y,X,B0):
            R,Theta,H = self.precessed_jet_coords.cart_to_jet_coords(np.array((X,Y,Z)))
            R = unyt.unyt_array(R,"code_length")
            H = unyt.unyt_array(H,"code_length")

            B_r = B0*2*H/l*R/l*np.exp( - (R/l)**2 - (H/l)**2)
            B_theta = B0*self.magnetic_tower_alpha*R/l*np.exp( - (R/l)**2 - (H/l)**2)
            B_h = B0*2*( 1 - (R/l)**2 )*np.exp( - (R/l)**2 - (H/l)**2)

            B_x,B_y,B_z = self.precessed_jet_coords.jet_to_cart_vec(np.array((X,Y,Z)),np.array((B_r,B_theta,B_h)))

            return unyt.unyt_array((B_z, B_y, B_x),magnetic_units)

        def zjet_field(Z,Y,X,B0):
            R,Theta,H = self.zjet_coords.cart_to_jet_coords(np.array((X,Y,Z)))
            R = unyt.unyt_array(R,"code_length")
            H = unyt.unyt_array(H,"code_length")

            B_r = B0*2*H/l*R/l*np.exp( - (R/l)**2 - (H/l)**2)
            B_theta = B0*self.magnetic_tower_alpha*R/l*np.exp( - (R/l)**2 - (H/l)**2)
            B_h = B0*2*( 1 - (R/l)**2 )*np.exp( - (R/l)**2 - (H/l)**2)

            B_x,B_y,B_z = self.zjet_coords.jet_to_cart_vec(np.array((X,Y,Z)),np.array((B_r,B_theta,B_h)))

            return unyt.unyt_array((B_z, B_y, B_x),magnetic_units)

        for step in range(1,self.steps+1):
            feedback_mode,precessed_jet = self.step_params_list[step-1]
            output_id = f"{feedback_mode}_precessed_{precessed_jet}"
            step_status = True

            print(f"Testing {output_id}")

            B0_initial = self.initial_magnetic_tower_field

            if feedback_mode == "const_field":
                B0_final = self.feedback_magnetic_tower_field*self.tlim + self.initial_magnetic_tower_field

                #Estimate the final magnetic field using the total energy of the tower out to inifinity
                EB_final = np.pi**(3./2.)/(8*np.sqrt(2))*(5 + self.magnetic_tower_alpha**2)*self.magnetic_tower_l_scale**3*B0_final**2
            elif feedback_mode == "const_power":
                #Estimate the final magnetic field using the total energy of the tower out to inifinity
                #Slightly inaccurate due to finite domain
                B0_final = np.sqrt(self.feedback_magnetic_tower_power*self.tlim/(
                    np.pi**(3./2.)/(8*np.sqrt(2))*(5 + self.magnetic_tower_alpha**2)*self.magnetic_tower_l_scale**3)
                    + self.initial_magnetic_tower_field**2)
                EB_final = self.feedback_magnetic_tower_power*self.tlim + \
                    (np.pi**(3./2.)/(8*np.sqrt(2))*(5 + self.magnetic_tower_alpha**2)*self.magnetic_tower_l_scale**3)*B0_initial**2
            else:
                raise Exception(f"Feedback mode {feedback_mode} not supported in analysis")

            if precessed_jet is True:
                field_func = precessed_field
            else:
                field_func = zjet_field

            b_energy_func = lambda Z,Y,X,B0 : 0.5*np.sum(field_func(Z,Y,X,B0)**2,axis=0)

            #Check that the initial and final outputs match the expected tower
            sys.path.insert(1, parameters.parthenon_path + '/scripts/python/packages/parthenon_tools/parthenon_tools')

            try:
                import compare_analytic
            except ModuleNotFoundError:
                print("Couldn't find module to compare Parthenon hdf5 files.")
                return False


            initial_analytic_components,final_analytic_components = [{
                    "Density":lambda Z,Y,X,time :
                        np.ones_like(Z)*self.uniform_gas_rho.in_units("code_mass/code_length**3").v,
                    #"MomentumDensity1":lambda Z,Y,X,time :
                    #    np.ones_like(Z)*self.uniform_gas_Mx.in_units("code_mass*code_length**-2*code_time**-1").v,
                    #"MomentumDensity2":lambda Z,Y,X,time :
                    #    np.ones_like(Z)*self.uniform_gas_My.in_units("code_mass*code_length**-2*code_time**-1").v,
                    #"MomentumDensity3":lambda Z,Y,X,time :
                    #    np.ones_like(Z)*self.uniform_gas_Mz.in_units("code_mass*code_length**-2*code_time**-1").v,
                    "TotalEnergyDensity":lambda Z,Y,X,time : (self.uniform_gas_energy_density 
                        + b_energy_func(Z,Y,X,B0_initial)).in_units("code_mass*code_length**-1*code_time**-2").v,
                    "MagneticField1":lambda Z,Y,X,time : 
                        field_func(Z,Y,X,B0_initial).in_units("sqrt(code_mass)/sqrt(code_length)/code_time")[0].v,
                    "MagneticField2":lambda Z,Y,X,time : 
                        field_func(Z,Y,X,B0_initial).in_units("sqrt(code_mass)/sqrt(code_length)/code_time")[1].v,
                    "MagneticField3":lambda Z,Y,X,time : 
                        field_func(Z,Y,X,B0_initial).in_units("sqrt(code_mass)/sqrt(code_length)/code_time")[2].v,
                    } for field in (B0_initial,B0_final)]

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

            print("Analytic Statuses:",initial_analytic_status,final_analytic_status)

            analyze_status &= initial_analytic_status & final_analytic_status


        return analyze_status
