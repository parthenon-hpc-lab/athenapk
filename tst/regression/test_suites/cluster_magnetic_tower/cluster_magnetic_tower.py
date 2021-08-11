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
    #Note: Does note rotate the vector around the jet axis, only rotates the vector to `z_hat`
    def __init__(self,phi_jet,theta_jet):
        self.phi_jet = phi_jet
        self.theta_jet = theta_jet

    def cart_to_jet_coords(self,pos_sim):
        """
        Convert from simulation cartesian coordinates to jet cylindrical coordinates
        """

        x_sim = pos_sim[0]
        y_sim = pos_sim[1]
        z_sim = pos_sim[2]

        x_jet = x_sim*np.cos(self.phi_jet)*np.cos(self.theta_jet) + y_sim*np.sin(self.phi_jet) - z_sim*np.sin(self.theta_jet)*np.cos(self.phi_jet)
        y_jet = -x_sim*np.sin(self.phi_jet)*np.cos(self.theta_jet) + y_sim*np.cos(self.phi_jet) + z_sim*np.sin(self.phi_jet)*np.sin(self.theta_jet)
        z_jet = x_sim*np.sin(self.theta_jet) + z_sim*np.cos(self.theta_jet)

        r_jet = np.sqrt( x_jet**2 + y_jet**2)
        theta_jet = np.arctan2(y_jet,x_jet)
        h_jet = z_jet
        return (r_jet, theta_jet, h_jet)

    def jet_to_cart_vec(self,pos_sim,vec_jet):

        r_pos, theta_pos, h_pos = self.cart_to_jet_coords(pos_sim)

        #Convert jet-cylindrical vec_jet into jet-cartesian
        R = np.array((( np.cos(theta_pos),-np.sin(theta_pos), np.zeros_like(theta_pos)),
                      ( np.sin(theta_pos), np.cos(theta_pos), np.zeros_like(theta_pos)),
                      ( np.zeros_like(theta_pos),np.zeros_like(theta_pos), np.ones_like(theta_pos)))).reshape(
                              (3,3,*(theta_pos.shape)))
        #vec_jet_cart = np.einsum("ij,jxyz",R,vec_jet)
        vec_jet_cart = unyt.unyt_array((R[0,0]*vec_jet[0] + R[0,1]*vec_jet[1] + R[0,2]*vec_jet[2],
                                   R[1,0]*vec_jet[0] + R[1,1]*vec_jet[1] + R[1,2]*vec_jet[2],
                                   R[2,0]*vec_jet[0] + R[2,1]*vec_jet[1] + R[2,2]*vec_jet[2]),
                                   vec_jet.units)
                   

        R = np.array(((np.cos(self.phi_jet)*np.cos(self.theta_jet),
                      -np.sin(self.phi_jet)*np.cos(self.theta_jet),
                       np.sin(self.theta_jet)),

                      (np.sin(self.phi_jet),
                       np.cos(self.phi_jet),
                       0),

                      (-np.sin(self.theta_jet)*np.cos(self.phi_jet),
                       np.sin(self.phi_jet)*np.sin(self.theta_jet),
                       np.cos(self.theta_jet))
                       ))

        #vec_cart = np.matmul(R,vec_jet_cart)
        vec_cart = unyt.unyt_array((R[0,0]*vec_jet_cart[0] + R[0,1]*vec_jet_cart[1] + R[0,2]*vec_jet_cart[2],
                               R[1,0]*vec_jet_cart[0] + R[1,1]*vec_jet_cart[1] + R[1,2]*vec_jet_cart[2],
                               R[2,0]*vec_jet_cart[0] + R[2,1]*vec_jet_cart[1] + R[2,2]*vec_jet_cart[2]),
                               vec_jet_cart.units)

        return vec_cart

class ZJetCoords:
    def __init__(self):
        pass

    def cart_to_jet_coords(self,pos_cart):
        """
        Convert from cartesian coordinates to jet coordinates
        """

        pos_rho = np.sqrt(pos_cart[0]**2 + pos_cart[1]**2)
        pos_theta = np.arctan2(pos_cart[1],pos_cart[0]) 
        pos_theta[pos_rho == 0] = 0
        pos_h = pos_cart[2]

        return (pos_rho,pos_theta,pos_h)

    def jet_to_cart_vec(self,pos_cart,vec_jet):
        
        vec_rho = vec_jet[0]
        vec_theta = vec_jet[1]
        vec_h = vec_jet[2]

        r_pos,theta_pos,h_pos = self.cart_to_jet_coords(pos_cart)
        
        #Compute vector in cartesian coords
        vec_x = vec_rho*np.cos(theta_pos) - vec_theta*np.sin(theta_pos)
        vec_y = vec_rho*np.sin(theta_pos) + vec_theta*np.cos(theta_pos)
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
        self.theta_jet = 0.2
        self.phi_dot_jet = 0 #Use phi_dot = 0 for stationary jet
        self.phi_jet0 = 1 #Offset initial jet
        self.precessed_jet_coords = PrecessedJetCoords(self.phi_jet0,self.theta_jet)
        self.zjet_coords = ZJetCoords()

        #Initial and Feedback shared parameters
        self.magnetic_tower_alpha = 20
        #self.magnetic_tower_l_scale = unyt.unyt_quantity(1,"code_length")
        self.magnetic_tower_l_scale = unyt.unyt_quantity(10,"kpc")

        #The Initial Tower
        self.initial_magnetic_tower_field = unyt.unyt_quantity(1e-6,"G")

        #The Feedback Tower
        #For const field tests
        self.feedback_magnetic_tower_field = unyt.unyt_quantity(1e-4,"G/Gyr")
        #For const energy tests
        self.feedback_magnetic_tower_power = unyt.unyt_quantity(1e44,"erg/s")

        #Tolerance of magnetic fields and magnetic energy density
        self.norm_tol = 1e-3

        #Tolerance on total initial and final magnetic energy
        self.b_eng_initial_tol = 1e-5
        self.b_eng_final_tol = 1e-5

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

                f"problem/cluster/uniform_gas/init_uniform_gas=true",
                f"problem/cluster/uniform_gas/rho={self.uniform_gas_rho.in_units('code_mass*code_length**-3').v}",
                f"problem/cluster/uniform_gas/ux={self.uniform_gas_ux.in_units('code_length*code_time**-1').v}",
                f"problem/cluster/uniform_gas/uy={self.uniform_gas_uy.in_units('code_length*code_time**-1').v}",
                f"problem/cluster/uniform_gas/uz={self.uniform_gas_uz.in_units('code_length*code_time**-1').v}",
                f"problem/cluster/uniform_gas/pres={self.uniform_gas_pres.in_units('code_mass*code_length**-1*code_time**-2').v}",

                f"problem/cluster/precessing_jet/jet_theta={self.theta_jet if precessed_jet else 0}",
                f"problem/cluster/precessing_jet/jet_phi_dot={self.phi_dot_jet if precessed_jet else 0}",
                f"problem/cluster/precessing_jet/jet_phi0={self.phi_jet0 if precessed_jet else 0}",

                f"problem/cluster/magnetic_tower/alpha={self.magnetic_tower_alpha}",
                f"problem/cluster/magnetic_tower/l_scale={self.magnetic_tower_l_scale.in_units('code_length').v}",

                f"problem/cluster/magnetic_tower/initial_field={self.initial_magnetic_tower_field.in_units('sqrt(code_mass)/sqrt(code_length)/code_time').v}",

                f"problem/cluster/magnetic_tower/feedback_mode={feedback_mode}",
                f"problem/cluster/magnetic_tower/feedback_field_rate={self.feedback_magnetic_tower_field.in_units('sqrt(code_mass)/sqrt(code_length)/code_time**2').v}",
                f"problem/cluster/magnetic_tower/feedback_power={self.feedback_magnetic_tower_power.in_units('code_mass*code_length**2/code_time**3').v}",
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

        for step in range(1,self.steps+1):
            feedback_mode,precessed_jet = self.step_params_list[step-1]
            output_id = f"{feedback_mode}_precessed_{precessed_jet}"
            step_status = True

            print(">"*20)
            print(f"Testing {output_id}")
            print(">"*20)

            B0_initial = self.initial_magnetic_tower_field
            #Compute the initial magnetic energy
            b_eng_initial_anyl = np.pi**(3./2.)/(8*np.sqrt(2))*(5 + self.magnetic_tower_alpha**2)*self.magnetic_tower_l_scale**3*B0_initial**2

            if feedback_mode == "const_field":
                B0_final = self.feedback_magnetic_tower_field*self.tlim + self.initial_magnetic_tower_field
                #Estimate the final magnetic field using the total energy of the tower out to inifinity
                b_eng_final_anyl = np.pi**(3./2.)/(8*np.sqrt(2))*(5 + self.magnetic_tower_alpha**2)*self.magnetic_tower_l_scale**3*B0_final**2
            elif feedback_mode == "const_power":
                #Estimate the final magnetic field using the total energy of the tower out to inifinity
                #Slightly inaccurate due to finite domain
                B0_final = np.sqrt((b_eng_initial_anyl + self.feedback_magnetic_tower_power*self.tlim)/(
                    np.pi**(3./2.)/(8*np.sqrt(2))*(5 + self.magnetic_tower_alpha**2)*self.magnetic_tower_l_scale**3))
                b_eng_final_anyl = self.feedback_magnetic_tower_power*self.tlim + \
                    (np.pi**(3./2.)/(8*np.sqrt(2))*(5 + self.magnetic_tower_alpha**2)*self.magnetic_tower_l_scale**3)*B0_initial**2
            else:
                raise Exception(f"Feedback mode {feedback_mode} not supported in analysis")

            if precessed_jet is True:
                jet_coords = self.precessed_jet_coords
            else:
                jet_coords = self.zjet_coords

            def field_func(Z,Y,X,B0):
                l = self.magnetic_tower_l_scale
                alpha = self.magnetic_tower_alpha

                pos_cart = unyt.unyt_array((X,Y,Z),"code_length")
                R,Theta,H = jet_coords.cart_to_jet_coords(pos_cart)

                B_r = B0*2*(H/l)*(R/l)     *np.exp( -(R/l)**2 -(H/l)**2)
                B_theta = B0*alpha*(R/l)   *np.exp( -(R/l)**2 -(H/l)**2)
                B_h = B0*2*( 1 - (R/l)**2 )*np.exp( -(R/l)**2 -(H/l)**2)
                B_jet = unyt.unyt_array((B_r,B_theta,B_h),magnetic_units)

                B_x,B_y,B_z = jet_coords.jet_to_cart_vec(pos_cart,B_jet)


                return unyt.unyt_array((B_x, B_y, B_z),magnetic_units)

            b_energy_func = lambda Z,Y,X,B0 : 0.5*np.sum(field_func(Z,Y,X,B0)**2,axis=0)

            #Check that the initial and final outputs match the expected tower
            sys.path.insert(1, parameters.parthenon_path + '/scripts/python/packages/parthenon_tools/parthenon_tools')

            try:
                import compare_analytic
                import phdf
            except ModuleNotFoundError:
                print("Couldn't find module to compare Parthenon hdf5 files.")
                return False 

            ######################################## 
            # Compare to the analytically expected magnetic field
            ######################################## 

            phdf_filenames = [f"{parameters.output_path}/parthenon.{output_id}.{i:05d}.phdf" for i in range(2)]

            #Create a relative L-Inf errpr function, ignore where zero in gold data
            rel_linf_err_func = lambda gold, test: compare_analytic.norm_err_func(
                gold,test,norm_ord=np.inf,relative=True,ignore_gold_zero=True)

            #Create a linf error function scaled by a magnetic field
            #Avoids relative comparisons in areas where magnetic field is close to zero
            def B_scaled_linf_err(gold,test,B0):
                err_val = np.abs((gold - test)/B0)
                return err_val.max()

            #Use a very loose tolerance, linf relative error
            analytic_statuses = []
            for B_field,phdf_filename,label in zip(
                (B0_initial,B0_final),
                phdf_filenames,
                ("Initial","Final")):

                #Construct lambda functions for initial and final analytically
                #expected density and total energy density
                densities_analytic_components = {
                    "Density":lambda Z,Y,X,time :
                        np.ones_like(Z)*self.uniform_gas_rho.in_units("code_mass/code_length**3").v,
                    "TotalEnergyDensity":lambda Z,Y,X,time : (self.uniform_gas_energy_density 
                        + b_energy_func(Z,Y,X,B_field)).in_units("code_mass*code_length**-1*code_time**-2").v,
                }


                #Compare the simulation and analytic density and total energy density
                densities_analytic_status = compare_analytic.compare_analytic(
                        phdf_filename, densities_analytic_components,
                        err_func=rel_linf_err_func,
                        tol=1e-3)

                #Construct lambda functions for initial and final analytically expected magnetic fields
                field_analytic_components = {
                    "MagneticField1":lambda Z,Y,X,time : 
                        field_func(Z,Y,X,B_field).in_units(magnetic_units)[0].v,
                    "MagneticField2":lambda Z,Y,X,time : 
                        field_func(Z,Y,X,B_field).in_units(magnetic_units)[1].v,
                    "MagneticField3":lambda Z,Y,X,time : 
                        field_func(Z,Y,X,B_field).in_units(magnetic_units)[2].v,
                }

                #Compare the simulation and analytic magnetic fields,
                #scaled by magnetic tower field scale
                field_analytic_status = compare_analytic.compare_analytic(
                        phdf_filename, field_analytic_components,
                        err_func=lambda gold,test: B_scaled_linf_err(gold,test,
                            B_field.in_units(magnetic_units).v[()]),
                        tol=1e-3)

                analytic_status = (densities_analytic_status and field_analytic_status)
                if not analytic_status:
                    print(f"{label} Analytic comparison failed\n")

                analytic_statuses.append(analytic_status)

            analyze_status &= np.all(analytic_statuses)

            for phdf_filename,b_eng_anyl,B0,b_eng_tol,label in zip(phdf_filenames,
                    (b_eng_initial_anyl,b_eng_final_anyl),
                    (B0_initial,B0_final),
                    (self.b_eng_initial_tol,self.b_eng_final_tol),
                    ("Initial","Final")):

                ######################################## 
                # Compare with the analytically expected total magnetic energy
                ######################################## 
                phdf_file = phdf.phdf(phdf_filename)

                #Get the cell volumes from phdf_file
                xf = phdf_file.xf
                yf = phdf_file.yf
                zf = phdf_file.zf
                cell_vols =  unyt.unyt_array(np.einsum('ai,aj,ak->aijk',
                    np.diff(zf),np.diff(yf),np.diff(xf)),"code_length**3")

                #Get the magnetic energy from phdf_file
                B = unyt.unyt_array( 
                    list(phdf_file.GetComponents(["MagneticField1","MagneticField2","MagneticField3"],flatten=False).values()),
                    magnetic_units)
                b_eng = np.sum(0.5*np.sum(B**2,axis=0)*cell_vols)

                #Get the estimated magnetic energy from the expected mt_tower field

                Z,Y,X = phdf_file.GetVolumeLocations(flatten=False)
                Z = unyt.unyt_array(Z,"code_length")
                Y = unyt.unyt_array(Y,"code_length")
                X = unyt.unyt_array(X,"code_length")
                b_eng_numer = np.sum( b_energy_func(Z,Y,X,B0)*cell_vols )

                b_eng_anyl_rel_err = np.abs((b_eng - b_eng_anyl)/b_eng_anyl)
                b_eng_numer_rel_err = np.abs((b_eng - b_eng_numer)/b_eng_numer)

                if b_eng_anyl_rel_err > b_eng_tol:
                    print(f"{label} Analytically Integrated Relative Energy Error: {b_eng_anyl_rel_err} exceeds tolerance {b_eng_tol}",
                        f"Analytic {'>' if  b_eng_anyl > b_eng else '<'} Simulation")
                    analyze_status = False
                if b_eng_numer_rel_err > b_eng_tol:
                    print(f"{label} Numerically Integrated Relative Energy Error: {b_eng_numer_rel_err} exceeds tolerance {b_eng_tol}",
                        f"Numerical {'>' if  b_eng_numer > b_eng else '<'} Simulation")
                    analyze_status = False

                ######################################## 
                # Check divB
                ######################################## 
                
                #FIXME: This computation of the fluxes would work better with 1 ghostzone from the simulation
                #Compute cell lengths (note: these are NGridxNBlockSide)
                dxf = np.diff(xf,axis=1)
                dyf = np.diff(yf,axis=1)
                dzf = np.diff(zf,axis=1)

                dBxdx = 0.5*(B[0,:,:,:,2:]-B[0,:,:,:,:-2])[:,1:-1,1:-1,:]/dxf[:,np.newaxis,np.newaxis,1:-1]
                dBydy = 0.5*(B[1,:,:,2:,:]-B[1,:,:,:-2,:])[:,1:-1,:,1:-1]/dyf[:,np.newaxis,1:-1,np.newaxis]
                dBzdz = 0.5*(B[2,:,2:,:,:]-B[2,:,:-2,:,:])[:,:,1:-1,1:-1]/dzf[:,1:-1,np.newaxis,np.newaxis]

                divB = dBxdx + dBydy + dBzdz
                int_divB = np.sum(divB*cell_vols[:,1:-1,1:-1,1:-1])
                
                #DEBUGGING - alternative divergence calculation with too many interpolations
                #def consecutive_mean(a,axis):
                #    """
                #    Returns the mean of consecutive elements along an axis
                #    Reduces the shape of "a" by 1 along "axis"
                #    """
                #    ind = np.arange(a.shape[axis])
                #    return 0.5*( a.take(ind[1:],axis=axis) + a.take(ind[:-1],axis=axis))
                ##Interpoate X magnetic field along x edges by averaging in y and z
                #B_xe = consecutive_mean(consecutive_mean(B[0],2),1)
                ##Interpoate Y magnetic field along y edges by averaging in x and z
                #B_ye = consecutive_mean(consecutive_mean(B[1],3),1)
                ##Interpoate Z magnetic field along z edges by averaging in x and y
                #B_ze = consecutive_mean(consecutive_mean(B[2],3),2)

                ##Compute cell edge lengths
                #dxe = consecutive_mean(consecutive_mean(np.diff(X,axis=3),2),1)
                #dye = consecutive_mean(consecutive_mean(np.diff(Y,axis=2),3),1)
                #dze = consecutive_mean(consecutive_mean(np.diff(Z,axis=1),3),2)

                ##Compute the flux divergence at cell vertices
                #divB = np.diff(B_xe,axis=3)/dxe + np.diff(B_ye,axis=2)/dye + np.diff(B_ze,axis=1)/dze

                ##Integrate the flux divergence
                #int_divB = np.sum(consecutive_mean(consecutive_mean(consecutive_mean(cell_vols,1),2),3)*divB)
                #END DEBUGGING

                #DEBUGGING - alternative divergence calculation with numpy  
                #dxf = np.diff(xf,axis=1)
                #dyf = np.diff(yf,axis=1)
                #dzf = np.diff(zf,axis=1)
                #divB = unyt.unyt_array( np.empty(B.shape[1:]),'sqrt(code_mass)/(code_length)**(3/2)/code_time')
                #B_flipped = np.flip(B,axis=0) #B_z, B_y, B_x
                #for grid_idx in range(dxf.shape[0]):
                #    divB[grid_idx] = np.sum( [np.gradient(B_flipped[i,grid_idx], 
                #                            dxf[grid_idx][0], axis=i) for i in  np.arange(3)],axis=0)
                #int_divB = np.sum(divB*cell_vols)
                #END DEBUGGING

                print()
                print(f"{label} max divB={np.max(divB)}")
                print(f"{label} Integrated divB={int_divB}")
                print()
                



        return analyze_status
