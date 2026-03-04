import os
import sys
import glob
import shutil
import numpy as np



template_file = "./test_merged_dust_carbonaceous_and_silicates_32bins.in"
sbatch_fname = "sbatch_conv_test_multiGPU.sh"

### These must ALL be LISTS (don't do e.g. ["1"])
thermal_sputtering_metal_accretion_list  = ["true",]
init_grainsize_distributions_list        = ["MRN",]
num_grainsize_bins_list                  = [str(x) for x in np.arange(4, 24, 4)]  ##[str(x) for x in np.arange(44, 64, 4)] 
max_dM_in_bin_list                       = ["1",]
cooling_list                             = ["Dwek_Werner1981_INTEGRATED",]
piecewise_method_list                    = ["loglinear",]
time_integrator_list                     =  ["heun",]


if len(sys.argv) > 1:
    sbatch_jobs = int(sys.argv[1])
else:
    sbatch_jobs = 0
run_num = 0
for i_num_grainsize_bins, num_grainsize_bins in enumerate(num_grainsize_bins_list):
    for i_piecewise_method, piecewise_method in enumerate(piecewise_method_list):
        for i_thermal_sputtering_metal_accretion, thermal_sputtering_metal_accretion in enumerate(thermal_sputtering_metal_accretion_list):
            for i_init_grainsize_distribution, init_grainsize_distribution in enumerate(init_grainsize_distributions_list):
                for cooling in cooling_list:
                    for max_dM_in_bin in max_dM_in_bin_list:
                        for time_integrator in time_integrator_list:
                            
                            flat_single_bin_gs_i_index =  int(num_grainsize_bins) // 2  ## If just putting all mass in one bin, make it halfway
                            print(flat_single_bin_gs_i_index)
                            
                            ### If filename is too long, simulation segfaults
                            # output_file = f"./convtest_recon={piecewise_method}_cool={cooling}_n_bins={num_grainsize_bins}_initdist={init_grainsize_distribution}_sputt+acc={thermal_sputtering_metal_accretion}.in"
                            output_file = f"./N{num_grainsize_bins}P{piecewise_method}SM{thermal_sputtering_metal_accretion}D{init_grainsize_distribution}_dM{max_dM_in_bin}_I{time_integrator}.in"
                            shutil.copy(template_file, output_file)
                            # exit()
                            with open(output_file, "r") as f:
                                lines = f.readlines()
                            with open(output_file, "w") as f:
                                in_dust_block = False
                                for line in lines:
                                    stripped = line.strip()
                                    
                                    if stripped.startswith("<"):
                                        if stripped.startswith("<dust"):
                                            in_dust_block = True
                                        else:
                                            ## n another block
                                            in_dust_block = False
                                            
                                    if in_dust_block:
                                        if stripped.startswith("cooling"):
                                            f.write(f"cooling = {cooling}")
                                            f.write("\n")
                                        elif stripped.startswith("piecewise_method"):
                                            f.write(f"piecewise_method = {piecewise_method}")
                                            f.write("\n")
                                        elif stripped.startswith("time_integrator"):
                                            f.write(f"time_integrator = {time_integrator}")
                                            f.write("\n")
                                        elif stripped.startswith("num_grainsize_bins"):
                                            f.write(f"num_grainsize_bins = {num_grainsize_bins}")  
                                            f.write("\n")    
                                        elif stripped.startswith("cooling"):
                                            f.write(f"cooling = {cooling}")
                                            f.write("\n")
                                        elif stripped.startswith("metal_accretion"):
                                            f.write(f"metal_accretion = {thermal_sputtering_metal_accretion}")
                                            f.write("\n")
                                        elif stripped.startswith("thermal_sputtering"):
                                            f.write(f"thermal_sputtering = {thermal_sputtering_metal_accretion}")   
                                            f.write("\n")                                              
                                        elif stripped.startswith("init_grainsize_distribution"):
                                            f.write(f"init_grainsize_distribution = {init_grainsize_distribution}")  
                                            f.write("\n")
                                        elif stripped.startswith("max_dM_in_bin"):
                                            f.write(f"max_dM_in_bin = {max_dM_in_bin}")
                                            f.write("\n")
                                        elif stripped.startswith("flat_single_bin_gs_i_index"):
                                            f.write(f"flat_single_bin_gs_i_index = {flat_single_bin_gs_i_index}")
                                            f.write("\n")
                                        else:
                                            f.write(line)
                                        
                                    elif not stripped.startswith("#"):
                                        f.write(line)
                                        
                            run_tag = output_file.removesuffix(".in").strip("./")    
                            sbatch_str = f"sbatch {sbatch_fname}  {run_tag}.in {run_tag}"   
                            print(output_file) 
                            print(sbatch_str) 
                            if sbatch_jobs == 1:
                                
                                # os.system(f"rm -rf ./{run_tag}")
                                os.makedirs(f"./{run_tag}", exist_ok=True)
                                os.chdir(f"./{run_tag}")
                                shutil.copy(f"../{sbatch_fname}", f"./{sbatch_fname}")
                                shutil.copy(f"../{run_tag}.in", f"./{run_tag}.in")
                                # os.system("rm -rf athenaPK")
                                ### Give each run its own executable
                                # os.system("cp /leonardo/home/userexternal/fjenning/DustBlackHoleWeather/athenapk/build-cuda/bin/athenaPK .")
                                
                                os.system(sbatch_str)
                                os.system(f"rm -rf {sbatch_fname}")
                                
                                os.chdir(f"..")
                                # exit()
                                
                            run_num += 1
                            
                                                             
