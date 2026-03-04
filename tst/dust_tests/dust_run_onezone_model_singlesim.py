import numpy as np
from onezone_funcs import *





running_mode = "convergence"

if running_mode == "single":
    ### For single run analysis
    N_particles = 1_000_000
    sims_parent_dir = "/leonardo_scratch/large/userexternal/fjenning/Jan26_test_merged_dust+main/test_dust_convergence_with_Nbins/test_merged_dust_carbonaceous_and_silicates_32bins/"
    save_tag = f"_N_particles={N_particles}_test_merged_dust_carbonaceous_and_silicates"
    run_onezone(sims_parent_dir, N_particles, save_tag, t_end_Myr = 7., dt_coarse = 1.)
    onezone_data_path = sims_parent_dir   + "/Testing_Dust_Model_vs_OneZone/"
    norm_factor_Ni, norm_factor_Mi, frac_err_dict = get_fractional_errors(onezone_data_path, sims_parent_dir, snapshot_number = 1)
    for key in list(frac_err_dict.keys()):
        if "frac_e" in key:
            print(key, frac_err_dict[key])



if running_mode == "convergence":
    ### For convergence testing analysis
    N_particles = 5_00_000
    Nbins_dummy = 4
    sims_parent_dir = "/leonardo_scratch/large/userexternal/fjenning/MARCH26_ConvTests/N{}PloglinearSMtrueDMRN_dM1_Iheun/N{}PloglinearSMtrueDMRN_dM1_Iheun/"
    save_tag = f"_N_particles={N_particles}_test_merged_dust_carbonaceous_and_silicates"
    onezone_data_path = sims_parent_dir   + "/Testing_Dust_Model_vs_OneZone/"
    # print(onezone_data_path.format(Nbins_dummy,Nbins_dummy))
    # exit()
    run_onezone(sims_parent_dir.format(Nbins_dummy,Nbins_dummy), N_particles, save_tag, t_end_Myr = 20., dt_coarse = 1.)
    ### For convergence testing analysis
    Nbins_list = list(np.arange(4, 24, 4))
    M_frac_errs = []
    N_frac_errs = []
    T_frac_errs = []
    successful_N = []
    for x in Nbins_list:
            norm_factor_Ni, norm_factor_Mi, frac_err_dict = get_fractional_errors(onezone_data_path.format(Nbins_dummy,Nbins_dummy),save_tag, sims_parent_dir.format(x,x), snapshot_number = 3, plot_style = "line")
            if np.isnan(norm_factor_Ni): continue
            M_frac_errs.append(frac_err_dict["frac_e_M_C_vs_onezone"])
            N_frac_errs.append(frac_err_dict["frac_e_N_C_vs_onezone"])
            T_frac_errs.append(frac_err_dict["frac_e_T_vs_onezone"])
            successful_N.append(x)
            # exit()


        
        
        
    plt.figure(figsize=(4,4))
    plt.xlabel(r"N$_{bins}$")
    plt.ylabel(f"Fractional Error M")
    plt.scatter(successful_N, M_frac_errs, marker = "x", alpha = 0.8,)
    plt.xscale("log")
    plt.yscale("log")
    # plt.ylim(1e-4, 1)
    plt.tight_layout()
    mysavefig(f"./frac_err_M.png")
    plt.close()

    plt.figure(figsize=(4,4))
    plt.xlabel(r"N$_{bins}$")
    plt.ylabel(f"Fractional Error N")
    plt.scatter(successful_N, N_frac_errs, marker = "x", alpha = 0.8,)
    plt.xscale("log")
    plt.yscale("log")
    # plt.ylim(1e-4, 1)
    plt.tight_layout()
    mysavefig(f"./frac_err_N.png")
    plt.close()

    plt.figure(figsize=(4,4))
    plt.xlabel(r"N$_{bins}$")
    plt.ylabel(f"Fractional Error T")
    plt.scatter(successful_N, T_frac_errs, marker = "x", alpha = 0.8,)
    plt.xscale("log")
    plt.yscale("log")
    # plt.ylim(1e-4, 1)
    plt.tight_layout()
    mysavefig(f"./frac_err_T.png")
    plt.close()
    
    
    







