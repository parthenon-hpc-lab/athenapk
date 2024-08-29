import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    ## plot *solar abundance* CIE cooling curves

    ## Sutherland and Dopita (1993) [NOTE: n_i n_e convention for Lambda_N, where n_t == n_i in their notation]
    # Columns: Log(T), n_e, n_H, n_t, log(\Lambda_net), log(\Lambda_N)
    sd_LogT, sd_ne, sd_nH, sd_nt, sd_logLambdaNet, sd_logLambdaN = np.loadtxt(
        "sutherland_dopita_table_6.txt", unpack=True
    )
    # convert to Lambda_hd:
    sd_LambdaN = 10.0**sd_logLambdaN
    sd_LambdaHD = (sd_ne * sd_nt * sd_LambdaN) / sd_nH**2
    # convert to nH ne Lambda(T):
    sd_nHneLambda = (sd_ne * sd_nt * sd_LambdaN) / (sd_nH * sd_ne)

    ## Schure et al. (2009) [NOTE: n_H n_e convention for Lambda_N; n_H^2 convention for Lambda_hd]
    # Columns: log T (K), log Λ_N (erg s^−1 cm^3), log Λ_hd (erg s^−1 cm^3), n_e/n_H
    # BEWARE: This Lambda_N is NOT the same as the Sutherland and Dopita Lambda_N!
    s_LogT, s_logLambdaN, s_logLambdaHD, s_ne_over_nH = np.loadtxt(
        "schure_table_2.txt", unpack=True
    )
    s_LambdaHD = 10.0**s_logLambdaHD

    ## Gnat and Sternberg (2007) [NOTE: n_H n_e convention for Lambda]
    # Columns: Temperature Lambda(Z=1e-3) Lambda(Z=1e-2) Lambda(Z=1e-1) Lambda(Z=1) Lambda(Z=2)
    gs_T, gs_LambdaZem3, gs_LambdaZem2, gs_LambdaZem1, gs_LambdaZ1, gs_LambdaZ2 = (
        np.loadtxt("gnat_sternberg_cie_table.txt", unpack=True, skiprows=23)
    )
    # NOTE: There is not enough information in this table alone to compute n_e!
    #  (The electron fraction must be computed from datafile2a.txt)

    # plot nH^2-normalized Lambda(T)
    plt.figure()
    plt.plot(
        sd_LogT,
        sd_LambdaHD,
        label=r"Sutherland & Dopita $\Lambda_{hd}$ for $Z_{\odot}$",
    )
    plt.plot(s_LogT, s_LambdaHD, label=r"Schure et al. $\Lambda_{hd}$ for $Z_{\odot}$")
    # (Gnat & Sternberg cannot be converted to this convention without the electron fraction, which is missing.)
    plt.yscale("log")
    plt.ylim(1e-23, 3e-21)
    plt.xlabel(r"$\log T$ (K)")
    plt.ylabel(r"$\Lambda(T)$ [$n_H^2$ convention]")
    plt.legend()
    plt.tight_layout()
    plt.savefig("cooling_curves_LambdaHD.png")

    # plot nH ne-normalized Lambda(T)
    plt.figure()
    plt.plot(
        sd_LogT, sd_nHneLambda, label=r"Sutherland & Dopita $\Lambda$ for $Z_{\odot}$"
    )
    plt.plot(
        s_LogT, 10.0**s_logLambdaN, label=r"Schure et al. $\Lambda$ for $Z_{\odot}$"
    )
    plt.plot(
        np.log10(gs_T), gs_LambdaZ1, label=r"Gnat & Sternberg $\Lambda$ for $Z_{\odot}$"
    )
    plt.yscale("log")
    plt.ylim(1e-23, 3e-21)
    plt.xlabel(r"$\log T$ (K)")
    plt.ylabel(r"$\Lambda(T)$ [$n_H n_e$ convention]")
    plt.legend()
    plt.tight_layout()
    plt.savefig("cooling_curves_nHneLambda.png")
