import yaml
import numpy as np
import matplotlib.pyplot as plt

with open("ascent_session.yaml",'r') as file:
    session = yaml.load(file, Loader=yaml.FullLoader)

def plot_profile(name, skip=40, symmetric=False):
    print(f"plotting {name}...")
    series = session[name]
    profiles = []
    times = []
    bin_centers = []

    for cycle in series.values():
        profiles.append(cycle['attrs']['value']['value'])
        times.append(cycle['time'])
        bins = np.linspace( -106.25, 106.250002125, 21+1 )
        bin_centers.append( 0.5*(bins[:-1] + bins[1:]) )

    plt.figure(figsize=(6,4))
    for i in range(0, len(profiles), skip):
        plt.plot(bin_centers[i], profiles[i], label="{:.1f} Gyr".format(times[i]/1e3))
    plt.plot(bin_centers[-1], profiles[-1], label="{:.1f} Gyr".format(times[-1]/1e3))

    ymax = np.max(np.abs(np.array(profiles)))
    if symmetric:
        plt.ylim(-ymax, ymax)
        plt.plot(bin_centers[-1], np.zeros_like(bin_centers[-1]), '--', color='gray')
    else:
        plt.ylim(0, ymax)
    
    plt.legend(loc='best')
    plt.xlabel("z height (kpc)")
    plt.ylabel(f"{name}")
    plt.tight_layout()
    plt.savefig(name + ".pdf")

plot_profile("drho_profile", symmetric=True)
plot_profile("dK_profile", symmetric=True)
plot_profile("dP_profile", symmetric=True)
plot_profile("Edot_profile", symmetric=True)

plot_profile("drho_rms_profile")
plot_profile("dP_rms_profile")
plot_profile("dK_rms_profile")
plot_profile("dEdot_rms_profile")
