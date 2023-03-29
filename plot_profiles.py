import yaml
import numpy as np
import matplotlib.pyplot as plt

with open("ascent_session.yaml",'r') as file:
    session = yaml.load(file, Loader=yaml.FullLoader)

def plot_profile(name, skip=10):
    print(f"plotting {name}...")
    series = session[name]
    profiles = []
    times = []
    bin_centers = []

    for cycle in series.values():
        profiles.append(cycle['attrs']['value']['value'])
        times.append(cycle['time'])
        bins = np.linspace( -106.25, 106.250002125, 20+1 )
        bin_centers.append( 0.5*(bins[:-1] + bins[1:]) )

    plt.figure(figsize=(6,4))
    for i in range(0, len(profiles), skip):
        plt.plot(bin_centers[i], profiles[i], label=times[i])

    #plt.legend(loc='best')
    plt.xlabel("z height (kpc)")
    plt.ylabel(f"{name}")
    plt.tight_layout()
    plt.savefig(name + ".pdf")

plot_profile("drho_profile")
plot_profile("dK_profile")
plot_profile("dP_profile")

plot_profile("drho_rms_profile")
plot_profile("dP_rms_profile")
plot_profile("dK_rms_profile")
