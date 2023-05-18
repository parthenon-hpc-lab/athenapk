import yaml
import numpy as np
import matplotlib.pyplot as plt

def plot_profile(name, session=None, log=False):
    data = session[name]
    profiles = []
    times = []
    bin_centers = []

    # loop over output times
    for cycle in data.values():
        # compute bins
        bin_props = cycle['attrs']['bin_axes']['value']['cell_radius']
        xmin = bin_props['min_val']
        xmax = bin_props['max_val']
        nbins = bin_props['num_bins']
        bins = np.linspace(xmin, xmax, nbins+1)
        bin_centers.append(0.5*(bins[:-1] + bins[1:]))

        # get profile
        profiles.append(np.asarray(
            cycle['attrs']['value']['value'], dtype=np.float64))
        times.append(cycle['time'])

    plt.figure(figsize=(6, 4))
    for i in range(len(profiles)):
        plt.plot(bin_centers[i], profiles[i],
                 label=r"$t = {:.2f}$".format(times[i]))

    plt.xlim(0, 0.5)
    if (log):
        plt.yscale('log')
    plt.legend(loc='upper right')
    plt.ylabel(f"{name}")
    plt.xlabel(r"radius")
    plt.tight_layout()
    plt.savefig(name + ".png")

if __name__ == "__main__":
    # plot profiles produced by Ascent DataBinning function
    
    with open("ascent_session.yaml", 'r') as file:
        session = yaml.load(file, Loader=yaml.FullLoader)
        plot_profile("density_profile", session=session)
        plot_profile("pressure_profile", session=session, log=True)
        plot_profile("vorticity_profile", session=session)

