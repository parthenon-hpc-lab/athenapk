import yaml
import numpy as np
import matplotlib.pyplot as plt

with open("ascent_session.yaml", "r") as file:
    session = yaml.load(file, Loader=yaml.FullLoader)


def plot_profile(name, skip=10, symmetric=False):
    print(f"plotting {name}...")
    series = session[name]
    profiles = []
    times = []
    bin_centers = []

    for cycle in series.values():
        profiles.append(np.asarray(cycle["attrs"]["value"]["value"], dtype=np.float64))
        times.append(cycle["time"])
        bins = np.linspace(-106.25, 106.250002125, 64 + 1)
        bin_centers.append(0.5 * (bins[:-1] + bins[1:]))

    plt.figure(figsize=(6, 4))
    for i in range(0, len(profiles), skip):
        plt.plot(bin_centers[i], profiles[i], label="{:.1f} Gyr".format(times[i] / 1e3))

    # plt.plot(bin_centers[-1], profiles[-1], label="{:.1f} Gyr".format(times[-1]/1e3))

    plt.xlim(-100, 100)
    plt.legend(loc="upper left")
    plt.xlabel("z height (kpc)")
    plt.ylabel(f"{name}")
    plt.tight_layout()
    plt.savefig(name + ".pdf")


def plot_timeseries(name=None):
    print(f"plotting timeseries of {name}...")
    series = session[name]
    values = []
    times = []
    for cycle in series.values():
        values.append(cycle["value"])
        times.append(cycle["time"])

    plt.figure(figsize=(6, 4))
    plt.plot(np.array(times) / 1.0e3, values)
    plt.xlabel("time (Gyr)")
    plt.ylabel(f"{name}")
    plt.tight_layout()
    plt.savefig(name + ".pdf")


def plot_history(name=None):
    """Plot timeseries from history file."""
    print(f"plotting timeseries of {name}...")
    history = np.loadtxt("parthenon.hst")
    plt.figure(figsize=(6, 4))
    plt.plot(history[:, 0] / 1e3, history[:, 6], label="kinetic energy")
    plt.plot(
        history[:, 0] / 1e3, history[:, 7], "--", color="black", label="total energy"
    )
    plt.yscale("log")
    plt.xlabel("time (Gyr)")
    plt.ylabel(f"{name}")
    plt.tight_layout()
    plt.savefig(name + ".pdf")


plot_profile("drho_profile", symmetric=True)
plot_profile("dK_profile", symmetric=True)
plot_profile("dP_profile", symmetric=True)
# plot_profile("Edot_profile", symmetric=True)

plot_profile("drho_rms_profile")
plot_profile("dP_rms_profile")
plot_profile("dK_rms_profile")
# plot_profile("grav_phi_profile")
# plot_profile("density_profile")
# plot_profile("pressure_profile")
# plot_profile("dEdot_rms_profile")

plot_timeseries("total_entropy")
plot_history("kinetic energy")
