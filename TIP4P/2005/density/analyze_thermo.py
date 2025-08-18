import numpy as np
import matplotlib.pyplot as plt
import glob

N = 512  # Number of water molecules
kB = 8.617333e-5  # eV/K
Na = 6.022e23  # mol^-1
eV_to_cal = 23.0609  # eV/cal
mass = N * (15.9994 + 2 * 1.00794) / Na  # g/mol to g

temps = []
rho, rho_err = [], []
Cp, Cp_err = [], []
kappa, kappa_err = [], []
alpha, alpha_err = [], []

for fname in sorted(glob.glob("thermo_*.txt")):
    T = float(fname.split("_")[1].split(".")[0])
    data = np.loadtxt(fname)
    temp = data[:, 0]
    press = data[:, 1]
    vol = data[:, 2]
    etot = data[:, 3]
    enthalpy = data[:, 4]

    V = vol.mean()
    H = enthalpy.mean()
    dV = vol - V
    dH = enthalpy - H

    Cp_val = np.mean(dH**2) / (kB * T**2)
    Cp_err_val = np.std(dH**2) / (kB * T**2) / np.sqrt(len(enthalpy))
    Cp.append(Cp_val * eV_to_cal)
    Cp_err.append(Cp_err_val * eV_to_cal)

    kappa_val = np.mean(dV**2) / (kB * T * V)
    kappa_err_val = np.std(dV**2) / (kB * T * V) / np.sqrt(len(vol))
    kappa.append(kappa_val)
    kappa_err.append(kappa_err_val)

    alpha_val = np.mean(dH * dV) / (kB * V * T**2)
    alpha_err_val = np.std(dH * dV) / (kB * V * T**2) / np.sqrt(len(vol))
    alpha.append(alpha_val)
    alpha_err.append(alpha_err_val)

    temps.append(T)

    rho_mean = mass / (V * 1e-24)  # Convert volume from Angstrom^3 to cm^3
    rho_std = (vol.std() / V) * rho_mean
    rho.append(rho_mean)
    rho_err.append(rho_std)


def plot_with_error(x, y, yerr, ylabel):
    plt.errorbar(x, y, yerr=yerr, fmt="o-", capsize=3)
    plt.xlabel("Temperature (K)")
    plt.ylabel(ylabel)
    plt.grid()
    plt.tight_layout()
    # plt.savefig(f"{ylabel.replace(' ', '_').lower()}.png")
    plt.show()


plot_with_error(temps, rho, rho_err, "Density (g/cm^3)")
plot_with_error(temps, Cp, Cp_err, "Heat Capacity (cal/mol/K)")
plot_with_error(temps, kappa, kappa_err, "Isothermal Compressibility (1/atm)")
plot_with_error(temps, alpha, alpha_err, "Thermal Expansion Coefficient (1/K)")
