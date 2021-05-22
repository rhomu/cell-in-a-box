"""
Simple example of a simulation of a single cell in a box in 1 dimension.

(c) 2021, Romain Mueller, name dot surname at gmail dot com.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as ani


# Finite difference kernels (first and second order derivatives)
d1_kernel = np.array([1.0, -8, 0, +8, -1.0])
d2_kernel = np.array([-1.0, 16.0, -30.0, 16.0, -1.0])

# Simulation parameters:
# - gamma parameter of the cell
gam = 1.0
# - lambda parameter of the cell
lam = 1.0
# - length of the domain
length = 300
# - size of the cell
size = 100
# - strength of the area constraint
mu = 1.0
# - strength of the repulsion with the wall
kappa = 20.0
# - time step
dt = 5e-5
# - coupling constant between polarisation and force
J = 1
# - diffusion constant of the polarisation
D = 30.0
# - number of steps between plotting
nsteps = 500
# - substrate friction
xi = 10.0
# - coupling between polarisation and velocity
alpha = 10.0

# Dofs:
# - phase field (with boundary nodes)
phi = np.zeros(length + 4)
# - cell polarisation
pol = 100.0
# - walls phase field (with boundary nodes)
bdr = np.zeros(length + 4)


def initial_cond():
    """Initial conditions for the simulation."""
    start, end = length // 2 - size // 2, length // 2 + size // 2
    phi[:start] = 0
    phi[start:end] = 1
    phi[end:] = 0

    bdr[:] = 0
    bdr[-32:] = 1
    bdr[:32] = 1


def step(relax=False):
    """Single time step."""
    global phi
    global vel
    global pol

    # periodic boundary conditions
    phi[-2:] = phi[2:4]
    phi[:2] = phi[-4:-2]

    # compute derivatives using the central difference
    nabla_phi = np.convolve(phi, d1_kernel, "same")
    delta_phi = np.convolve(phi, d2_kernel, "same")

    # compute the free energy terms
    dFch_dphi = 8 * gam / lam * phi * (1 - phi) * (1 - 2 * phi) - 2 * gam * lam * delta_phi
    dFarea_dphi = -4 * mu / size * (1 - np.sum(phi[2:-2] ** 2) / size)
    dFrep_dphi = +2 * kappa / lam * phi * bdr ** 2

    # compute force
    F = np.mean(((dFch_dphi - dFarea_dphi - dFrep_dphi) * delta_phi)[2:-2])

    # update polarisation to align with force
    if not relax:
        pol += J * np.abs(F) * (F - pol) + np.sqrt(dt) * D * np.random.randn()

    # free enery terms
    phi -= (dFch_dphi + dFarea_dphi + dFrep_dphi) * dt

    # advection term
    if not relax:
        phi += (alpha * pol + F) / xi * nabla_phi * dt

    return pol, F * nsteps


def init():
    """Animation initialisation."""
    # set fig axes
    ax1.set_xlim(0, length)
    ax1.set_ylim(-0.1, 1.1)
    line1.set_data(np.arange(length), phi[2:-2])
    line2.set_data(np.arange(length), bdr[2:-2])

    return (line1, line2, linev, linef)


def plot(time):
    """Animation plotting."""
    global phi
    global vel
    global all_p
    global all_f

    for _ in range(nsteps):
        p, f = step()

    line1.set_data(np.arange(length), phi[2:-2])

    all_p.append(p)
    all_f.append(f)
    linev.set_data(nsteps * np.arange(len(all_p)), all_p)
    linef.set_data(nsteps * np.arange(len(all_f)), all_f)
    ax2.set_xlim(0, nsteps * len(all_p))
    ax2.set_ylim(min(min(all_p), min(all_f)), max(max(all_p), max(all_f)))
    ax2.legend()

    return (line1, line2, linev, linef)


# relax cell for some time while setting a high friction to remove forces.
initial_cond()
old_xi = xi
xi = 1e100
for _ in range(10 * nsteps):
    step(relax=True)
xi = old_xi  # restore substrate friction coefficient

# initialise figure and start animation
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
(line1,) = ax1.plot([], [], lw=2)
(line2,) = ax1.plot([], [], lw=2, color="r")
(linev,) = ax2.plot([], [], lw=1, color="b", label="polarisation")
(linef,) = ax2.plot([], [], lw=1, color="orange", label="force")
all_p = []
all_f = []

# start animation
a = ani.FuncAnimation(fig, plot, 1000, init)
plt.show()
