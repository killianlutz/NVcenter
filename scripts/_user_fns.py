import jax.numpy as jnp
import jax
import matplotlib.pyplot as plt

from src._classes import ControlSystem
from src._quantum import *

###############################
##### AUXILIARY FUNCTIONS #####
###############################
def plot_results(csys:ControlSystem, control, dynamic_p, losses, **kwargs):
    T = control[0]
    Ms = csys.static_p["constraints"]["max_amplitude"]
    ts = csys.static_p["integrator"]["ts"]
    pulses = csys.pulses(control, dynamic_p)  # g -> pulse
    fig, axes = plt.subplots(1, 1 + len(pulses), **kwargs)

    # LOSS
    axes[0].semilogy(losses, '-x', color="k")
    axes[0].set_title(f"Loss | Gate time: {T[0]/(2*jnp.pi):.3f} ($\\times 2\\pi/\\omega_c$)")
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True)

    # PULSES
    for (i, (pulse, M)) in enumerate(zip(pulses, Ms)):
        energy = jnp.linalg.norm(pulse, axis=1)
        ax = axes[i+1]
        ax.plot(ts, pulse, linewidth=3)
        ax.plot(ts, energy, linestyle='--', color='purple', label="$norm$", linewidth=3)
        ax.plot(ts, M*jnp.ones_like(ts), color='k', linewidth=0.75)
        ax.plot(ts, -M*jnp.ones_like(ts), color='k', linewidth=0.75)
        ax.plot(ts, 0*ts, color='k', linewidth=0.75)
        ax.set_xlabel(r"Time $t/T$")
        ax.grid(True)
        ax.legend(loc='upper right')

    plt.tight_layout()
    fig.show()

def runge_kutta(f, x, t, p):
    h = p[-1]["integrator"]["h"]
    k1 = f(x           , t + 0.0*h, p)
    k2 = f(x + 0.5*h*k1, t + 0.5*h, p)
    k3 = f(x + 0.5*h*k2, t + 0.5*h, p)
    k4 = f(x +     h*k3, t + 1.0*h, p)
    return x + h*(k1 + 2*k2 + 2*k3 + k4)/6


def vector_field(U, t, p):
    control, dynamic_p, static_p = p
    networks = static_p["system"]["network"]
    Ms = static_p["constraints"]["max_amplitude"]
    H0 = dynamic_p["drift"]
    Hc = static_p["system"]["ctrl"]
    su_basis = static_p["su_basis"]

    T, g_vec, weights = control[0], control[1], control[2:]
    g = vec_to_matrix(g_vec, su_basis)
    g_conjugate_U = U @ g @ dagger(U)

    def pulses_fn(H, M, network_fn, weight):
        pulses = jax.vmap(
            lambda x, y: jnp.real(trace_dot(x, y)),
            in_axes=(0, None)
        )(H, g_conjugate_U)
        pulses_scaled = pulses * 1 / jnp.linalg.norm(pulses)
        control_hamiltonian = jnp.tensordot(pulses_scaled, H, axes=1)
        amplitude = network_fn(t, weight)
        return M*amplitude*control_hamiltonian

    control_hamiltonian = jnp.sum(
        jnp.stack(jax.tree.map(pulses_fn, Hc, Ms, networks, weights)),
        axis=0
    )

    return (-1j * T) * (H0 + control_hamiltonian) @ U

def loss_fn(x, p):
    # # classical infidelity
    # identity = p[-1]["system"]["initial_state"]
    # return infidelity(x, identity)

    # allows local phases
    x_diag = jnp.diag(x)
    F = jnp.square(jnp.linalg.norm(x_diag))/jnp.size(x_diag)
    return jnp.abs(1 - F)