import jax.numpy as jnp
import jax
import matplotlib.pyplot as plt

from src._classes import ControlSystem
from src._quantum import *
from src._networks import network, piecewise_cst_interp, normalize_if_not_zero, proj_ball

###############################
##### AUXILIARY FUNCTIONS #####
###############################
def runge_kutta(f, x, t, p):
    h = p[-1]["integrator"]["h"]
    k1 = f(x           , t + 0.0*h, p)
    k2 = f(x + 0.5*h*k1, t + 0.5*h, p)
    k3 = f(x + 0.5*h*k2, t + 0.5*h, p)
    k4 = f(x +     h*k3, t + 1.0*h, p)
    return x + h*(k1 + 2*k2 + 2*k3 + k4)/6


def loss_fn(x, p):
    # second argument is p = (control, dynamic_p, static_p)

    # # classical infidelity
    # identity = p[-1]["system"]["initial_state"]
    # return infidelity(x, identity)

    # allows local phases
    x_diag = jnp.diag(x)
    F = jnp.square(jnp.linalg.norm(x_diag))/jnp.size(x_diag)
    return jnp.abs(1 - F)


def method_specific_fn(method):
    # functions defined below
    if method == "MAGICARP":
        return plot_results_MAGICARP, vector_field_MAGICARP, projector_MAGICARP, params_to_fn_MAGICARP
    elif method == "GRAPE":
        return plot_results_GRAPE, vector_field_GRAPE, projector_GRAPE, params_to_fn_GRAPE
    else:
        raise ValueError("only available methods: MAGICARP, GRAPE")

###############################
########## MAGICARP ###########
###############################
def plot_results_MAGICARP(csys:ControlSystem, control, dynamic_p, losses, **kwargs):
    T = control[0]
    Ms = csys.static_p["constraints"]["max_amplitude"]
    ts = csys.static_p["integrator"]["ts"]
    pulses = csys.pulses(control, dynamic_p)  # g -> pulse
    fig, axes = plt.subplots(1, 1 + len(pulses), **kwargs)

    # LOSS
    axes[0].semilogy(losses, '-x', color="k")
    axes[0].set_title(f"Gate time: $T =${T[0]/(2*jnp.pi):.3f} ($\\times 2\\pi/\\omega_c$)")
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

def vector_field_MAGICARP(U, t, p):
    control, dynamic_p, static_p = p
    params_to_fns = static_p["system"]["params_to_fns"]
    Ms = static_p["constraints"]["max_amplitude"]
    H0 = dynamic_p["drift"]
    Hc = static_p["system"]["ctrl"]
    su_basis = static_p["su_basis"]

    T, g_vec, weights = control[0], control[1], control[2:]
    g = vec_to_matrix(g_vec, su_basis)
    g_conjugate_U = U @ g @ dagger(U)

    def pulses_fn(H, M, params_to_fn, weight):
        pulses = jax.vmap(
            lambda x, y: jnp.real(trace_dot(x, y)),
            in_axes=(0, None)
        )(H, g_conjugate_U)
        pulses_scaled = pulses/jnp.linalg.norm(pulses)
        control_hamiltonian = jnp.tensordot(pulses_scaled, H, axes=1)
        amplitude = params_to_fn(t, weight)
        return M*amplitude*control_hamiltonian

    control_hamiltonian = jnp.sum(
        jnp.stack(jax.tree.map(pulses_fn, Hc, Ms, params_to_fns, weights)),
        axis=0
    )

    return (-1j * T) * (H0 + control_hamiltonian) @ U

def projector_MAGICARP():
    # same pytree structure as the control, one projector per control variable
    # if T must be constant, just set the corresponding projector to (lambda T, dT, lr: T)
    return (
        lambda T, dT, lr: jnp.maximum(T + lr * dT, 0.0),
        lambda g, dg, lr: g + lr * dg,
        lambda w, dw, lr: jax.tree.map(lambda x, dx: x + lr * dx, w, dw),
        lambda w, dw, lr: jax.tree.map(lambda x, dx: x + lr * dx, w, dw),
    )

def params_to_fn_MAGICARP(t, weights):
   return network(t, weights)

###############################
########## GRAPE ##############
###############################
def plot_results_GRAPE(csys:ControlSystem, control, dynamic_p, losses, **kwargs):
    T = control[0]
    Ms = csys.static_p["constraints"]["max_amplitude"]
    ts = csys.static_p["integrator"]["ts"]
    def fn(network_fn, weights):
        return jax.vmap(network_fn, in_axes=(0, None))(ts, weights)
    pulses = jax.tree.map(fn, csys.static_p["system"]["params_to_fns"], control[1:])
    fig, axes = plt.subplots(1, 1 + len(pulses), **kwargs)

    # LOSS
    axes[0].semilogy(losses, '-x', color="k")
    axes[0].set_title(f"Gate time: $T =${T[0]/(2*jnp.pi):.3f} ($\\times 2\\pi/\\omega_c$)")
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True)

    # PULSES
    for (i, (pulse, M)) in enumerate(zip(pulses, Ms)):
        energy = jnp.linalg.norm(pulse, axis=1)
        ax = axes[i+1]
        ax.plot(ts, M*pulse, linewidth=3)
        ax.plot(ts, M*energy, linestyle='--', color='purple', label="$norm$", linewidth=3)
        ax.plot(ts, M*jnp.ones_like(ts), color='k', linewidth=0.75)
        ax.plot(ts, -M*jnp.ones_like(ts), color='k', linewidth=0.75)
        ax.plot(ts, 0*ts, color='k', linewidth=0.75)
        ax.set_xlabel(r"Time $t/T$")
        ax.grid(True)
        ax.legend(loc='upper right')

    plt.tight_layout()
    fig.show()

def vector_field_GRAPE(U, t, p):
    control, dynamic_p, static_p = p
    params_to_fns = static_p["system"]["params_to_fns"]
    Ms = static_p["constraints"]["max_amplitude"]
    H0 = dynamic_p["drift"]
    Hc = static_p["system"]["ctrl"]

    T, weights = control[0], control[1:]

    def pulses_fn(H, M, params_to_fn, weight):
        pulses = params_to_fn(t, weight)
        control_hamiltonian = jnp.tensordot(pulses, H, axes=1)
        return M * control_hamiltonian

    control_hamiltonian = jnp.sum(
        jnp.stack(jax.tree.map(pulses_fn, Hc, Ms, params_to_fns, weights)),
        axis=0
    )

    return (-1j * T) * (H0 + control_hamiltonian) @ U

def projector_GRAPE():
    return (
        lambda T, dT, lr: jnp.maximum(T + lr*dT, 0.0),
        lambda u, du, lr: u + lr*du,
        lambda v, dv, lr: v + lr*dv,
    )

def params_to_fn_GRAPE(t, weights):
    n_pieces = jnp.size(weights, 0)

    return proj_ball(piecewise_cst_interp(t, weights, n_pieces))