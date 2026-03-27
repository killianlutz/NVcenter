import jax.numpy as jnp
import jax
import matplotlib.pyplot as plt
from scipy.optimize import newton

from jax.flatten_util import ravel_pytree
from jax.scipy.linalg import expm
from src._classes import ControlSystem
from src._line_search import golden_section
from src._quantum import *
from src._networks import *
key = jax.random.PRNGKey(0)

###############################
##### AUXILIARY FUNCTIONS #####
###############################
def plot_results(csys:ControlSystem, control, dynamic_p, losses):
    T = control[0]
    ts = csys.static_p["integrator"]["ts"]
    pulses = csys.pulses(control, dynamic_p)  # g -> pulse
    energy = jnp.linalg.norm(pulses, axis=1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 8))

    # LOSS
    axes[0].semilogy(losses, '-x', color="k")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True)

    # PULSES
    axes[1].plot(ts, pulses[:, 0], c='b', label="$u_1(t)$", linewidth=3)
    axes[1].plot(ts, pulses[:, 1], c='r', label="$u_2(t)$", linewidth=3)
    axes[1].plot(ts, energy, linestyle='--', color='k', label="$|u(t)|$")
    axes[1].plot(ts, M*jnp.ones_like(ts), color='k', linewidth=0.75)
    axes[1].plot(ts, -M*jnp.ones_like(ts), color='k', linewidth=0.75)
    axes[1].plot(ts, 0*ts, color='k', linewidth=0.75)
    axes[1].set_title(f"Pulses | Gate time: {T[0]/(2*jnp.pi):.3f} (x 2pi)")
    axes[1].set_xlabel(r"Time $t/T$")
    axes[1].grid(True)
    axes[1].legend(loc='upper right')

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
    control, _, static_p = p
    network = static_p["system"]["network"]
    M = static_p["constraints"]["max_amplitude"]
    H0 = static_p["system"]["drift"]
    Hc = static_p["system"]["ctrl"]
    su_basis = static_p["su_basis"]

    T, g_vec, weights = control[0], control[1], control[2]
    g = vec_to_matrix(g_vec, su_basis)
    g_conjugate_U = U @ g @ dagger(U)

    pulses = jax.vmap(
        lambda x, y: jnp.real(trace_dot(x, y)),
        in_axes=(0, None)
    )(Hc, g_conjugate_U)

    pulses_scaled = pulses * 1/jnp.linalg.norm(pulses)
    control_hamiltonian = jnp.tensordot(pulses_scaled, Hc, axes=1)
    a = network(t, weights)
    return (-1j * T) * (H0 + a*M*control_hamiltonian) @ U

def loss_fn(x, p):
    # classical infidelity
    # identity = p[-1]["system"]["initial_state"]
    # return infidelity(x, identity)

    # allows local phases
    x_diag = jnp.diag(x)
    F = 0.25*jnp.square(jnp.linalg.norm(x_diag))
    return jnp.abs(1 - F)

##############################
##### NUMERICAL SCHEME #######
##############################
n = 500
ts = jnp.linspace(0.0, 1.0, n)
h = ts[1] - ts[0]

##############################
##### CONTROL SYSTEM #########
##############################
d = 4
su_dim = d**2 - 1
keys = jax.random.split(key, 100)

mat_basis = basis(d)
su_basis = subasis(d)
S, I, SI = spin_matrices()
drift = SI["zz"]
ctrl = jnp.stack((S["x"], S["y"]))
U0 = jnp.eye(d, dtype=jnp.complex64)
M = 10.0 # maximal control amplitude
neurons = jnp.array([1, 8, 8, 1]) # = jnp.array([1]) for constant control amplitude

###### DYNAMIC ARGUMENTS
U1 = sampleSU(d, keys[1]) # target
dynamic_p = U1

###### STATIC ARGUMENTS
# same pytree structure as the control, one projector per control variable
# if T must be constant, just set the corresponding projector to (lambda T, dT, lr: T)
projector = (
    lambda T, dT, lr: jnp.maximum(T + lr*dT, 0.0),
    lambda g, dg, lr: g + lr*dg,
    lambda w, dw, lr: jax.tree_util.tree_map(lambda x, dx: x + lr*dx, w, dw)
)

static_p = {
    "loss_fn": loss_fn,
    "projector": projector,
    "mat_basis": mat_basis,
    "su_basis": su_basis,
    "constraints": {
        "max_amplitude": M
    },
    "system": {
        "initial_state": U0,
        "drift": drift,
        "ctrl": ctrl,
        "network": network if len(neurons) > 1 else lambda x, _: jnp.ones_like(x)
    },
    "integrator": {
        "h": h,
        "ts": ts,
        "scheme": runge_kutta,
        "vector_field": vector_field
    },
    "optimizer": {
        "normalize_gradient": True,
        "regularization": 1e-3,
        "n_max": 100,
        "abstol_loss": 1e-4,
        "reltol_dist": 1e-4,
        "line_search": {
            "search_fn": golden_section, # signature (f, dynamic, static) -> step, val
            "log_interval": (-4.0, 0.0),
            "abstol": 1e-2,
            "n_max": 200
        }
    }
}
