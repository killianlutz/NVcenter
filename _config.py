import jax.numpy as jnp
import jax
import matplotlib.pyplot as plt
from scipy.optimize import newton

from jax.flatten_util import ravel_pytree
from _classes import ControlSystem
from _line_search import golden_section
from _quantum import *
key = jax.random.PRNGKey(0)

###############################
##### AUXILIARY FUNCTIONS #####
###############################
def plot_results(csys:ControlSystem, control, dynamic_p, losses):
    ts = csys.static_p["integrator"]["ts"]
    t_to_idx = csys.static_p["t_to_idx"]

    T, a, _ = control
    A = jax.vmap(lambda t: a[t_to_idx(t)])(ts[:-1])
    pulses = csys.pulses(control, dynamic_p)  # g -> pulse, w/o a(t)
    energy = jnp.linalg.norm(pulses, axis=1)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # LOSS
    axes[0, 1].semilogy(losses, linestyle='-', color="k")
    axes[0, 1].set_title("Loss")
    axes[0, 1].set_xlabel("Iteration")
    axes[0, 1].set_ylabel("Infidelity")
    axes[0, 1].grid(True)

    # AMPLITUDE
    sc = axes[0, 0].scatter(ts[:-1], A, c=jnp.sign(A), cmap='bwr', edgecolor='k')
    axes[0, 0].hlines(1.0, 0.0, 1.0, color='r', linestyles='dashed')
    axes[0, 0].hlines(0.0, 0.0, 1.0, color='k', linestyles='solid')
    axes[0, 0].set_title(f"$T$ = {T[0]:.2f}")
    axes[0, 0].set_xlabel(r"Time $t/T$")
    axes[0, 0].set_ylabel(r"Amplitude $a(t)$")
    axes[0, 0].grid(True)
    fig.colorbar(sc, ax=axes[0, 0], label="sign(a)")

    # PULSES
    axes[1, 0].scatter(ts, pulses[:, 0], c='b', label="$u_1(t)$")
    axes[1, 0].scatter(ts, pulses[:, 1], c='r', label="$u_2(t)$")
    axes[1, 0].plot(ts, energy, color='k', label="$|u(t)|$")
    axes[1, 0].set_title(f"Pulses w/o a(t)")
    axes[1, 0].set_xlabel(r"Time $t/T$")
    axes[1, 0].grid(True)
    axes[1, 0].legend()

    # NORM PULSES
    axes[1, 1].plot(ts[:-1], energy[:-1]*jnp.abs(A), color='k')
    axes[1, 1].set_title(f"Effective pulses norm $|a(t)|*|u(t)|$")
    axes[1, 1].set_xlabel(r"Time $t/T$")
    axes[1, 1].grid(True)

    plt.tight_layout()
    fig.show()


def vector_field(U, t, p):
    control, _, static_p = p
    H0 = static_p["system"]["drift"]
    Hc = static_p["system"]["ctrl"]
    su_basis = static_p["su_basis"]
    t_to_idx = static_p["t_to_idx"]

    T, a, g_vec = control
    a_t = a[t_to_idx(t)]
    g = vec_to_matrix(g_vec, su_basis)

    g_conjugate_U = jnp.einsum('ij, jk, kl -> il', U, g, dagger(U))
    pulse_times_hamiltonian = jax.vmap(
        lambda x, y: jnp.real(trace_dot(x, y))*x,
        in_axes=(0, None)
    )(Hc, g_conjugate_U)
    control_hamiltonian = jnp.sum(pulse_times_hamiltonian, axis=0)

    return -1j * T * (H0 + a_t*control_hamiltonian) @ U

def loss_fn(x, p):
    U1 = p[1]
    return infidelity(x, U1)

def runge_kutta(f, x, t, p):
    h = p[-1]["integrator"]["h"]
    k1 = f(x           , t + 0.0*h, p)
    k2 = f(x + 0.5*h*k1, t + 0.5*h, p)
    k3 = f(x + 0.5*h*k2, t + 0.5*h, p)
    k4 = f(x +     h*k3, t + 1.0*h, p)
    return x + h*(k1 + 2*k2 + 2*k3 + k4)/6

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

###### CONTROL
time_horizon = jnp.ones(1)
amplitude = jnp.ones(len(ts) - 1)
g_vec = jax.random.normal(keys[0], su_dim)
init_control = (time_horizon, amplitude, g_vec)

###### DYNAMIC ARGUMENTS
U1 = sampleSU(d, keys[1]) # target
dynamic_p = U1

###### STATIC ARGUMENTS
M = 50.0 # roughly the maximal control amplitude
def proj_a(a, da, lr):
    b = a + lr*da
    return (jnp.abs(b) <= M)*b + (jnp.abs(b) > M)*M*b/abs(b)

# same pytree structure as the control, one projector per control variable
# if the control is not to be updated, set the corresponding projector to (x, dx, lr) -> x
projector = (
    lambda T, dT, lr: jnp.maximum(T + lr*dT, 0.0),
    proj_a,
    lambda g, dg, lr: g + lr*dg
)

t_to_idx = lambda t: 1 + jnp.array(jnp.floor(t/h - 1/2), dtype=int)

static_p = {
    "loss_fn": loss_fn,
    "projector": projector,
    "t_to_idx": t_to_idx,
    "mat_basis": mat_basis,
    "su_basis": su_basis,
    "system": {
        "initial_state": U0,
        "drift": drift,
        "ctrl": ctrl
    },
    "integrator": {
        "h": h,
        "ts": ts,
        "scheme": runge_kutta,
        "vector_field": vector_field
    },
    "optimizer": {
        "normalize_gradient": False,
        "regularization": 1e-3,
        "n_max": 500,
        "abstol_loss": 1e-5,
        "reltol_dist": 1e-5,
        "line_search": {
            "search_fn": golden_section, # signature (f, dynamic, static) -> step, val
            "log_interval": (-4.0, 0.0),
            "abstol": 1e-2,
            "n_max": 200
        }
    },
    "root_finding": {
        "reltol_dist": 1e-8,
        "n_max": 1_000
    }
}
