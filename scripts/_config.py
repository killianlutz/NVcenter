import jax.numpy as jnp
import jax
import matplotlib.pyplot as plt
import lineax as lx

from jax.flatten_util import ravel_pytree
from src._classes import ControlSystem
from src._line_search import golden_section
from src._quantum import *
from src._networks import *
key = jax.random.PRNGKey(0)

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
    F = 0.25*jnp.square(jnp.linalg.norm(x_diag))
    return jnp.abs(1 - F)

##############################
##### NUMERICAL SCHEME #######
##############################
n = 500
ts = jnp.linspace(0.0, 1.0, n)
h = ts[1] - ts[0]

##############################
##### PHYSICAL PARAMETERS ####
##############################
# We rescale the time variable by the characteristic (angular) frequency:
#           t -> tau where tau = char_freq*t
# Thus we optimize char_freq*T where T is the physical time (in seconds).
char_freq = 1e6 # MHz

omega_S = 1e9/char_freq # zero-splitting
omega_I = 1e-4*omega_S # zero-splitting
A_parallel = 1e6/char_freq # hyperfine coupling
Omega_R = 1e6/char_freq # Rabi frequency

##############################
##### CONTROL SYSTEM #########
##############################
d = 4
su_dim = d**2 - 1
keys = jax.random.split(key, 100)
mat_basis = basis(d)
su_basis = subasis(d)
S, I, SI = spin_matrices()
U0 = jnp.eye(d, dtype=jnp.complex64)
drift = A_parallel*SI["zz"] + omega_I*I["z"]
electronic_ctrl = jnp.stack((S["x"], S["y"]))
nuclear_ctrl = I["x"][None, :, :]

ctrl = (electronic_ctrl, nuclear_ctrl)
Ms = (1e0*Omega_R, 1e0*Omega_R) # maximal control amplitude
neurons = (jnp.array([1, 8, 8, 1]), jnp.array([1, 8, 8, 1])) # = jnp.array([1]) for constant control amplitude
networks = jax.tree.map(network_or_not, neurons)

###### DYNAMIC ARGUMENTS
U1 = sampleSU(d, keys[1]) # target
dynamic_p = {"target": U1, "drift": drift}

###### STATIC ARGUMENTS
# same pytree structure as the control, one projector per control variable
# if T must be constant, just set the corresponding projector to (lambda T, dT, lr: T)
projector = (
    lambda T, dT, lr: jnp.maximum(T + lr*dT, 0.0),
    lambda g, dg, lr: g + lr*dg,
    lambda w, dw, lr: jax.tree.map(lambda x, dx: x + lr * dx, w, dw),
    lambda w, dw, lr: jax.tree.map(lambda x, dx: x + lr * dx, w, dw),
)

static_p = {
    "loss_fn": loss_fn,
    "projector": projector,
    "mat_basis": mat_basis,
    "su_basis": su_basis,
    "constraints": {
        "max_amplitude": Ms
    },
    "system": {
        "initial_state": U0,
        "ctrl": ctrl,
        "network": networks,
    },
    "integrator": {
        "h": h,
        "ts": ts,
        "scheme": runge_kutta,
        "vector_field": vector_field
    },
    "optimizer": {
        "normalize_gradient": True,
        "n_max": 100,
        "abstol_loss": 1e-6,
        "reltol_dist": 1e-4,
        "line_search": {
            "search_fn": golden_section, # signature (f, dynamic, static) -> step, val
            "log_interval": (-4.0, 0.0),
            "abstol": 1e-2,
            "n_max": 200
        },
        "least_squares": {
            "regularization": 1e-2,
            "is_iterative": True,
            "tags": (lx.positive_semidefinite_tag,),
            "iterative_solver": lx.CG(atol=1e-8, rtol=1e-3, max_steps=500),
            "direct_solver": lx.QR()
        }
    }
}
