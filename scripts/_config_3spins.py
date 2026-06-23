import jax.numpy as jnp
import jax
import matplotlib.pyplot as plt
import lineax as lx
from functools import reduce

from jax.flatten_util import ravel_pytree
from src._classes import ControlSystem
from src._line_search import golden_section
from src._quantum import *
from src._networks import *
key = jax.random.PRNGKey(0)

from scripts._user_fns import plot_results, runge_kutta, vector_field, loss_fn

def model_parameters():
    Id = jnp.eye(2, dtype=jnp.complex64)
    P = jax.tree.map(lambda x: 0.5 * x, pauli_matrices())
    Sx = reduce(jnp.kron, [P["x"], Id, Id])
    Sy = reduce(jnp.kron, [P["y"], Id, Id])
    Iz1 = reduce(jnp.kron, [Id, P["z"], Id])
    Iz2 = reduce(jnp.kron, [Id, Id, P["z"]])
    Ix1 = reduce(jnp.kron, [Id, P["x"], Id])
    Ix2 = reduce(jnp.kron, [Id, Id, P["x"]])
    SzIz1 = reduce(jnp.kron, [P["z"], P["z"], Id])
    SzIz2 = reduce(jnp.kron, [P["z"], Id, P["z"]])
    drift = (
            A_parallel1 * SzIz1 + A_parallel2 * SzIz2
            + omega_I1 * Iz1 + omega_I2 * Iz2
    )
    electronic_ctrl = jnp.stack((Sx, Sy))
    nuclear_ctrl = Ix1[None, :, :] + Ix2[None, :, :]
    return drift, electronic_ctrl, nuclear_ctrl

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
omega_I1 = 1e-4*omega_S # zero-splitting: nuclear 1
omega_I2 = 1e-3*omega_S # zero-splitting: nuclear 2
A_parallel1 = 1e6/char_freq # hyperfine coupling: electron - nuclear 1
A_parallel2 = 1e5/char_freq # hyperfine coupling: electron - nuclear 2
Omega_R = 1e6/char_freq # Rabi frequency

##############################
##### CONTROL SYSTEM #########
##############################
n_nuclei = 2 # nuclear spins
d = 2**(n_nuclei + 1) # electrons + nuclei
su_dim = d**2 - 1
keys = jax.random.split(key, 100)
mat_basis = basis(d)
su_basis = subasis(d)
S, I, SI = spin_matrices()
U0 = jnp.eye(d, dtype=jnp.complex64)

drift, electronic_ctrl, nuclear_ctrl = model_parameters()
ctrl = (electronic_ctrl, nuclear_ctrl)
Ms = (
    1e1*Omega_R,
    1e0*Omega_R
) # maximal control amplitude
neurons = (
    jnp.array([1, 8, 8, 1]),
    jnp.array([1, 8, 8, 1])
) # = jnp.array([1]) for constant control amplitude
networks = jax.tree.map(network_or_not, neurons)

###### DYNAMIC ARGUMENTS
U1 = electron_flip_conditional_nuclear((1, 2), n_nuclei) # toffoli
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
            "regularization": 1e-3,
            "is_iterative": False,
            "tags": (lx.positive_semidefinite_tag,),
            "iterative_solver": lx.CG(atol=1e-8, rtol=1e-3, max_steps=500),
            "direct_solver": lx.QR()
        }
    }
}
