import lineax as lx

from src._line_search import golden_section
from src._networks import *
key = jax.random.PRNGKey(0)
keys = jax.random.split(key, 100)

from scripts._user_fns import *

##############################
######## METHOD CHOICE #######
##############################
# method = "grape"
method = "magicarp"

##############################
##### PHYSICAL PARAMETERS ####
##############################
# We rescale the time variable by the characteristic (angular) frequency:
#           t -> tau where tau = char_freq*t
# Thus we optimize char_freq*T where T is the physical time (in seconds).
char_freq = 1e6 # MHz
n_nuclei = 1 # nuclear spins

omega_S = 1e9/char_freq # zero-splitting
omega_I = 1e-4*omega_S # zero-splitting: nuclear (same for every nuclei)
Omega_Re = 1e6/char_freq # Rabi frequency
Omega_Rn = Omega_Re
A_parallels = jnp.array([1e6/char_freq for _ in range(n_nuclei)]) # hyperfine coupling for each nuclei

Ms = (1e1*Omega_Re, 1e1*Omega_Rn) # maximal control amplitudes
model = nvcenter_model(n_nuclei, A_parallels)
target_gate = electron_flip_conditional_nuclear((1, ), n_nuclei) # electron flip conditional first nuclear spin


##############################
##### NUMERICAL SCHEME #######
##############################
n = 1_000
ts = jnp.linspace(0.0, 1.0, n)
h = ts[1] - ts[0]

##############################
##### CONTROL SYSTEM #########
##############################
d = 2**(n_nuclei + 1) # electrons + nuclei
su_dim = d**2 - 1
mat_basis = basis(d)
su_basis = subasis(d)
U0 = jnp.eye(d, dtype=jnp.complex64)
drift, ctrl = model[0], model[1:]

###### DYNAMIC ARGUMENTS
U1 = target_gate
dynamic_p = {"target": U1, "drift": drift}

###### STATIC ARGUMENTS
static_p = {
    "loss_fn": loss_fn,
    "mat_basis": mat_basis,
    "su_basis": su_basis,
    "constraints": {
        "max_amplitude": Ms
    },
    "system": {
        "initial_state": U0,
        "ctrl": ctrl
    },
    "integrator": {
        "h": h,
        "ts": ts,
        "scheme": runge_kutta
    },
    "optimizer": {
        "normalize_gradient": True,
        "n_max": 100,
        "abstol_loss": 1e-7,
        "reltol_dist": 1e-6,
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
