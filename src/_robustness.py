from scripts._config import *
from diffrax import diffeqsolve, ODETerm, Dopri8, PIDController
import jax.numpy as jnp
import jax

def pulse_fns(pulses):
    # take discrete pulses and identify them to piecewise constant functions
    def zoh(weights):
        n_pieces = jnp.size(weights, 0)
        return lambda t: piecewise_cst_interp(t, weights, n_pieces)

    return jax.tree.map(zoh, pulses)

def robustness(time_pulse_fns_pair, hamiltonians, dt0=1e-1, solver=Dopri8(),
                             stepsize_controller=PIDController(atol=1e-7, rtol=1e-5), **kwargs):
    U_final = diffeqsolve(
        ODETerm(vector_field_schrodinger),
        t0=0.0, t1=1.0, dt0=dt0,
        y0=jnp.eye(jnp.size(hamiltonians[0], 0), dtype=jnp.complex64),
        args=(*hamiltonians, *time_pulse_fns_pair),
        solver=solver,
        stepsize_controller=stepsize_controller,
        **kwargs
    ).ys[0, ...]

    return loss_fn(dagger(U1) @ U_final, None)

def vmap_robustness(Tuv, **kwargs):
    Tuv = jax.tree.map(jnp.array, (*Tuv, ))
    time_pulse_fns_pair = (Tuv[0], pulse_fns(Tuv[1]))
    ref_loss = robustness(time_pulse_fns_pair, (drift, ctrl))

    def fn(variation):
        hamiltonians = ((1 + variation) * drift, ctrl)
        return robustness(time_pulse_fns_pair, hamiltonians, **kwargs) - ref_loss

    return jax.vmap(fn)