# _config.py contains the setup
from _config import *

csys = ControlSystem(static_p)
solve_fn = jax.jit(csys.solve_ocp)

##############################
######### SOLVE ##############
##############################

init_control = (
    5*jnp.ones(1),
    jnp.ones(len(ts) - 1),
    jax.random.normal(keys[10], su_dim)
)

U1 = CNOT()
dynamic_p = U1

control, losses, n_iter = solve_fn(init_control, dynamic_p)
losses = losses[:n_iter + 1] # not supported inside solve_fn (jit-compilation + n_iter dynamic)
print(f"\n Iter: {n_iter} \n Loss: {losses[-1]:.1e} \n Gate time: {control[0][0]:.2f}")

##############################
######### POST PROCESS #######
##############################
plot_results(csys, control, dynamic_p, losses)

pulses = csys.pulses(control, dynamic_p) # w/o amplitude a(t)
propagators = csys.trajectory(control, dynamic_p)
final_propagator = csys.final_state(control, dynamic_p)
loss_after_optimizer = csys.loss(control, dynamic_p)