# _config.py contains the setup
from scripts._config import *
print(jax.devices())

csys = ControlSystem(static_p)
solve_fn = jax.jit(csys.solve_ocp)

##############################
######### SOLVE ##############
##############################
init_control = (
    0.25*jnp.ones(1),
    jax.random.normal(keys[25], su_dim)
)

U1 = CNOT() # or random gate using: sampleSU(d, keys[50])
dynamic_p = U1

control, losses, n_iter = solve_fn(init_control, dynamic_p)
losses = losses[:n_iter + 1] # not supported inside solve_fn (jit-compilation + n_iter dynamic)
print(f"\n Iter: {n_iter} \n Loss: {losses[-1]:.1e} \n Gate time: {control[0][0]:.2f}")

##############################
#### VALIDATE AND SAVE #######
##############################
loss_check = csys.validate(control, dynamic_p, method='DOP853', max_step=1e-2)
print(f"\n Loss validation: {loss_check:.1e}")

csys.save_to_npz("./sims/example.npz", control, dynamic_p)
##############################
######### POST PROCESS #######
##############################
# plot_results(csys, control, dynamic_p, losses)

pulses = csys.pulses(control, dynamic_p)
propagators = csys.trajectory(control, dynamic_p)
final_propagator = csys.final_state(control, dynamic_p)
loss_after_optimizer = csys.loss(control, dynamic_p)
