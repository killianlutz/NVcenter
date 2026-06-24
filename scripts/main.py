############################################################
################## CONTROL OF NV CENTERS ###################
############################################################
# 2 spins:
from scripts._config import *
# 3 spins:
# from scripts._config_3spins import *

csys = ControlSystem(static_p)
solve_fn = jax.jit(csys.solve_ocp)

##############################
######### SOLVE ##############
##############################
dynamic_p = {"target": U1, "drift": drift}

K = jax.random.split(keys[0], 3)
init_control = (
    0.01*2*jnp.pi*jnp.ones(1), # T
    jax.random.normal(K[0], su_dim), # g
    rand_weights(K[1], neurons[0]), # w -> u
    rand_weights(K[2], neurons[1]), # w -> v
)

control, losses, n_iter = solve_fn(init_control, dynamic_p)
losses = losses[:n_iter + 1] # not supported inside solve_fn (jit-compilation + n_iter dynamic)
print(f"\n Iter: {n_iter} \n Loss: {losses[-1]:.1e} \n Gate time (scaled): {control[0][0]/(2*jnp.pi):.3f}")

##############################
######### POST PROCESS #######
##############################
plot_results(csys, control, dynamic_p, losses, figsize=(16, 6))

pulses = csys.pulses(control, dynamic_p)
propagators = csys.trajectory(control, dynamic_p)
final_propagator = csys.final_state(control, dynamic_p)
loss_after_optimizer = csys.loss(control, dynamic_p)

##############################
#### VALIDATE AND SAVE #######
##############################
loss_check = csys.validate(control, dynamic_p, method='DOP853', atol=1e-9, rtol=1e-6, max_step=1e-2)
print(f"\n Loss validation: {loss_check:.1e}")

csys.save_to_npz("./sims/example.npz", control, dynamic_p)