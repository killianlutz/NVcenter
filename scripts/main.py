############################################################
################## CONTROL OF NV CENTERS ###################
############################################################
from scripts._config import *
csys = Magicarp(static_p) if method == "magicarp" else Grape(static_p)
solve_fn = jax.jit(csys.solve_ocp)

def initial_guess(method):
    if method == "magicarp":
        K = jax.random.split(keys[0], 3)
        T = 0.1 * 2 * jnp.pi * jnp.ones(1) # time horizon
        g = jax.random.normal(K[0], su_dim) # initial covector
        theta_u = rand_weights(K[1], jnp.array([1, 5, 5, 1])) # amplitude u
        theta_v = rand_weights(K[2], jnp.array([1, 5, 5, 1])) # amplitude v
        return T, g, theta_u, theta_v

    elif method == "grape":
        n_pieces = 10*(d + 1)
        T = 0.1 * 2 * jnp.pi * jnp.ones(1)  # time horizon
        u = 1e-2 * jnp.ones((n_pieces, jnp.size(ctrl[0], 0))) # u
        v = 1e-2 * jnp.ones((n_pieces, jnp.size(ctrl[1], 0))) # v
        return T, u, v
    else:
        raise ValueError("only available methods: magicarp, grape")

##############################
######### SOLVE ##############
##############################
init_control = initial_guess(method)
dynamic_p = {"target": U1, "drift": drift}
control, losses, n_iter = solve_fn(init_control, dynamic_p)
losses = losses[:n_iter + 1] # not supported inside solve_fn (jit-compilation + n_iter dynamic)
print(f"\n Optimization results: \n Iter: {n_iter} \n Loss: {losses[-1]:.1e} \n Gate time (scaled): {control[0][0]/(2*jnp.pi):.3f}")

##############################
######### POST PROCESS #######
##############################
csys.plot_results(control, dynamic_p, 1e-15 + losses, figsize=(16, 6))

pulses = csys.pulses(control, dynamic_p)
propagators = csys.trajectory(control, dynamic_p)
final_propagator = csys.final_state(control, dynamic_p)
loss_after_optimizer = csys.loss(control, dynamic_p)

##############################
#### VALIDATE AND SAVE #######
##############################
loss_check = csys.validate(control, dynamic_p)
time_pulse_fns_pair = (control[0], csys.pulse_fns(control, dynamic_p))
loss_check_concrete = csys.validate_concrete_pulses(time_pulse_fns_pair, dynamic_p)
print(f"\n Loss validation: \n Direct: {loss_check:.1e} \n Concrete pulses: {loss_check_concrete:.1e}")

csys.save_to_npz("./sims/example.npz", control, dynamic_p)