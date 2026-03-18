# _config.py contains the setup
from _config import *

csys = ControlSystem(static_p)
solve_fn = jax.jit(csys.solve_ocp)

##############################
######### SOLVE ##############
##############################

init_control = (
    0.25*jnp.ones(1),
    jax.random.normal(keys[10], su_dim)
)

U1 = CNOT() # or random gate using: sampleSU(d, keys[50])
dynamic_p = U1

control, losses, n_iter = solve_fn(init_control, dynamic_p)
losses = losses[:n_iter + 1] # not supported inside solve_fn (jit-compilation + n_iter dynamic)
print(f"\n Iter: {n_iter} \n Loss: {losses[-1]:.1e} \n Gate time: {control[0][0]:.2f}")

##############################
######### POST PROCESS #######
##############################
plot_results(csys, control, dynamic_p, losses)

pulses = csys.pulses(control, dynamic_p)
propagators = csys.trajectory(control, dynamic_p)
final_propagator = csys.final_state(control, dynamic_p)
loss_after_optimizer = csys.loss(control, dynamic_p)


from scipy.integrate import solve_ivp

args = (control, dynamic_p, static_p)
def ff(t, y_vec):
    y = vec_to_matrix(y_vec, mat_basis)
    return matrix_to_vec(vector_field(y, t, args), mat_basis)


y0 = np.asarray(matrix_to_vec(U0, mat_basis))
sol = solve_ivp(ff, (0.0, 1.0), y0, method='DOP853', max_step=1e-2)
Uf = vec_to_matrix(sol.y[:, -1], mat_basis)
infidelity(jnp.exp(-1j*control[0]*drift) @ Uf, U1)