############################################################
################## CONTROL OF NV CENTERS ###################
############################################################
from scripts._config import *
csys = ControlSystem(static_p)
solve_fn = jax.jit(csys.solve_ocp)

def initial_guess(method):
    if method =="MAGICARP":
        K = jax.random.split(keys[0], 3)
        T = 0.1 * 2 * jnp.pi * jnp.ones(1) # time horizon
        g = jax.random.normal(K[0], su_dim) # initial covector
        theta_u = rand_weights(K[1], jnp.array([1, 5, 5, 1])) # amplitude u
        theta_v = rand_weights(K[2], jnp.array([1, 5, 5, 1])) # amplitude v
        return T, g, theta_u, theta_v
    elif method == "GRAPE":
        n_pieces = 10*(d + 1)
        T = 0.1 * 2 * jnp.pi * jnp.ones(1)  # time horizon
        u = 1e-2 * jnp.ones((n_pieces, jnp.size(ctrl[0], 0))) # u
        v = 1e-2 * jnp.ones((n_pieces, jnp.size(ctrl[1], 0))) # v
        return T, u, v
    else:
        raise ValueError("only available methods: MAGICARP, GRAPE")

##############################
######### SOLVE ##############
##############################
init_control = initial_guess(method)
dynamic_p = {"target": U1, "drift": drift}
control, losses, n_iter = solve_fn(init_control, dynamic_p)
losses = losses[:n_iter + 1] # not supported inside solve_fn (jit-compilation + n_iter dynamic)
print(f"\n Iter: {n_iter} \n Loss: {losses[-1]:.1e} \n Gate time (scaled): {control[0][0]/(2*jnp.pi):.3f}")

##############################
######### POST PROCESS #######
##############################
plot_results(csys, control, dynamic_p, 1e-15 + losses, figsize=(16, 6))

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

#
# def affine_interp(t, ys, dt):
#     i = jnp.floor(t/dt).astype(jnp.int16)
#     return ys[i] + (ys[i+1] - ys[i])*(t - i*dt)/dt
#
# # pulses_opt = csys.pulses(control, dynamic_p)
# pulses_opt = control[-2:]
# model = lambda t, ys: affine_interp(t, ys, csys.static_p["integrator"]["h"])
#
# from scipy.integrate import solve_ivp
# def check(H0, Hc, T, weights, **kwargs):
#     U1 = dynamic_p["target"]
#     U0 = csys.static_p["system"]["initial_state"]
#     loss_fn = csys.static_p["loss_fn"]
#     mat_basis = csys.static_p["mat_basis"]
#
#     def vector_field(U, t, _):
#         def pulses_fn(H, u):
#             pulses_t = model(t, u)
#             control_hamiltonian = jnp.tensordot(pulses_t, H, axes=1)
#             return control_hamiltonian
#
#         control_hamiltonian = jnp.sum(
#             jnp.stack(jax.tree.map(pulses_fn, Hc, weights)),
#             axis=0
#         )
#
#         return (-1j * T) * (H0 + control_hamiltonian) @ U
#
#     args = (control, dynamic_p, csys.static_p)
#
#     def ode_velocity(t, y_vec):
#         y = vec_to_matrix(y_vec, mat_basis)
#         return matrix_to_vec(vector_field(y, t, args), mat_basis)
#
#     tspan = (0.0, 1.0)
#     y0 = np.asarray(matrix_to_vec(U0, mat_basis))
#     ys = solve_ivp(ode_velocity, tspan, y0, **kwargs).y
#
#     U_final = vec_to_matrix(ys[:, -1], mat_basis)
#     return loss_fn(dagger(U1) @ U_final, args)
#
# def robustness(x, **kwargs):
#     return check((1 + x)*drift, ctrl, control[0], pulses_opt, method='DOP853', atol=1e-9, rtol=1e-6) #- losses[-1]
#
# dIF = jax.grad(robustness)
# d2IF = jax.grad(dIF)
# fo = dIF(0.0)
# so = d2IF(0.0)
# print(f"fo: {fo} \n so: {so}")
#
#
# perturb = jnp.linspace(-1e-2, 1e-2, 10)
# infidelities = jax.vmap(robustness)(perturb)
# infidelities = [robustness(x) for x in perturb]
# plt.scatter(perturb, infidelities)
# jnp.savez("robustness_MAGICARP.npz", drift_perturb=perturb, IF=infidelities, fo=fo, so=so)
#
#
# M = jnp.load("robustness_MAGICARP.npz")
# G = jnp.load("robustness_GRAPE.npz")
# perturb = M["drift_perturb"]
# IFM = M["IF"]
# IFG = G["IF"]
# plt.scatter(perturb, IFM, color='r', label=f"M : $\\partial^2 IF/\\partial^2 H_0 = {M["so"]:.1e}$")
# plt.scatter(perturb, IFG, color='b', label=f"G : $\\partial^2 IF/\\partial^2 H_0 = {G["so"]:.1e}$")
# plt.plot(perturb, M["fo"]*perturb + 0.5*M["so"]*perturb**2, color='k')
# plt.plot(perturb, G["fo"]*perturb + 0.5*G["so"]*perturb**2, color='k', linestyle='--')
# plt.legend()
# plt.grid()
