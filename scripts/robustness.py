from src._robustness import *

systems = [Grape, Magicarp]
method_names = ["grape", "magicarp"]
filenames = ["./sims/"+method+".npz" for method in method_names]

##############################
######## OPTIMIZE ############
##############################
def initial_guess(control_system):
    if isinstance(control_system, Magicarp):
        K = jax.random.split(keys[10], 3)
        T = 0.1 * 2 * jnp.pi * jnp.ones(1) # time horizon
        g = jax.random.normal(K[0], su_dim) # initial covector
        theta_u = rand_weights(K[1], jnp.array([1, 5, 5, 1])) # amplitude u
        theta_v = rand_weights(K[2], jnp.array([1, 5, 5, 1])) # amplitude v
        return T, g, theta_u, theta_v

    elif isinstance(control_system, Grape):
        n_pieces = 10*(d + 1)
        T = 0.1 * 2 * jnp.pi * jnp.ones(1)  # time horizon
        u = 1e-3 * jnp.ones((n_pieces, jnp.size(ctrl[0], 0))) # u
        v = 1e-3 * jnp.ones((n_pieces, jnp.size(ctrl[1], 0))) # v
        return T, u, v

    else:
        raise ValueError("only available systems: Magicarp, Grape")

for (filename, control_system, method) in zip(filenames, systems, method_names):
    csys = control_system(static_p)
    solve_fn = jax.jit(csys.solve_ocp)

    # optimize
    init_control = initial_guess(csys)
    control, losses, n_iter = solve_fn(init_control, dynamic_p)
    losses = losses[:n_iter + 1]  # not supported inside solve_fn (jit-compilation + n_iter dynamic)
    print("\n Optimization results ("+method+f"): \n Iter: {n_iter} \n Loss: {losses[-1]:.1e} \n Gate time (scaled): {control[0][0] / (2 * jnp.pi):.3f}")

    # save concrete pulses
    filename = "./sims/magicarp.npz" if isinstance(csys, Magicarp) else "./sims/grape.npz"
    csys.save_to_npz(filename, control, dynamic_p)


##############################
#### EVALUATE ROBUSTNESS #####
##############################
colors = ["b", "r"]
labels = ["GRAPE", "MAGICARP"]
markers = ["v", "o"]

dH0 = jnp.linspace(-1e-2, 1e-2, 100)
dIF = []
quadratic_coeffs = []
control_pulses = []

for filename in filenames:
    data = jnp.load(filename)
    ts = data["ts"]
    pulses = (data["u"], data["v"])
    Tuv = (data["T"], pulses)
    # fix the control and compute the infidelity when the data varies around the nominal value 0, wlog.
    variations = vmap_robustness(
        Tuv,
        dt0=1e-2,
        stepsize_controller=PIDController(atol=1e-7, rtol=1e-6)
    )(dH0)
    # fit the variations around the nominal value to a degree 2 polynomial
    quadratic_coeffs.append(jnp.polyfit(dH0, variations, deg=2)[:2])
    dIF.append(variations)
    control_pulses.append(pulses)

# visualize robustness
for (variations, coeffs, color, label, marker, method) in zip(dIF, quadratic_coeffs, colors, labels, markers, method_names):
    plt.scatter(dH0, variations, c=color, label=label, marker=marker, s=10)
    so, fo = coeffs
    quadratic_fit = fo * dH0 + so * dH0 ** 2
    plt.plot(dH0, quadratic_fit, c=color, label=f"Curvature = {0.5*so:.1e}")
    print("\n Quadratic fit ("+method+f"): \n 1st order: {fo:.3e} \n 2nd order: {0.5*so:.3e}")

plt.legend()
plt.xlabel("drift variation $\\delta H_0$")
plt.ylabel("infidelity variation $\\delta $IF")
plt.title("robustness")
plt.grid()
plt.savefig("./sims/robustness.pdf")

# visualize electronic controls
plt.close()
fig, axs = plt.subplots(1, 2, figsize=(16, 8))
for (pulses, color, label, marker) in zip(control_pulses, colors, labels, markers):
    axs[0].plot(ts, pulses[0][:, 0], c=color, label=label+str(1))
    axs[0].plot(ts, pulses[0][:, 1], c=color, label=label+str(2), linestyle='--')
    axs[1].plot(ts, pulses[1][:, 0], c=color, label=label+str(1))
    axs[1].plot(ts, pulses[1][:, 1], c=color, label=label+str(2), linestyle='--')

axs[0].legend()
axs[0].set_xlabel("$t$")
axs[0].set_ylabel("$u_j(t)$")
axs[0].set_title("electronic control pulses")
axs[0].grid()

axs[1].legend()
axs[1].set_xlabel("$t$")
axs[1].set_ylabel("$v_j(t)$")
axs[1].set_title("nuclear control pulses")
axs[1].grid()

fig.savefig("./sims/robustness_controls.pdf")