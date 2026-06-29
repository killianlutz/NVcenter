import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree

import matplotlib.pyplot as plt
from src._networks import normalize_if_not_zero, piecewise_cst_interp, proj_ball, network
from src._quantum import matrix_to_vec, vec_to_matrix, dagger, trace_dot, vector_field_schrodinger
from diffrax import diffeqsolve, ODETerm, Dopri8, PIDController
import lineax as lx


@jax.tree_util.register_pytree_node_class
class ControlSystem:
    def __init__(self, static_parameters):
        self.static_p = static_parameters

    def final_state(self, control, dynamic_p):
        return self.trajectory(control, dynamic_p)[-1, ...]

    def trajectory(self, control, dynamic_p):
        p = self.static_p["integrator"]
        ts = p["ts"]
        scheme = p["scheme"]
        x0 = self.static_p["system"]["initial_state"]

        # close over static parameters with e.g. Python functions
        def one_step(carry, _):
            x, i = carry
            y = scheme(self.vector_field, x, ts[i], (control, dynamic_p, self.static_p))
            return (y, i + 1), y

        carry = (x0, 0) # only dynamic variables (jax traceable) inside carry
        xs = jax.lax.scan(one_step, carry, None, length=len(ts) - 1)[1]

        return jnp.concatenate((x0[None, ...], xs), axis=0) # view of x0

    def loss(self, control, dynamic_p):
        loss_fn = self.static_p["loss_fn"]
        T, U1, H0 = control[0], dynamic_p["target"], dynamic_p["drift"]
        U_final = self.final_state(control, dynamic_p)

        return loss_fn(dagger(U1) @ U_final, (control, dynamic_p, self.static_p))

    def natural_gradient(self, control, dynamic_p):
        lstq_p = self.static_p["optimizer"]["least_squares"]
        flat_control, unravel = ravel_pytree(control)

        def model(u_flat, p):
            control = unravel(u_flat)
            T, U1, H0 = control[0], p["target"], p["drift"]
            U_vec = matrix_to_vec(
                dagger(U1) @ self.final_state(control, p),
                self.static_p["mat_basis"]
            )
            return U_vec, U_vec

        def cost(U_vec, u_flat, p):
            loss_fn = self.static_p["loss_fn"]
            return loss_fn(
                vec_to_matrix(U_vec, self.static_p["mat_basis"]),
                (unravel(u_flat), p, self.static_p)
            )

        eps = lstq_p["regularization"]
        if lstq_p["is_iterative"]:
            partial_fn = lambda x: model(x, dynamic_p)[0]
            U_vec, f_jvp = jax.linearize(partial_fn, flat_control)
            _, f_vjp = jax.vjp(partial_fn, flat_control)
            current_loss, grad_cost = jax.value_and_grad(cost)(U_vec, flat_control, dynamic_p)
            def gram_mvp(x):
                # Tikhonov least-squares operator (normal equations)
                # Gram matrix calculated by combined vjp - jvp.
                return f_vjp(f_jvp(x))[0] + eps*x

            A = lx.FunctionLinearOperator(gram_mvp, flat_control, tags=lstq_p["tags"])
            b = f_vjp(-grad_cost)[0]
            solver = lstq_p["iterative_solver"]
        else:
            linearized_model, U_vec = jax.jacobian(model, has_aux=True)(flat_control, dynamic_p)
            current_loss, grad_cost = jax.value_and_grad(cost)(U_vec, flat_control, dynamic_p)

            A = lx.MatrixLinearOperator(
                linearized_model.T @ linearized_model + eps*jnp.eye(jnp.size(flat_control)),
                tags=lstq_p["tags"]
            )
            b = -(linearized_model.T @ grad_cost)
            solver = lstq_p["direct_solver"]

        flat_step = lx.linear_solve(A, b, solver).value

        normalize_gradient = self.static_p["optimizer"]["normalize_gradient"]
        _flat_step = jax.lax.cond(
            normalize_gradient,
            normalize_if_not_zero,
            lambda y: y,
            flat_step
        )

        return unravel(_flat_step), current_loss

    def apply_update(self, control, direction, learning_rate):
        return tuple(
            proj(p, d, learning_rate) for (p, d, proj) in zip(control, direction, self.projector())
        )

    def line_search(self, control, dynamic_p, direction, reference_loss):
        p = (control, dynamic_p, direction)
        search_parameters = self.static_p["optimizer"]["line_search"]
        search_fn = search_parameters["search_fn"]

        def loss_along_line(e, p):
            lr = jnp.pow(10.0, e)
            control, dynamic_p, direction = p
            new_control = self.apply_update(control, direction, lr)
            return self.loss(new_control, dynamic_p)

        e, next_loss = search_fn(loss_along_line, p, (search_parameters, reference_loss))
        return jnp.pow(10.0, e), next_loss

    def optimizer_step(self, control, dynamic_p):
        direction, current_loss = self.natural_gradient(control, dynamic_p)
        learning_rate, next_loss = self.line_search(control, dynamic_p, direction, current_loss)
        control = self.apply_update(control, direction, learning_rate)

        return control, next_loss

    def solve_ocp(self, init_control, dynamic_p):
        optimizer_p = self.static_p["optimizer"]
        abstol_loss = optimizer_p["abstol_loss"]
        reltol_dist = optimizer_p["reltol_dist"]
        n_max = optimizer_p["n_max"]

        # initial guess
        init_loss = self.loss(init_control, dynamic_p)
        losses = jnp.zeros(n_max).at[0].set(init_loss)
        # two steps recurrence
        control, current_loss = self.optimizer_step(init_control, dynamic_p)
        losses = losses.at[1].set(current_loss)
        init = (control, init_control, losses, 1, dynamic_p)

        def cond(val):
            current_u, old_u, losses, i, _ = val
            current_loss = losses[i]
            x, _ = ravel_pytree(current_u)
            y, _ = ravel_pytree(old_u)

            # relative distance between consecutive iterates
            # iteration counter
            # abs tolerance on objective value
            return jnp.logical_and(
                i < n_max - 1,
                jnp.logical_and(
                    jnp.linalg.norm(x - y) > reltol_dist*jnp.linalg.norm(y),
                    current_loss > abstol_loss
                )
            )

        def body(val):
            current_u, _, losses, i, dyn_p = val
            new_u, new_loss = self.optimizer_step(current_u, dyn_p)
            return (
                new_u,
                current_u,
                losses.at[i + 1].set(new_loss),
                i + 1,
                dyn_p
            )

        val = jax.lax.while_loop(cond, body, init)
        optimized_control = val[0]
        losses, n_iter = val[-3:-1]
        return optimized_control, losses, n_iter

    def validate(self, control, dynamic_p, dt0=1e-1, solver=Dopri8(), stepsize_controller=PIDController(atol=1e-7, rtol=1e-5), **kwargs):
        U0 = self.static_p["system"]["initial_state"]
        args = (control, dynamic_p, self.static_p)

        U_final = diffeqsolve(
            ODETerm(lambda t, U, args: self.vector_field(U, t, args)),
            t0=0.0, t1=1.0, dt0=dt0,
            y0=U0,
            args=args,
            solver=solver,
            stepsize_controller=stepsize_controller,
            **kwargs
        ).ys[0, ...]

        loss_fn = self.static_p["loss_fn"]
        U1 = dynamic_p["target"]
        return loss_fn(dagger(U1) @ U_final, args)

    def save_to_npz(self, filename, control, dynamic_p):
        target = dynamic_p
        gate_time = control[0]
        pulses = self.pulses(control, dynamic_p)
        final_propagator = self.final_state(control, dynamic_p)
        loss_after_optimizer = self.loss(control, dynamic_p)
        time_points = self.static_p["integrator"]["ts"]

        jnp.savez(filename, target, gate_time, *pulses, time_points, final_propagator, loss_after_optimizer)

    def pulse_fns(self, control, dynamic_p):
        control_pulses = self.pulses(control, dynamic_p)
        def zoh(weights):
            n_pieces = jnp.size(weights, 0)
            return lambda t: piecewise_cst_interp(t, weights, n_pieces)

        return jax.tree.map(zoh, control_pulses)

    def validate_concrete_pulses(self, time_pulse_fns_pair, dynamic_p, dt0=1e-1, solver=Dopri8(), stepsize_controller=PIDController(atol=1e-7, rtol=1e-5), **kwargs):
        U0 = self.static_p["system"]["initial_state"]
        Hc = self.static_p["system"]["ctrl"]
        H0 = dynamic_p["drift"]

        U_final = diffeqsolve(
            ODETerm(vector_field_schrodinger),
            t0=0.0, t1=1.0, dt0=dt0,
            y0=U0,
            args=(H0, Hc, *time_pulse_fns_pair),
            solver=solver,
            stepsize_controller=stepsize_controller,
            **kwargs
        ).ys[0, ...]

        loss_fn = self.static_p["loss_fn"]
        U1 = dynamic_p["target"]
        return loss_fn(dagger(U1) @ U_final, None)

    def plot_results(self, control, dynamic_p, losses, **kwargs):
        T = control[0]
        Ms = self.static_p["constraints"]["max_amplitude"]
        ts = self.static_p["integrator"]["ts"]
        pulses = self.pulses(control, dynamic_p)  # g -> pulse
        fig, axes = plt.subplots(1, 1 + len(pulses), **kwargs)

        # LOSS
        axes[0].semilogy(losses, '-x', color="k")
        axes[0].set_title(f"Gate time: $T =${T[0] / (2 * jnp.pi):.3f} ($\\times 2\\pi/\\omega_c$)")
        axes[0].set_xlabel("Iteration")
        axes[0].set_ylabel("Loss")
        axes[0].grid(True)

        # PULSES
        for (i, (pulse, M)) in enumerate(zip(pulses, Ms)):
            energy = jnp.linalg.norm(pulse, axis=1)
            ax = axes[i + 1]
            ax.plot(ts, pulse, linewidth=3)
            ax.plot(ts, energy, linestyle='--', color='purple', label="$norm$", linewidth=3)
            ax.plot(ts, M * jnp.ones_like(ts), color='k', linewidth=0.75)
            ax.plot(ts, -M * jnp.ones_like(ts), color='k', linewidth=0.75)
            ax.plot(ts, 0 * ts, color='k', linewidth=0.75)
            ax.set_xlabel(r"Time $t/T$")
            ax.grid(True)
            ax.legend(loc='upper right')

        plt.tight_layout()
        fig.show()

    def vector_field(self, U, t, p):
        control, dynamic_p, _ = p
        H0 = dynamic_p["drift"]
        Hc = self.static_p["system"]["ctrl"]

        pulses = self.params_to_pulses(t, control, dynamic_p, U)
        control_hamiltonian = jnp.sum(
            jnp.stack(
                jax.tree.map(
                    lambda x, y: jnp.tensordot(x, y, axes=1), pulses, Hc
                )
            ),
            axis=0
        )
        return -1j * (H0 + control_hamiltonian) @ U

    def projector(self):
        pass

    def params_to_pulses(self, t, control, dynamic_p, state):
        pass

    def pulses(self, control, dynamic_p):
        ts = self.static_p["integrator"]["ts"]
        Us = self.trajectory(control, dynamic_p)
        control_pulses = jax.vmap(
                lambda t, U: jnp.array(self.params_to_pulses(t, control, dynamic_p, U)),
                in_axes=(0, 0),
                out_axes=1
        )(ts, Us)

        return tuple(control_pulses)

    def tree_flatten(self):
        return (), self.static_p

    # reconstruct from children + aux_data
    @classmethod
    def tree_unflatten(cls, aux_data, _):
        return cls(aux_data)
