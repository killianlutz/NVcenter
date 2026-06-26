import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree

from src._networks import normalize_if_not_zero
from src._quantum import matrix_to_vec, vec_to_matrix, dagger, trace_dot, matrix_to_coeff, infidelity
from scipy.integrate import solve_ivp
import numpy as np
import lineax as lx

@jax.tree_util.register_pytree_node_class
class ControlSystem:
    def __init__(self, static_parameters):
        self.static_p = static_parameters

    def final_state(self, control, dynamic_p):
        return self.trajectory(control, dynamic_p)[-1, ...]

    def trajectory(self, control, dynamic_p):
        p = self.static_p["integrator"]
        h = p["h"]
        ts = p["ts"]
        scheme = p["scheme"]
        f = p["vector_field"]

        x0 = self.static_p["system"]["initial_state"]

        # close over static parameters with e.g. Python functions
        def one_step(carry, _):
            x, i = carry
            y = scheme(f, x, ts[i], (control, dynamic_p, self.static_p))
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
        projector = self.static_p["projector"]
        return tuple(
            proj(p, d, learning_rate) for (p, d, proj) in zip(control, direction, projector)
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

    def pulses_shape(self, control, dynamic_p):
        H0 = dynamic_p["drift"]
        Hc = self.static_p["system"]["ctrl"]
        T, g_vec = control[0], control[1]
        g = vec_to_matrix(g_vec, self.static_p["su_basis"])
        Us = self.trajectory(control, dynamic_p)

        def feedback(U, g, H):
            g_conjugate_U = U @ g @ dagger(U)
            u = jax.vmap(
                lambda Hj, y: jnp.real(trace_dot(Hj, y)),
                in_axes=(0, None)
            )(H, g_conjugate_U)
            scaling = 1/jnp.linalg.norm(u)
            return u * scaling

        return jax.tree.map(
            lambda H: jax.vmap(feedback, in_axes=(0, None, None))(Us, g, H),
            Hc
        )

    def pulses_amplitude(self, control, dynamic_p):
        Ms = self.static_p["constraints"]["max_amplitude"]
        ts = self.static_p["integrator"]["ts"]
        params_to_fns = self.static_p["system"]["params_to_fns"]
        weights = control[2:]

        return jax.tree.map(
            lambda M, params_to_fn, weight: M*jax.vmap(params_to_fn, (0, None))(ts, weight).reshape(-1),
            Ms, params_to_fns, weights
        )

    def pulses(self, control, dynamic_p):
        pulses_amplitude = self.pulses_amplitude(control, dynamic_p)
        pulses_shape = self.pulses_shape(control, dynamic_p)
        fn = jax.vmap(lambda x, y: x*y, (None, 1), 1)
        return jax.tree.map(lambda a, s: fn(a, s),
            pulses_amplitude, pulses_shape
        )

    def validate(self, control, dynamic_p, **kwargs):
        T = control[0]
        U1 = dynamic_p["target"]
        U0 = self.static_p["system"]["initial_state"]
        H0 = dynamic_p["drift"]
        loss_fn = self.static_p["loss_fn"]
        mat_basis = self.static_p["mat_basis"]
        vector_field = self.static_p["integrator"]["vector_field"]

        args = (control, dynamic_p, self.static_p)
        def ode_velocity(t, y_vec):
            y = vec_to_matrix(y_vec, mat_basis)
            return matrix_to_vec(vector_field(y, t, args), mat_basis)

        tspan = (0.0, 1.0)
        y0 = np.asarray(matrix_to_vec(U0, mat_basis))
        ys = solve_ivp(ode_velocity, tspan, y0, **kwargs).y

        U_final = vec_to_matrix(ys[:, -1], mat_basis)
        return loss_fn(dagger(U1) @ U_final, args)

    def save_to_npz(self, filename, control, dynamic_p):
        target = dynamic_p
        gate_time = control[0]
        covector = control[1]
        weights = control[2:]
        pulses = self.pulses(control, dynamic_p)
        final_propagator = self.final_state(control, dynamic_p)
        loss_after_optimizer = self.loss(control, dynamic_p)
        time_points = self.static_p["integrator"]["ts"]

        jnp.savez(filename, target, gate_time, covector, *pulses, time_points, final_propagator, loss_after_optimizer, *weights)

    def tree_flatten(self):
        return (), self.static_p

    # reconstruct from children + aux_data
    @classmethod
    def tree_unflatten(cls, aux_data, _):
        return cls(aux_data)