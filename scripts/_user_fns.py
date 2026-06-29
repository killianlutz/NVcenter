import jax.numpy as jnp
import jax
import matplotlib.pyplot as plt

from src._classes import ControlSystem
from src._quantum import *
from src._networks import network, piecewise_cst_interp, normalize_if_not_zero, proj_ball

###############################
##### AUXILIARY FUNCTIONS #####
###############################
def runge_kutta(f, x, t, p):
    h = p[-1]["integrator"]["h"]
    k1 = f(x           , t + 0.0*h, p)
    k2 = f(x + 0.5*h*k1, t + 0.5*h, p)
    k3 = f(x + 0.5*h*k2, t + 0.5*h, p)
    k4 = f(x +     h*k3, t + 1.0*h, p)
    return x + h*(k1 + 2*k2 + 2*k3 + k4)/6


def loss_fn(x, p):
    # second argument is p = (control, dynamic_p, static_p)

    # # classical infidelity
    # identity = p[-1]["system"]["initial_state"]
    # return infidelity(x, identity)

    # allows local phases
    x_diag = jnp.diag(x)
    F = jnp.square(jnp.linalg.norm(x_diag))/jnp.size(x_diag)
    return jnp.abs(1 - F)

###############################
########## MAGICARP ###########
###############################
class Magicarp(ControlSystem):
    def __init__(self, static_parameters):
        super().__init__(static_parameters)

    # def vector_field(self, U, t, p):
    #     control, dynamic_p, static_p = p
    #     Ms = static_p["constraints"]["max_amplitude"]
    #     H0 = dynamic_p["drift"]
    #     Hc = static_p["system"]["ctrl"]
    #     su_basis = static_p["su_basis"]
    #
    #     T, g_vec, weights = control[0], control[1], control[2:]
    #     g = vec_to_matrix(g_vec, su_basis)
    #     g_conjugate_U = U @ g @ dagger(U)
    #
    #     def pulses_fn(H, M, weight):
    #         pulses = jax.vmap(
    #             lambda x, y: jnp.real(trace_dot(x, y)),
    #             in_axes=(0, None)
    #         )(H, g_conjugate_U)
    #         pulses_scaled = pulses / jnp.linalg.norm(pulses)
    #         control_hamiltonian = jnp.tensordot(pulses_scaled, H, axes=1)
    #         amplitude = self.params_to_pulses(t, weight)
    #         return M * amplitude * control_hamiltonian
    #
    #     control_hamiltonian = jnp.sum(
    #         jnp.stack(jax.tree.map(pulses_fn, Hc, Ms, weights)),
    #         axis=0
    #     )
    #
    #     return (-1j * T) * (H0 + control_hamiltonian) @ U
    def vector_field(self, U, t, p):
        control, dynamic_p, static_p = p
        T = control[0]
        return T*super().vector_field(U, t, p)


    def projector(self):
        # same pytree structure as the control, one projector per control variable
        # if T must be constant, just set the corresponding projector to (lambda T, dT, lr: T)
        return (
            lambda T, dT, lr: jnp.maximum(T + lr * dT, 0.0),
            lambda g, dg, lr: g + lr * dg,
            lambda w, dw, lr: jax.tree.map(lambda x, dx: x + lr * dx, w, dw),
            lambda w, dw, lr: jax.tree.map(lambda x, dx: x + lr * dx, w, dw)
        )

    def params_to_pulses(self, t, control, dynamic_p, U):
        Hc = self.static_p["system"]["ctrl"]
        Ms = self.static_p["constraints"]["max_amplitude"]
        g_vec, weights = control[1], control[2:]

        g = vec_to_matrix(g_vec, self.static_p["su_basis"])
        g_conjugate_U = U @ g @ dagger(U)

        def pulses_fn(H, M, weight):
            pulses = jax.vmap(
                lambda x, y: jnp.real(trace_dot(x, y)),
                in_axes=(0, None)
            )(H, g_conjugate_U)
            amplitude = network(t, weight)
            return M * amplitude * normalize_if_not_zero(pulses)

        return jax.tree.map(pulses_fn, Hc, Ms, weights)

    def save_to_npz(self, filename, control, dynamic_p):
        target = dynamic_p["target"]
        pulses = self.pulses(control, dynamic_p)
        final_propagator = self.final_state(control, dynamic_p)
        loss_after_optimizer = self.loss(control, dynamic_p)
        time_points = self.static_p["integrator"]["ts"]

        jnp.savez(filename, U1=target, T=control[0], u=pulses[0], v=pulses[1], ts=time_points, U_final=final_propagator, loss=loss_after_optimizer, g=control[1], u_weights=control[2], v_weights=control[3])

###############################
########## GRAPE ##############
###############################
class Grape(ControlSystem):
    def __init__(self, static_parameters):
        super().__init__(static_parameters)

    def vector_field(self, U, t, p):
        control, dynamic_p, static_p = p
        T = control[0]
        return T*super().vector_field(U, t, p)

    def projector(self):
        return (
        lambda T, dT, lr: jnp.maximum(T + lr*dT, 0.0),
        lambda u, du, lr: u + lr*du,
        lambda v, dv, lr: v + lr*dv
    )

    def params_to_pulses(self, t, control, dynamic_p, U):
        Ms = self.static_p["constraints"]["max_amplitude"]
        weights = control[-2:]

        def pulses_fn(M, weight):
            n_pieces = jnp.size(weight, 0)
            return M*proj_ball(piecewise_cst_interp(t, weight, n_pieces))

        return jax.tree.map(pulses_fn, Ms, weights)

    def save_to_npz(self, filename, control, dynamic_p):
        target = dynamic_p["target"]
        pulses = self.pulses(control, dynamic_p)
        final_propagator = self.final_state(control, dynamic_p)
        loss_after_optimizer = self.loss(control, dynamic_p)
        time_points = self.static_p["integrator"]["ts"]

        jnp.savez(filename, U1=target, T=control[0], u=pulses[0], v=pulses[1], ts=time_points, U_final=final_propagator, loss=loss_after_optimizer, u_weights=control[1], v_weights=control[2])
