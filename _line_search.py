import jax
import jax.numpy as jnp

def golden_section(f, p, search_parameters):
    # signature of f: (s, p) -> float
    params, _ = search_parameters
    a, b = params["log_interval"]
    abstol = params["abstol"]
    max_iter = params["n_max"]

    g = 0.5*(1 + jnp.sqrt(5))  # golden ratio
    c = b - (g - 1)*(b - a)
    d = a + (g - 1)*(b - a)
    left_value = f(c, p)
    right_value = f(d, p)

    def cond_fn(state):
        i, a, b = state[:3]
        return jnp.logical_and(i <= max_iter, abstol < b - a)

    def body_fn(state):
        def true_fn(state):
            i, a, b, c, d, left_value, _ = state

            b = d
            d = c
            c = b - (g - 1) * (b - a)
            right_value = left_value
            left_value = f(c, p)

            return i + 1, a, b, c, d, left_value, right_value

        def false_fn(state):
            i, a, b, c, d, _, right_value = state

            a = c
            c = d
            d = a + (g - 1) * (b - a)
            left_value = right_value
            right_value = f(d, p)

            return i + 1, a, b, c, d, left_value, right_value

        left_value = state[-2]
        right_value = state[-1]
        return jax.lax.cond(left_value <= right_value, true_fn, false_fn, state)

    init_state = (1, a, b, c, d, left_value, right_value)
    interval = jax.lax.while_loop(cond_fn, body_fn, init_state)[1:3]
    m = 0.5*(interval[0] + interval[1])
    return m, f(m, p)