import jax.numpy as jnp
import jax

def rand_weights(key, neurons):
    K = jax.random.split(key, (2, len(neurons) - 1))
    matrices = lambda k, i, j: jax.random.normal(K[0, k], (i, j))
    biases = lambda k, i: jax.random.normal(K[1, k], (i, 1))
    return {
        f"layer_{k}":
            {"w": matrices(k, neurons[k], neurons[k - 1]),
             "b": biases(k, neurons[k]) if k + 1 < len(neurons) else jnp.zeros(neurons[-1])}
        for k in range(1, len(neurons))
    }

def network(t, weights):
    x = jnp.reshape(jnp.array(t), (-1, 1))
    for params in weights.values():
        matrix, bias = params["w"], params["b"]
        x = jnp.tanh(matrix @ x + bias)
    return (x[0] + 1)*0.5