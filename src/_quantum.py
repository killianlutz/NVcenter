import jax.numpy as jnp
import jax
import numpy as np
import scipy.sparse as sparse

def pauli_matrices():
    sigma_x = jnp.array([[0, 1], [1, 0]], dtype=jnp.complex64)
    sigma_y = jnp.array([[0, -1j], [1j, 0]], dtype=jnp.complex64)
    sigma_z = jnp.array([[1, 0], [0, -1]], dtype=jnp.complex64)
    return {"x": sigma_x, "y": sigma_y, "z": sigma_z}

def spin_matrices():
    pauli = pauli_matrices()
    keys = pauli.keys()
    Id = jnp.eye(2)
    S = {key: jnp.kron(0.5*sigma, Id) for key, sigma in zip(keys, pauli.values())}
    I = {key: jnp.kron(Id, 0.5*sigma) for key, sigma in zip(keys, pauli.values())}
    SI = {key1+key2: jnp.kron(0.5*pauli[key1], 0.5*pauli[key2]) for key1 in keys for key2 in keys}

    return S, I, SI

def basis(dim):
    B = []
    for i in range(dim):
        for j in range(dim):
            b = np.zeros((dim, dim), dtype=np.complex64)
            b[i, j] = 1
            B.append(b)
            B.append(1j*b)

    return jnp.array(B)

def subasis(dim):
    # orthonormal
    basis = []

    for k in np.arange(dim - 1):
        Dk = np.zeros((dim, dim), dtype=jnp.complex64)
        for i in np.arange(k + 1):
            Dk[i, i] = 1
        Dk[k + 1, k + 1] = -(k + 1)
        Dk /= np.linalg.norm(Dk)
        basis.append(Dk)

    k = 0
    for i in np.arange(dim):
        for j in np.arange(i):
            Sk = np.zeros((dim, dim), dtype=jnp.complex64)
            Ak = np.zeros((dim, dim), dtype=jnp.complex64)

            Sk[i, j] = 1
            Sk[j, i] = 1
            Sk /= np.linalg.norm(Sk)

            Ak[i, j] = 1j
            Ak[j, i] = -1j
            Ak /= np.linalg.norm(Ak)

            basis.append(Sk)
            basis.append(Ak)

        k += 1

    return jnp.array(basis)

def ctrlbasis(dim, graph, sigma_z=False):
    # orthonormal
    H = []

    data_x = np.array([1, 1], dtype=np.complex64) / np.sqrt(2)
    data_y = np.array([-1j, 1j], dtype=np.complex64) / np.sqrt(2)
    data_z = np.array([1, -1], dtype=np.complex64) / np.sqrt(2)

    for idx_pair in graph:
        i, j = idx_pair

        for datum in (data_x, data_y):
            rows = np.array([i, j])
            cols = np.array([j, i])
            h = sparse.coo_array((datum, (rows, cols)), shape=(dim, dim)).toarray()
            H.append(h)

        if sigma_z:
            rows = np.array([i, j])
            cols = np.array([i, j])
            h = sparse.coo_array((data_z, (rows, cols)), shape=(dim, dim)).toarray()
            H.append(h)

    return jnp.array(H)

def matrix_to_coeff(g, basis_vector):
    return jnp.real(trace_dot(g, basis_vector))

def matrix_to_vec(g, basis):
    return jax.vmap(matrix_to_coeff, (None, 0))(g, basis)

def vec_to_matrix(v, basis):
    z = jax.vmap(lambda x, y: x*y, 0, 0)(v, basis)
    return jnp.sum(z, axis=0)

def sampleSU(dim, key):
    A = jax.random.normal(key, (dim, dim), dtype=jnp.complex64)
    Q, R = np.linalg.qr(A)

    r = jnp.diag(R)
    q = Q @ jnp.diag(r/jnp.abs(r))
    return jnp.array(toSU(q))

def toSU(q):
    dim = jnp.size(q, 0)
    phase = jnp.angle(jnp.linalg.det(q))
    return q * jnp.exp(-1j * phase / dim)

def dagger(a):
    return a.T.conj()

def trace_dot(a, b):
    return jnp.einsum('ij, ij -> ', a, b.conj())

def infidelity(x, y):
    dim = jnp.size(x, 1)
    fidelity = jnp.abs(trace_dot(x, y)) / dim
    return jnp.abs(1 - fidelity)

def CNOT():
    return toSU(
        jnp.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0]
        ], dtype=jnp.complex64)
    )
