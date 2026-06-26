import jax.numpy as jnp
import jax
import numpy as np
import scipy.sparse as sparse
from functools import reduce

def pauli_matrices():
    sigma_x = jnp.array([[0, 1], [1, 0]], dtype=jnp.complex64)
    sigma_y = jnp.array([[0, -1j], [1j, 0]], dtype=jnp.complex64)
    sigma_z = jnp.array([[1, 0], [0, -1]], dtype=jnp.complex64)
    return {"x": sigma_x, "y": sigma_y, "z": sigma_z}

def qubit_spin_operators():
    return jax.tree.map(lambda x: 0.5 * x, pauli_matrices())

def nvc_op(op, n_nuclei):
    # nv center operatore: composite system with electron first (idx 0), nuclei then (1, 2, etc)
    qubit_spin_ops = qubit_spin_operators()
    alpha, j = list(op)

    return reduce(
        jnp.kron,
        [qubit_spin_ops[alpha] if str(i) == j
         else jnp.eye(2, dtype=jnp.complex64)
         for i in range(n_nuclei + 1)]
    )

def nvcenter_model(n_nuclei, A_parallels):
    # control operators assuming drift is linear combination of Sz and sum Izi
    if not jnp.size(A_parallels) == n_nuclei:
        raise ValueError("the size of the vector of couplings must match n_nuclei")

    Sx = nvc_op("x0", n_nuclei)
    Sy = nvc_op("y0", n_nuclei)
    Ixi = [nvc_op("x" + str(1 + i), n_nuclei) for i in range(n_nuclei)]
    Iyi = [nvc_op("y" + str(1 + i), n_nuclei) for i in range(n_nuclei)]
    SzIzi = [A_par * nvc_op("z0", n_nuclei) @ nvc_op("z" + str(1 + i), n_nuclei) for (i, A_par) in enumerate(A_parallels)]

    drift = jnp.sum(jnp.stack(SzIzi), axis=0)
    electronic_ctrl = jnp.stack((Sx, Sy))
    nuclear_ctrl = jnp.stack((
        jnp.sum(jnp.stack(Ixi), axis=0),
        jnp.sum(jnp.stack(Iyi), axis=0)
    ))

    return drift, electronic_ctrl, nuclear_ctrl

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

def CNOT(standard):
    if standard:
        gate = toSU(
            jnp.array([
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 0, 1],
                [0, 0, 1, 0]
            ], dtype=jnp.complex64)
        )
    else:
        gate = jnp.array([
            [1, 0, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0],
            [0, 1, 0, 0]
        ], dtype=jnp.complex64)
    return gate

def conditional_gate(d, basis_vector_pairs):
    G = jnp.eye(d, dtype=jnp.complex64)
    for (set_to_zero, set_to_one) in zip(basis_vector_pairs, basis_vector_pairs):
        for i in set_to_zero:
            G = G.at[i, i].set(0)
            for j in set_to_one:
                if not(j == i):
                    G = G.at[i, j].set(1)
    return G

def toffoli():
    return electron_flip_conditional_nuclear([1, 2], 2)

def electron_flip_conditional_nuclear(indices, n_nuclei):
    # find where the i-th nuclear spin has state = 1 (i >= 1)
    # basis ordering : |e n1 ... np > where electron and nuclei are qubits
    indices = list(indices)
    if list(filter(lambda x: (x < 1) or (x > n_nuclei), indices)):
        raise ValueError("nuclear indices satisfy 1 <= i <= n_nuclei")

    m = n_nuclei + 1
    d = 2**m
    labels = jnp.arange(0, 2**m)
    state_is_active = []

    for k in labels:
        # binary digits representation of k with m digits
        bits = list(map(int, bin(k)[2:]))
        while len(bits) < m:
            bits.insert(0, 0)

        # check if the conditional nuclear spins are active
        is_valid_state = True
        for i in indices:
            if not(bits[i] == 1):
                is_valid_state = False
                break

        if is_valid_state:
            state_is_active.append((k, bits))

    # given two composite states labeled k1 and k2
    # if their nuclear states are the same but their electronic state differs
    # then we associate a pair (k1, k2) to flip the coefficients at these indices.
    l = len(state_is_active)
    basis_vector_pairs = []
    for q in range(l):
        for p in range(q + 1, l):
            k1, bits1 = state_is_active[q]
            k2, bits2 = state_is_active[p]
            # take out the electronic state and compare binary decompositions
            if bits1[1:] == bits2[1:]:
                basis_vector_pairs.append([k1, k2])

    return conditional_gate(d, basis_vector_pairs)

def electron_flip(n_nuclei):
    m = 2**n_nuclei # half of Hilbert space dimension
    return jnp.block(
        [
            [jnp.zeros((m, m), dtype=jnp.complex64), jnp.eye(m)],
            [jnp.eye(m), jnp.zeros((m, m))]
        ]
    )