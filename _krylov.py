import numpy as np
from _quantum import matrix_to_vec, vec_to_matrix

def krylov_subspace(mvp, dot, b, n, abstol=1e-15, reltol=1e-8):
    # krylov space of dimension <= n
    G = np.zeros((n, n)) # gram matrix
    K = np.zeros((n, *np.shape(b)), dtype=b.dtype) # basis

    nX = np.linalg.norm(b)
    K[0] = b if nX < abstol else b/nX
    G[0, 0] = dot(K[0], K[0])
    dim_krylov = 0 if nX < abstol else 1
    is_maximal = True if nX < abstol else False

    while dim_krylov < n and not is_maximal:
        l = dim_krylov
        # iterate linear operator on b
        X = mvp(K[l - 1])
        if np.linalg.norm(X) < abstol:
            break

        K[l] = X/np.linalg.norm(X)
        G[l, l] = dot(K[l], K[l])
        # fill new col/row of gram matrix
        for i in range(l):
            g = dot(K[i], K[l])
            G[i, l] = g
            G[l, i] = g

        # if projection equals the vector, then linearly dependent
        x = G[:l, l] # vector
        p = np.linalg.solve(G[:l, :l], x) # projection

        iterable = np.fromiter((u*v for (u, v) in zip(p, K[:l])), dtype=np.matrix)
        P = np.sum(iterable)
        is_maximal = np.linalg.norm(K[l] - P) < reltol * np.linalg.norm(K[l])
        if not is_maximal:
            dim_krylov += 1

    K = K[:dim_krylov]
    G = G[:dim_krylov, :dim_krylov]
    return K, G, dim_krylov


def block_krylov(ad, dot, bs, basis, *kwargs):
    n = len(basis)
    # compute the different krylov subspaces associated to the bs
    dim_block = 0
    K_block = []
    for b in bs:
        K, _, dim = krylov_subspace(ad, dot, b, n, *kwargs)
        dim_block += dim
        K_block.append(K)
    K_block = np.concatenate(K_block)

    # extract an orthonormal basis
    K_vec = np.asarray([matrix_to_vec(x, basis) for x in K_block]).T
    Q, _ = np.linalg.qr(K_vec)
    K = np.asarray([vec_to_matrix(x, basis) for x in Q.T])

    return K, dim_block

# ##### INTERACTION PICTURE
# ad = lambda x: 1j*(drift @ x - x @ drift)
# dot = lambda x, y: np.real(trace_dot(x, y))
# K = block_krylov(ad, dot, ctrl, su_basis)[0]
#
# ctrl_cst = jnp.asarray(K)