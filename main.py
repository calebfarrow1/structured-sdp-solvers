from scipy.sparse.linalg import eigsh
from scipy.sparse import diags
import numpy as np

np.set_printoptions(precision=3, suppress=True)

def M_shape(M_data):
    """
        M_data: data
            Some encoding of the matrix M.
    """
    return (M_data.shape[0], M_data.shape[1])

def M_mv(M_data, v):
    """
        M_data: data
            Some encoding of the matrix M.
        v: vector
            Vector to multiply with the matrix M.
    """
    return M_data @ v

def approx_min_evec(M_data, M_shape, M_mv, q, eps=1e-10):
    """
        M_data: data
            Some encoding of the matrix M.
        M_shape: function
            Function that returns the shape of the matrix M.
        M_mv: function
            Function that computes the matrix-vector product.
        q: int
            Number of iterations.
        eps: float
            Tolerance for convergence.
    """
    n = M_shape(M_data)[0]
    v = np.random.normal(size=n)
    v_old = np.zeros(n)
    v = v / np.linalg.norm(v)
    v0 = v
    rho = np.zeros(min(q, n-1)+1)
    omega = np.zeros(min(q, n-1)+1)

    for i in range(min(q, n-1)):
        omega[i+1] = np.real( np.dot( v, M_mv( M_data, v ) ) )
        v, v_old = M_mv( M_data, v ) - omega[i+1]*v - rho[i]*v_old, v
        rho[i+1] = np.linalg.norm(v) # This is different than what is shown in the paper, we believe the paper has a typo
        if rho[i+1] < eps:
            break
        v = v / rho[i+1]

    T = diags([rho[1:i+1], omega[1:i+2], rho[1:i+1]], offsets=[-1, 0, 1]).toarray()
    # print(rho)
    # print(omega)
    # print(T)
    _, u = eigsh(T, k=1, which='SM', return_eigenvectors=True)
    u = u[:, 0]
    
    v = v0
    v_sum = np.zeros(n)
    v_old = np.zeros(n)

    for i in range(min(q, n-1)):
        v_sum += u[i]*v
        v, v_old = M_mv( M_data, v ) - omega[i+1]*v - rho[i]*v_old, v
        if rho[i+1] < eps:
            break
        v = v / rho[i+1]

    return v_sum

def nystrom_sketch_init(n, R):
    """
        n: int
            Dimension of input matrix.
        R: int
            Size of sketch.
    """
    Omega = np.random.normal(size=(n, R))
    S = np.zeros((n, R))
    return Omega, S

def nystrom_sketch_rank_one_update(v, eta, S, Omega):
    """
        v: vector
            Update vector.
        eta: float
            Step size.
        S: matrix
            Sketch matrix.
        Omega: matrix
            Random test matrix.
    """
    S = (1 - eta)*S + eta * (v @ (v.conj().T @ Omega))
    return S

def nystrom_sketch_recontruct(n, S, Omega):
    """
        n: int
            Dimension of input matrix.
        S: matrix
            Sketch matrix.
        Omega: matrix
            Random test matrix.
    """
    sigma = np.sqrt(n)*np.finfo(np.float64).eps*np.linalg.norm(S, ord=2)
    S_sigma = S + sigma * Omega
    L = np.linalg.cholesky(Omega.conj().T @ S_sigma, upper=True)
    U, Sigma, _ = np.linalg.svd(np.linalg.solve(L.T, S_sigma.T).T, full_matrices=False)
    Lambda = np.maximum( 0, np.diag( np.square(Sigma) ) - sigma*np.eye( Sigma.shape[0] ) )
    return U, Lambda

if __name__ == "__main__":
    # Test approx_min_evec    
    # M_data = np.random.normal(size=(100, 50))
    # M_data = M_data @ M_data.T  # Making it symmetric positive definite
    
    # q = 100
    # eps = 1e-10

    # v = approx_min_evec(M_data, M_shape, M_mv, q, eps)
    # print("Approximate minimum eigenvector:", v)

    # xi, u = eigsh(M_data, k=1, which='SM', return_eigenvectors=True)
    # xi = xi[0]
    # u = u[:, 0]
    # print("Exact minimum eigenvector:", u)

    # print(abs( (v.T @ M_data @ v)  / np.dot( v, v ) ))
    # print("Error:", abs( np.real( np.dot( v, M_mv( M_data, v ) ) / np.dot( v, v ) ) - xi) )
    # # print(M_data)

    # Test Nystrom sketch
    n = 100
    R = 30
    Omega, S = nystrom_sketch_init(n, R)
    L = np.eye(n)

    # Test Single Rank One Update
    v = L[:, 1]
    v = v.reshape((-1, 1))
    eta = 1
    S = nystrom_sketch_rank_one_update(v, eta, S, Omega)
    U, Lambda = nystrom_sketch_recontruct(n, S, Omega)
    X_hat = U @ Lambda @ U.conj().T
    error = np.linalg.norm(X_hat - v*v.T, ord=2)
    print("Error:", error)

    # Test Reconstruction
    U, Lambda = nystrom_sketch_recontruct(n, Omega, Omega)
    X_hat = U @ Lambda @ U.conj().T
    error = np.linalg.norm(X_hat - L, ord='nuc')
    print("Error:", error)
