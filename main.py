from scipy.sparse.linalg import eigsh
from scipy.sparse import diags
import numpy as np
# import networkx as nx

np.set_printoptions(precision=3, suppress=True)

def primitive1(C_data, v):
    """
        C_data: data
            Some encoding of the matrix C.
        v: vector
            Vector to multiply with the matrix C.
    """
    return C_data @ v

def primitive2(A_data, z, u):
    """
        A_data: data
            Some encoding of the tensor A.
        z: vector
            Vector to apply A* to.
        u: vector
            Vector to multiply with A*z.
    """
    v = np.zeros(len(A_data))
    for i, a in enumerate(A_data):
        v += z[i] * (a @ u)
    return v

def primitive3(A_data, u):
    """
        A_data: data
            Some encoding of the tensor A.
        u: vector
            Vector to apply the tensor A to.
    """
    v = np.zeros(len(A_data))
    for i, a in enumerate(A_data):
        v[i] = ( (a @ u) @ u.conj().T ).trace()
    return v

def M_shape_other(M_data):
    """
        M_data: data
            Some encoding of the matrix M.
    """
    return (M_data.shape[0], M_data.shape[1])

def M_mv_other(M_data, v):
    """
        M_data: data
            Some encoding of the matrix M.
        v: vector
            Vector to multiply with the matrix M.
    """
    return M_data @ v


def M_shape(M_data):
    """
        M_data: data
            Some encoding of the matrix M.
    """
    [C, A, z] = M_data
    return C.shape

def M_mv(M_data, v):
    """
        M_data: data
            Some encoding of the matrix M.
        v: vector
            Vector to multiply with the matrix M.
    """
    [C, A, z] = M_data
    return primitive1(C, v) + primitive2(A, z, v)

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


def sketchy_CGAL(C, A, b, n, d, alpha, A_norm, R, T):
    """
        C: matrix
            Matrix C.
        A: list of matrices
            List of matrices A.
        b: vector
            Vector b.
        n: int
            Dimension of input matrix.
        d: int
            Length of A.
        alpha: float
            Trace restriction.
        A_norm: float
            Operator norm of the tensor A.
        R: int
            Size of sketch.
        T: int
            Number of iterations.
    """
    # Add code to Scale Problem data???

    beta0 = 1
    Omega, S = nystrom_sketch_init(n, R)

    z = np.zeros(d)
    y = np.zeros(d)

    for t in range(1, T+1):
        beta = beta0*np.sqrt(t+1)
        eta = 2/(t+1)
        q = int( ( t**(1/4) )*np.log(n) )

        v = approx_min_evec([C, A, y + beta*(z - b)], M_shape, M_mv, q, eps=1e-10)
        v = v.reshape((-1, 1)) # So transpose works

        z = (1 - eta)*z + eta*primitive3(A, np.sqrt(alpha)*v)
        gamma = 4*(alpha**2)*beta0*A_norm**2 / ( ( (t+1)**(3/2) )*( np.linalg.norm(z - b, ord=2)**2 ) )
        y = y + gamma*(z - b)
        S = nystrom_sketch_rank_one_update(np.sqrt(alpha)*v, eta, S, Omega)
    
    U, Lambda = nystrom_sketch_recontruct(n, S, Omega)
        
    # Add code to Enforce trace constraint??

    return U, Lambda

if __name__ == "__main__":
    # Test sketchy CGAL
    n = 100
    A = []


    # ## Original Max Cut A and C that Aidan mentioned
    # a0 = np.zeros(n)
    # a0[0] = 1
    # A.append(diags(a0))

    # for i in range(n-1):
    #     ai = np.zeros(n)
    #     ai[i] = 1
    #     ai[i+1] = -1
    #     A.append(diags(ai))

    # C = -(1/2)*diags([np.ones(n-1), np.ones(n-1), 1, 1], offsets=[-1, 1, -(n-1), n-1])



    # ## Max Cut A and C from paper
    for i in range(n):
        ai = np.zeros(n)
        ai[i] = 1
        A.append(diags(ai))

    C = -diags([2*np.ones(n), -np.ones(n-1), -np.ones(n-1), -1, -1], offsets=[0, -1, 1, -(n-1), n-1])



    d = len(A)

    Y = np.zeros((n, d))

    for i in range(d):
        x = np.zeros(n)
        x[i] = 1
        Xi = diags(x)
        Y[i,:] = primitive3(A, Xi)

    A_norm = np.linalg.norm(Y, ord=2)

    b = np.zeros(n)



    U, Lambda = sketchy_CGAL(C, A, b, n, d, alpha=n, A_norm=A_norm, R=10, T=100)


    # Objective Value Using the Sketch
    X_hat = U @ Lambda @ U.conj().T
    calculated_cut = ( X_hat.conj().T @ -C ).trace()
    print("Calculated Objective:", calculated_cut)

    # True Max Cut Objective
    x = np.zeros(n)
    x[range(0,n,2)] = 1
    x[range(1,n,2)] = -1
    x = x.reshape((-1, 1))
    max_cut = ( (x @ x.T) @ -C ).trace()
    print("Max Cut Objective:", max_cut)

    # Objective Residual as definied in the paper
    obj_residual = abs( calculated_cut - max_cut ) / ( 1 + abs(max_cut) )
    print("Objective Residual:", obj_residual)


    # ## Test cut_size (need to import networkx)
    # cycle = nx.cycle_graph(n)
    # cuts = np.sign(U)
    # for i in range(np.shape(cuts)[1]):
    #     S = set(np.asarray(cuts[:,i] == -1).nonzero()[0])
    #     T = set(np.asarray(cuts[:,i] == 1).nonzero()[0])
    #     print(nx.cut_size(cycle, S, T))

    # S = set(range(0,n+1,2))
    # T = set(range(1,n+1,2))
    # print(nx.cut_size(cycle, S, T))
                  

    # print(np.sign(U))
    # print(np.cumsum(np.sign(U), axis=0))


    # ## Test approx_min_evec
    # M_data = np.random.normal(size=(100, 50))
    # M_data = M_data @ M_data.T  # Making it symmetric positive definite
    
    # q = 100
    # eps = 1e-10

    # v = approx_min_evec(M_data, M_shape_other, M_mv_other, q, eps)
    # print("Approximate minimum eigenvector:", v)

    # xi, u = eigsh(M_data, k=1, which='SM', return_eigenvectors=True)
    # xi = xi[0]
    # u = u[:, 0]
    # print("Exact minimum eigenvector:", u)

    # print(abs( (v.T @ M_data @ v)  / np.dot( v, v ) ))
    # print("Error:", abs( np.real( np.dot( v, M_mv( M_data, v ) ) / np.dot( v, v ) ) - xi) )
    # # print(M_data)

    # ## Test Nystrom sketch
    # n = 100
    # R = 30
    # Omega, S = nystrom_sketch_init(n, R)
    # L = np.eye(n)

    # ## Test Single Rank One Update
    # v = L[:, 1]
    # v = v.reshape((-1, 1))
    # eta = 1
    # S = nystrom_sketch_rank_one_update(v, eta, S, Omega)
    # U, Lambda = nystrom_sketch_recontruct(n, S, Omega)
    # X_hat = U @ Lambda @ U.conj().T
    # error = np.linalg.norm(X_hat - v*v.T, ord=2)
    # print("Error:", error)

    # # Test Reconstruction
    # U, Lambda = nystrom_sketch_recontruct(n, Omega, Omega)
    # X_hat = U @ Lambda @ U.conj().T
    # error = np.linalg.norm(X_hat - L, ord='nuc')
    # print("Error:", error)