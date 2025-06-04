from scipy.sparse.linalg import eigsh
from scipy.sparse import diags
import numpy as np
import time
import math
# import networkx as nx

np.set_printoptions(precision=3, suppress=True)

def primitive1(C_data, v):
    """
        C_data: data
            Some encoding of the matrix C.
        v: vector
            Vector to multiply with the matrix C.
    """
    return  C_data @ v

def primitive2(A_data, z, u, A_shape):
    """
        A_data: data
            Some encoding of the tensor A.
        z: vector
            Vector to apply A* to.
        u: vector
            Vector to multiply with A*z.
    """
    v = np.zeros(A_shape(A_data), dtype=complex)
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
    v = np.zeros(len(A_data), dtype=complex)
    for i, a in enumerate(A_data):
        v[i] = ( u.conj().T @ (a @ u) ).trace()
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

def A_shape(A_data):
    return A_data[0].shape[0]

def M_shape(M_data):
    """
        M_data: data
            Some encoding of the matrix M.
    """
    [C, A, z] = M_data
    return C.shape

def M_mv(M_data, v, A_shape):
    """
        M_data: data
            Some encoding of the matrix M.
        v: vector
            Vector to multiply with the matrix M.
    """
    [C, A, z] = M_data
    return primitive1(C, v) + primitive2(A, z, v, A_shape)

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
    # print(omega.shape)

    iter = 0
    for i in range(min(q, n-1)):
        omega[i+1] = np.real( np.dot( v, M_mv( M_data, v, A_shape ) ) )
        v, v_old = M_mv( M_data, v, A_shape ) - omega[i+1]*v - rho[i]*v_old, v
        rho[i+1] = np.linalg.norm(v) # This is different than what is shown in the paper, we believe the paper has a typo
        iter = i
        if rho[i+1] < eps:
            break
        v = v / rho[i+1]

    i = iter
    # print(rho)
    # print(omega)
    T = diags([rho[1:i+1], omega[1:i+2], rho[1:i+1]], offsets=[-1, 0, 1]).toarray()
    # print(T)
    xi, u = eigsh(T, k=1, which='SA', return_eigenvectors=True) # We think the paper is using SA not SM which we thought originally
    u = u[:, 0]
    xi = xi[0]
    
    v = v0
    v_sum = np.zeros(n, dtype=complex)
    v_old = np.zeros(n, dtype=complex)

    for i in range(min(q, n-1)):
        v_sum += u[i]*v
        v, v_old = M_mv( M_data, v, A_shape ) - omega[i+1]*v - rho[i]*v_old, v
        if rho[i+1] < eps:
            break
        v = v / rho[i+1]

    return xi, v_sum

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


def sketchy_CGAL(C, A, b, n, d, alpha, A_norm, R, T,
                 enforce_trace=True,
                 enforce_A_norm=True,
                 normalize_A=(lambda tensor, scalar : [scalar * matrix for matrix in tensor]),
                 normalize_b=(lambda tensor, scalar : scalar * tensor),
                 do_log=False, log_data=[], logging_function=None, trace_mode='eq', max_restarts=0, Omega=None, S=None, z=None, y=None, eps=10e-10):
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

    if do_log:
        epoc = time.time()

    if enforce_A_norm:
        A = normalize_A(A, 1/A_norm)
        b = normalize_b(b, 1/A_norm)

    TRACE = 0

    for t in range(1, T+1):
        beta = beta0*np.sqrt(t+1)
        eta = 2/(t+1)
        q = math.ceil( ( t**(1/4) )*np.log(n + 0.1) )

        xi, v = approx_min_evec([C, A, y + beta*(z - b)], M_shape, M_mv, q, eps=1e-10)
        v = v.reshape((-1, 1)) # So transpose works

        temp_alpha = alpha
        if trace_mode != 'eq':
            if xi >= 0:
                temp_alpha = 0
                # v = np.zeros_like(v)

        TRACE = (1-eta)*TRACE + eta*temp_alpha

        z = (1 - eta)*z + eta*primitive3(A, np.sqrt(temp_alpha)*v)
        gamma = min( beta0, 4*( temp_alpha**2 )*beta0*( A_norm**2 ) / ( ( (t+1)**(3/2) )*( np.linalg.norm(z - b, ord=2)**2 ) ) )
        # gamma = min( beta0, 16*( temp_alpha**2 )*beta0*( A_norm**2 ) / ( ( (t+1)*(t+2)**(1/2) )*( np.linalg.norm(z - b, ord=2)**2 ) ) ) # This is technicall what they are doing
        y = y + gamma*(z - b)
        S = nystrom_sketch_rank_one_update(np.sqrt(temp_alpha)*v, eta, S, Omega)
        if do_log:
            log_data.append(
                logging_function(beta, eta, q, v, z, gamma, y, S, Omega, epoc)
            )
    
    U, Lambda = nystrom_sketch_recontruct(n, S, Omega)
        
    # if enforce_trace and trace_mode == 'eq': # Only allow enforcement of trace if equality of trace is required
    if enforce_trace:
        # Lambda += (alpha - Lambda.trace()) * np.eye(R) / R
        # print(TRACE)
        # print(Lambda.trace())
        Lambda += (TRACE - Lambda.trace()) * np.eye(R) / R

    return U, Lambda, Omega, z, y, S

def run_solver(C, A, b, n, d, alpha, A_norm, R, T,
                 enforce_trace=True,
                 enforce_A_norm=True,
                 normalize_A=(lambda tensor, scalar : [scalar * matrix for matrix in tensor]),
                 normalize_b=(lambda tensor, scalar : scalar * tensor),
                 do_log=False, log_data=[], logging_function=None, trace_mode='eq', max_restarts=0, hot_start=False, 
                 Omega=None, S=None, z=None, y=None, eps=1e-10):
    """
        Wrapper function to run the sketchy CGAL solver.
    """
    
    if not hot_start:
        Omega, S = nystrom_sketch_init(n, R)

        z = np.zeros(d)
        y = np.zeros(d)

    if trace_mode == 'min':
        iter = 0
        objective = None
        prev_objective = None
        while iter <= max_restarts:
            # Omega, S = nystrom_sketch_init(n, R) # Should most likely remove this line eventually
            # z = np.zeros(d) # Should most likely remove this line eventually
            # y = np.zeros(d) # Should most likely remove this line eventually
            iter += 1
            U, Lambda, Omega, z, y, S = sketchy_CGAL(C, A, b, n, d, alpha, A_norm, R, T,
                        enforce_trace=enforce_trace,
                        enforce_A_norm=enforce_A_norm,
                        normalize_A=normalize_A,
                        normalize_b=normalize_b,
                        do_log=do_log,
                        log_data=log_data,
                        logging_function=logging_function,
                        trace_mode=trace_mode,
                        max_restarts=max_restarts,
                        Omega=Omega, S=S, z=z, y=y, eps=eps)
            
            if objective is not None:
                prev_objective = objective

            objective = ( (U.conj().T @ C.conj().T @ U) @ Lambda).trace()

            print(iter-1, objective)

            if prev_objective is not None and abs(objective - prev_objective) < eps:
                break
            else:
                alpha *= 2
    else:
        U, Lambda, Omega, z, y, S = sketchy_CGAL(C, A, b, n, d, alpha, A_norm, R, T,
            enforce_trace=enforce_trace,
            enforce_A_norm=enforce_A_norm,
            normalize_A=normalize_A,
            normalize_b=normalize_b,
            do_log=do_log,
            log_data=log_data,
            logging_function=logging_function,
            trace_mode=trace_mode,
            max_restarts=max_restarts,
            Omega=Omega, S=S, z=z, y=y, eps=eps)
        
        objective = ( (U.conj().T @ C.conj().T @ U) @ Lambda).trace()


    return U, Lambda, objective, Omega, z, y, S


if __name__ == "__main__":
    # # Test sketchy CGAL
    # n = 100
    # A = []


    # # ## Original Max Cut A and C that Aidan mentioned
    # # a0 = np.zeros(n)
    # # a0[0] = 1
    # # A.append(diags(a0))

    # # for i in range(n-1):
    # #     ai = np.zeros(n)
    # #     ai[i] = 1
    # #     ai[i+1] = -1
    # #     A.append(diags(ai))

    # # C = -(1/2)*diags([np.ones(n-1), np.ones(n-1), 1, 1], offsets=[-1, 1, -(n-1), n-1])



    # # ## Max Cut A and C from paper
    # for i in range(n):
    #     ai = np.zeros(n)
    #     ai[i] = 1
    #     A.append(diags(ai))

    # C = -diags([2*np.ones(n), -np.ones(n-1), -np.ones(n-1), -1, -1], offsets=[0, -1, 1, -(n-1), n-1]) / math.sqrt(6 * n)

    # d = len(A)

    # Y = np.zeros((n, d))

    # for i in range(d):
    #     x = np.zeros(n)
    #     x[i] = 1
    #     Xi = diags(x)
    #     v = np.zeros(len(A))
    #     for i, a in enumerate(A):
    #         v[i] = (a @ Xi ).trace()
    #     Y[i,:] = v

    # A_norm = np.linalg.norm(Y, ord=2)

    # b = np.ones(n)



    # U, Lambda, objective, Omega, z, y, S = run_solver(C, A, b, n, d, alpha=n, A_norm=A_norm, R=10, T=100)

    # print(Lambda.diagonal())


    # # Objective Value Using the Sketch
    # X_hat = U @ Lambda @ U.conj().T
    # print(X_hat.diagonal())
    # calculated_cut = ( X_hat.conj().T @ -C ).trace()
    # print("Calculated Objective:", calculated_cut)

    # # True Max Cut Objective
    # x = np.zeros(n)
    # x[range(0,n,2)] = 1
    # x[range(1,n,2)] = -1
    # x = x.reshape((-1, 1))
    # max_cut = ( (x @ x.T) @ -C ).trace()
    # print("Max Cut Objective:", max_cut)

    # # Objective Residual as definied in the paper
    # obj_residual = abs( calculated_cut - max_cut ) / ( 1 + abs(max_cut) )
    # print("Objective Residual:", obj_residual)


    # # ## Test cut_size (need to import networkx)
    # # cycle = nx.cycle_graph(n)
    # # cuts = np.sign(U)
    # # for i in range(np.shape(cuts)[1]):
    # #     S = set(np.asarray(cuts[:,i] == -1).nonzero()[0])
    # #     T = set(np.asarray(cuts[:,i] == 1).nonzero()[0])
    # #     print(nx.cut_size(cycle, S, T))

    # # S = set(range(0,n+1,2))
    # # T = set(range(1,n+1,2))
    # # print(nx.cut_size(cycle, S, T))
                  

    # # print(np.sign(U))
    # # print(np.cumsum(np.sign(U), axis=0))


    # # ## Test approx_min_evec
    # # M_data = np.random.normal(size=(100, 50))
    # # M_data = M_data @ M_data.T  # Making it symmetric positive definite
    
    # # q = 100
    # # eps = 1e-10

    # # xi, v = approx_min_evec(M_data, M_shape_other, M_mv_other, q, eps)
    # # print("Approximate minimum eigenvector:", v)

    # # xi, u = eigsh(M_data, k=1, which='SM', return_eigenvectors=True)
    # # xi = xi[0]
    # # u = u[:, 0]
    # # print("Exact minimum eigenvector:", u)

    # # print(abs( (v.T @ M_data @ v)  / np.dot( v, v ) ))
    # # print("Error:", abs( np.real( np.dot( v, M_mv( M_data, v, A_shape ) ) / np.dot( v, v ) ) - xi) )
    # # # print(M_data)

    # # ## Test Nystrom sketch
    n = 100
    R = 30
    Omega, S = nystrom_sketch_init(n, R)
    L = np.eye(n)

    # # ## Test Single Rank One Update
    # # v = L[:, 1]
    # # v = v.reshape((-1, 1))
    # # eta = 1
    # # S = nystrom_sketch_rank_one_update(v, eta, S, Omega)
    # # U, Lambda = nystrom_sketch_recontruct(n, S, Omega)
    # # X_hat = U @ Lambda @ U.conj().T
    # # error = np.linalg.norm(X_hat - v*v.T, ord=2)
    # # print("Error:", error)

    # Test Reconstruction
    U, Lambda = nystrom_sketch_recontruct(n, Omega, Omega)
    X_hat = U @ Lambda @ U.conj().T
    error = np.linalg.norm(X_hat - L, ord='nuc')
    print("Error:", error)
