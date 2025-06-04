from solve_LMI import solve_LMI
from solve_LP import solve_LP
import numpy as np
from scipy.linalg import toeplitz
from circulant_embedding import faster_A
from scipy.linalg import circulant
from scipy.sparse import diags, csr_array
from SCGAL import run_solver

if __name__ == '__main__':
    n = 1
    d = 1

    # # Circulant embedding
    # A = faster_A(n, d)
    # A_0 = np.ones(np.shape(A)[0])
    # c = np.random.normal(size=np.shape(A)[1])
    
    # status, optimal_value, optimal_vars = solve_LP(A_0, A, c, verbose=True)
    
    # print("LP Status:", status)
    # print("LP Optimal Value:", optimal_value)
    # print("LP Optimal Variables:", [var.value for var in optimal_vars])

    # LMI formulation with toeplitz constraints
    col = np.zeros(2*d + 1, dtype=complex)
    col[0] = 1
    A_0 = toeplitz(col)

    A = []

    for m in range(1, 2*d + 1):
        col = np.zeros(2*d + 1, dtype=complex)
        col[m] = 1
        mx = toeplitz(col)
        A.append(mx)

    for m in range(1, 2*d + 1):
        col = np.zeros(2*d + 1, dtype=complex)
        col[m] = -1j
        mx = toeplitz(col)
        A.append(mx)

    c = np.random.normal(size=len(A))
    # c = np.array([0.537667139546100, 1.83388501459509, -2.25884686100365, 0.862173320368121])
    
    status, optimal_value, optimal_vars = solve_LMI(A_0, A, c, verbose=False)
    
    print("Toeplitz LMI Status:", status)
    print("Toeplitz LMI Optimal Value:", optimal_value)
    print("Toeplitz LMI Optimal Variables:", [var.value for var in optimal_vars])

    # # LMI formulation with circulant constraints
    # col = np.zeros(4*d + 1, dtype=complex)
    # col[0] = 1
    # A_0 = circulant(col)

    # A = []

    # for n in range(1, 2*d + 1):
    #     col = np.zeros(4*d + 1, dtype=complex)
    #     col[n] = 1
    #     col[-n] = 1
    #     A.append(circulant(col))

    # for n in range(1, 2*d + 1):
    #     col = np.zeros(4*d + 1, dtype=complex)
    #     col[n] = -1j
    #     col[-n] = 1j
    #     A.append(circulant(col))
    
    # status, optimal_value, optimal_vars = solve_LMI(A_0, A, c, verbose=True)
    
    # print("Circulant LMI Status:", status)
    # print("Circulant LMI Optimal Value:", optimal_value)
    # print("Circulant LMI Optimal Variables:", [var.value for var in optimal_vars])


    # SPARSE LMI formulation for Sketchy CGAL
    diag = np.ones(2*d + 1, dtype=complex)
    A_0 = diags(diag)

    A = []

    for m in range(1, 2*d + 1):
        diag = np.ones(2*d + 1 - m, dtype=complex)
        mx = diags([diag, diag],offsets=[-m,m])
        A.append(mx)

    for m in range(1, 2*d + 1):
        diag = np.ones(2*d + 1 - m, dtype=complex)
        mx = diags([-1j*diag, 1j*diag],offsets=[-m,m])
        A.append(mx)

    # Y = np.zeros((2*d + 1, len(A)), dtype=complex)

    # for i in range(2*d + 1):
    #     row = np.array([0])
    #     col = np.array([i])
    #     data = np.array([1])
    #     Xi = csr_array((data, (row, col)), shape=(2*d + 1, 2*d + 1))
    #     v = np.zeros(len(A), dtype=complex)
    #     for k, a in enumerate(A):
    #         v[k] = ( a.conj().T @ Xi ).trace()
    #     Y[i,:] = v

    Y = np.zeros((len(A), ( 2*d + 1 )**2), dtype=complex)
    
    for i in range(len(A)):
        mx = np.matrix.flatten(A[i].toarray())
        Y[i, :] = mx

    

    A_norm = np.linalg.norm(Y, ord=2)
    # A_norm = 2 # !!! Remove this once the norm is calculated correctly


    U, Lambda, objective, Omega, z, y, S = run_solver(A_0, A, c, 2*d + 1, 4*d, alpha=n, A_norm=A_norm, R=1, T=10000, trace_mode='min', max_restarts=100)

    # print(Lambda.diagonal())


    # Objective Value Using the Sketch
    X_hat = U @ Lambda @ U.conj().T
    print(X_hat.diagonal())
    calculated_cut = np.real( ( X_hat.conj().T @ A_0 ).trace() )
    print("Calculated Objective:", calculated_cut)

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