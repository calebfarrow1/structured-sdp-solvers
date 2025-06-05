import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from solve_LMI import solve_LMI
from solve_LP import solve_LP
from scipy.linalg import toeplitz
from circulant_embedding import faster_A
#from scipy.linalg import circulant
from scipy.sparse import diags, csr_array
from SCGAL import run_solver


# Define the range of d values
d_values = [3, 5, 8, 10]

# Initialize a list to store results
results = []

for d in d_values:
    # Generate problem data for the current d


    # LMI formulation with toeplitz constraints. Code copied from main
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
    #print("C Norm", np.linalg.norm(c, ord=1))

    # Run LMI Solver
    start_time = time.time()
    status_LMI, optimal_value_LMI, optimal_vars_LMI = solve_LMI(A_0, A, c, verbose=False)
    time_LMI = time.time() - start_time




    # SPARSE LMI formulation for Sketchy CGAL. Code copied from main
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

    Y = np.zeros((len(A), ( 2*d + 1 )**2), dtype=complex)
    
    for i in range(len(A)):
        mx = np.matrix.flatten(A[i].toarray())
        Y[i, :] = mx

    A_norm = np.linalg.norm(Y, ord=2)

    # Run SCGAL Solver
    start_time = time.time()
    U, Lambda, objective, Omega, z, y, S = run_solver(A_0, A, c, 2*d + 1, 4*d,
                                                      alpha=np.linalg.norm(c,ord=1), A_norm=A_norm,
                                                      R=1, T=5000,
                                                      trace_mode='min', max_restarts=3)
    time_SCGAL = time.time() - start_time




    # Circulant Embedding data. Code copied from circulant_embedding
    n = 2*d + 1
    A = faster_A(n, d)
    A_0 = np.ones(np.shape(A)[0])
    c = np.random.normal(size=np.shape(A)[1])

    # Circulant Embedding LP Solver
    start_time = time.time()
    status_circ, optimal_value_circ, optimal_vars_circ = solve_LP(A_0, A, c)
    time_circulant = time.time() - start_time

    # Append results
    results.append({
        'd': d,
        'Objective_LMI': optimal_value_LMI,
        'Time_LMI': time_LMI,
        'Objective_SCGAL': objective,
        'Time_SCGAL': time_SCGAL,
        'Objective_Circulant': optimal_value_circ,
        'Time_Circulant': time_circulant
    })



# ----------------- Plotting and stuff -----------------
df_results = pd.DataFrame(results)

# Time vs d
plt.figure(figsize=(10, 6))
plt.plot(df_results['d'], df_results['Time_LMI'], label='LMI Solver')
plt.plot(df_results['d'], df_results['Time_SCGAL'], label='SCGAL Solver')
plt.plot(df_results['d'], df_results['Time_Circulant'], label='Circulant Embedding LP')
plt.xlabel('d')
plt.ylabel('Computation Time (s)')
plt.title('Solver Computation Time vs. d')
plt.legend()
plt.grid(True)
plt.show()

