import cvxpy as cp
import numpy as np
from scipy.linalg import toeplitz

def Formulate_LMI(A_0, A, c):
    """
    Convert a linear matrix inequality (LMI) into a linear program (LP).
    
    Parameters:
    A_0 : np.array
        The constant term in the LMI.
    A : np.ndarray
        The coefficients of the variables in the LMI.
    c : np.ndarray
        The coefficients for the objective function.
    
    Returns:
    cp.Problem
        The constructed linear program.
    """
    n = len(A)  # Number of variables
    x = cp.Variable(n)

    LA_x = A_0
    for i in range(n):
        LA_x = LA_x + A[i] * x[i]
    
    # Construct the LMI constraint
    constraints = [LA_x >> 0]
    
    # Objective function
    objective = cp.Minimize(c.T @ x)
    
    return cp.Problem(objective, constraints)

def solve_LMI(A_0, A, c, verbose=False):
    """
    Solve the linear program defined by the LMI.
    
    Parameters:
    A_0 : np.array
        The constant term in the LMI.
    A : np.ndarray
        The coefficients of the variables in the LMI.
    c : np.ndarray
        The coefficients for the objective function.
    
    Returns:
    tuple
        The status of the problem, optimal value, and optimal variable values.
    """
    prob = Formulate_LMI(A_0, A, c)
    prob.solve(verbose=verbose)  # Set verbose=True for detailed output
    
    return prob.status, prob.value, prob.variables()

if __name__ == "__main__":
    d = 3

    c = np.zeros(2*d + 1, dtype=complex)
    c[0] = 1
    A_0 = toeplitz(c)

    A = []

    for n in range(1, 2*d + 1):
        c = np.zeros(2*d + 1, dtype=complex)
        c[n] = 1
        A.append(toeplitz(c))

        c = np.zeros(2*d + 1, dtype=complex)
        c[n] = -1j
        A.append(toeplitz(c))

    c = np.random.normal(size=len(A))
    
    status, optimal_value, optimal_vars = solve_LMI(A_0, A, c)
    
    print("Status:", status)
    print("Optimal Value:", optimal_value)
    print("Optimal Variables:", [var.value for var in optimal_vars])