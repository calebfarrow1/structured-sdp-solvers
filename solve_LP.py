import cvxpy as cp
import numpy as np

def LP_from_LMI(A_0, A, c):
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
    n = A.shape[1]  # Number of variables
    x = cp.Variable(n)
    
    # Construct the LMI constraint
    constraints = [A_0 + A @ x >= 0]
    
    # Objective function
    objective = cp.Minimize(c.T @ x)
    
    return cp.Problem(objective, constraints)

def solve_LP(A_0, A, c):
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
    prob = LP_from_LMI(A_0, A, c)
    prob.solve(verbose=True)  # Set verbose=True for detailed output
    
    return prob.status, prob.value, prob.variables()

if __name__ == "__main__":
    # Simple example to test the LP solver (Correct answer for optimal variable is [12, 36])
    # n = 2
    # A_0 = np.array([18, 9, 9, 0, 0])
    # A = np.array([[0.3, 0.4], [0.3, 0.15], [0.17, 0.17], [-1, 0], [0, -1]])
    # c = -np.array([1.9, 2.1])

    n = 5  # Dimension of the problem
    A_0 = np.ones(n)
    A = np.random.normal(size=(n,n))
    c = np.random.normal(size=n)
    
    status, optimal_value, optimal_vars = solve_LP(A_0, A, c)
    
    print("Status:", status)
    print("Optimal Value:", optimal_value)
    print("Optimal Variables:", [var.value for var in optimal_vars])