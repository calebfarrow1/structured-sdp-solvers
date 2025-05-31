import numpy as np
import itertools

def embedding_eigenvalues(beta, alpha, d):
    arg = np.pi * np.dot(beta, alpha) / (2*d + 1)
    return np.cos(arg), np.sin(arg)

def generate_A(n, d):
    """Input: Integers n and d corresponding to n generalized d by d 
        Toeplitz matrix.
    Output: Array A where each column is a vector corresponding to 
        diagonal matrix Ai"""
    possible_vectors = list(itertools.product(range(-d, d + 1), repeat =
 n))
    L = len(possible_vectors)
    A = np.zeros((L, 2*L))
    for k in list(range(L)):
        beta = possible_vectors[k]
        for j in list(range(L)):
            alpha = possible_vectors[j]
            eig = embedding_eigenvalues(beta, alpha, d)
            A[j, k] = eig[0]
            A[j, k + L] = eig[1]
    return A

if __name__ == "__main__":
    # A = generate_A(3, 5)
    # print(A)
    # print(np.shape(A))

    from solve_LP import solve_LP
    n = 3
    d = 5

    A = generate_A(n, d)
    A_0 = np.ones(np.shape(A)[0])
    c = np.random.normal(size=np.shape(A)[1])
    
    status, optimal_value, optimal_vars = solve_LP(A_0, A, c)
    
    print("Status:", status)
    print("Optimal Value:", optimal_value)
    print("Optimal Variables:", [var.value for var in optimal_vars])
