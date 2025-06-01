import numpy as np
import timeit
import itertools

def embedding_eigenvalues(beta, alpha, d):
    arg = np.pi * np.dot(beta, alpha) / (2*d + 1)
    return np.cos(arg), np.sin(arg)

def generate_A(n, d):
    """Input: Integers n and d corresponding to n generalized d by d 
        Toeplitz matrix.
    Output: Array A where each column is a vector corresponding to 
        diagonal matrix Ai"""
    possible_vectors = list(itertools.product(range(-2*d, 2*d + 1), repeat =
 n))
    L = len(possible_vectors)
    A = np.zeros((L, 2*L))
    for k in list(range(L)):
        beta = possible_vectors[k]
        for j in list(range(L)):
            alpha = possible_vectors[j]
            eig = embedding_eigenvalues(beta, alpha, 2*d)
            A[j, k] = eig[0]
            A[j, k+L] = eig[1]
    return A

# def faster_A(n, d):
#     possible_alpha = itertools.product(range(-2*d, 2*d + 1), repeat = n)
#     Alpha = np.array(list(possible_alpha), dtype = np.int16)
#     cutoff = ((4 * d + 1) ** n + 1) / 2 
#     Beta = Alpha[int(cutoff):] # cut off all negative betas
#     C = Alpha @ Beta.T 
#     C = np.pi * C / (4 * d + 1)
#     S = np.sin(C)
#     C = np.cos(C)
#     C = np.concatenate((C, S), 1)
#     print(np.shape(C))
#     return C

def faster_A(n, d):
    possible_alpha = itertools.product(range(0, 4 * d + 1), repeat = n)
    Alpha = np.array(list(possible_alpha), dtype = np.int16)
    possible_beta = itertools.product(range(0, 2 * d + 1), repeat = n)
    Beta = np.array(list(possible_beta), dtype = np.int16)[1::]
    C = Alpha @ Beta.T
    C = 2 * np.pi * C / (4 * d + 1)
    S = np.sin(C)
    C = np.cos(C)
    C = 2 * np.concatenate((C, S), 1)
    return C


#2d + 1^n by (2d+1)^(n-1)/2
if __name__ == "__main__":
    # A = generate_A(3, 5)
    # print(A)
    # print(np.shape(A))

    from solve_LP import solve_LP
    n = 1
    d = 1

    A = faster_A(n, d)
    print(A)
    A_0 = np.ones(np.shape(A)[0])
    c = np.random.normal(size=np.shape(A)[1])
    
    status, optimal_value, optimal_vars = solve_LP(A_0, A, c)
    
    print("Status:", status)
    print("Optimal Value:", optimal_value)
    print("Optimal Variables:", [var.value for var in optimal_vars])
