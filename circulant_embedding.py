import numpy as np
import itertools

def embedding_eigenvalues(beta, alpha, d):
    arg = np.pi * np.dot(beta, alpha) / (2*d + 1)
    return np.cos(arg), np.sin(arg)

def generate_A(n, d):
    """Input: Integers n and d corresponding to n generalized d by d Toe
plitz matrix.
    Output: Array A where each column is a vector corresponding to diago
nal matrix Ai"""
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
            A[j, 2*k] = eig[1]
    return A

A = generate_A(3, 5)
print(A)
print(np.shape(A))
