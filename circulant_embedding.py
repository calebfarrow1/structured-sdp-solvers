import numpy as np
import timeit
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
            A[j, k+L] = eig[1]
    return A

def faster_A(n, d):
    possible_alpha = itertools.product(range(-d, d + 1), repeat = n)
    Alpha = np.array(list(possible_alpha), dtype = np.int16)
    cutoff = ((2 * d + 1) ** n + 1) / 2 
    Beta = Alpha[int(cutoff):] # cut off all negative betas
    C = Alpha @ Beta.T 
    C = np.pi * C / (2 * d + 1)
    S = np.sin(C)
    C = np.cos(C)
    C = np.concatenate((C, S), 1)
    print(np.shape(C))
    return C


#2d + 1^n by (2d+1)^(n-1)/2
