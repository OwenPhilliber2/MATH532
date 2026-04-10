import numpy as np
from GSVD import GSVD

A = np.array([[1,2],[0,1],[2,1],[0,-1],[1,1]])
B = np.array([[1,1],[0,1],[2,0],[0,1]])

U, V, C, S, X = GSVD(A,B)

# Part a is printed when running code
print("\n")
# Part b
print("Generalized singular value pairs for A and B:")
print(np.diag(C))
print(np.diag(S))
print("\n")


# Part d
print("Cosine matrix squared + Sine matrix squared")
print(C ** 2 + S ** 2)