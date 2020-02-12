# LINEAR ALGEBRA OPERATIONS --------------------------------
import numpy as np
import pandas as pd
from math import factorial

# arrays ----------------
# simple array
A = np.array([[1,2,3],[1,2,3],[1,2,3]])
# empty/zeros/ones array
A = np.empty([3,3])
A = np.zeros([3,5])
A = np.ones([5, 5])

# vectors ---------------
a = np.array([1,5,8,15,3])
b = a + 2
# vector multiplication
c = a * b
# vector dot-product
c = a.dot(b)
# vector norm
l2 = np.linalg.norm(a)

# matrices --------------
A = np.array([[1,2,3],[1,2,3],[1,2,3]])
B = np.array([[3.5,2,1.2],[6,3.2,9.8],[1,0.8,4]])
# basic operations
C = A * B
C = A / B
C = A.dot(B)
# scalar multiplication
C = A.dot(2.2)
# lower, upper, diagonal
lower = np.tril(A)
upper = np.triu(A)
d = np.diag(A)
# identity
I = np.identity(3)

# other matrix operations
# transpose
B = A.T
# inversion
B = np.linalg.inv(I)
# trace
B = np.trace(A)
# determinant, rank
B = np.linalg.det(A)
r = np.linalg.matrix_rank(A)

# matrix factorization -----
# matrix factorization, or matrix decomposition, breaks a matrix down into its constituent parts to make other
# operations simpler and more numerically stable.

# LU Decomposition
from scipy.linalg import lu
P, L, U = lu(A)
# QR Decomposition
from numpy.linalg import qr
Q, R = qr(A, 'complete')
# Eigendecomposition
from numpy.linalg import eig
values, vectors = eig(A)
# Singular-Value Decomposition
from scipy.linalg import svd
U, s, V = svd(A)

# statistics ---------------
# least-squares
from numpy.linalg import lstsq
b = np.linalg.lstsq(X, y)

# other --------------------
a = np.array([1,5,8,15,3])
b = a + 2
np.vstack([a, b]) # rbind(a, b)
np.hstack([a, b]) # cbind(a, b)
np.concatenate([a, b]) # c(a, b)
