import math
import numpy as np
import random
from scipy.sparse import csr_matrix


def reverse(l, u):
    revLdata = []
    revLindices = []
    revLindptr = [0]
    nn = l.shape[1]
    qtyOfNewL = 0
    for i in range(0, nn):
        for j in range(0, i + 1):
            z = 0
            if i == j:
                aij = 1
            else:
                aij = 0
                for k in range(0, i):
                    z = z + revL[k - i, j] * l[i, k]
            revLdata.append((aij - z) / l[i, i])
            revLindices.append(j)
            qtyOfNewL += 1
        revLindptr.append(qtyOfNewL)
        revL = csr_matrix((revLdata, revLindices, revLindptr), dtype=np.float64)
    revUdata = []
    revUindices = []
    revUindptr = [0]
    nn = u.shape[1]
    qtyOfNewU = 0
    for i in range(0, nn):
        for j in range(i, nn):
            if i == j:
                revUdata.append(1)
                revUindices.append(j)
                qtyOfNewU += 1
            else:
                z = 0
                aij = u[i, j]
                for k in range(i + 1, j):
                    z = z + revUdata[len(revUdata) - j + k] * abs(u[k, j])
                revUdata.append((aij + z) * -1)
                revUindices.append(j)
                qtyOfNewU += 1
        revUindptr.append(qtyOfNewU)
        revU = csr_matrix((revUdata, revUindices, revUindptr), dtype=np.float64)
    t = revU * revL
    return t


def Jacobi(matrix, b, epsilon):
    old_array = b.copy()
    b_new = b.copy()
    print(b)
    count_of_operations = 0
    current_epsilon = 1e9
    for i in range(len(b)):
        old_array[i] = 1
    new_array = np.zeros(len(b))
    while current_epsilon > epsilon:
        current_epsilon = 0
        for i in range(matrix.shape[0]):
            new_array[i] = b_new[i]
            for j in range(matrix.shape[1]):
                count_of_operations += 1
                if i != j:
                    new_array[i] -= matrix[i, j] * old_array[j]
            new_array[i] = new_array[i] / matrix[i, i]
            current_epsilon = max(current_epsilon, abs(new_array[i] - old_array[i]))
        old_array = new_array.copy()
    print("Answer")
    print(new_array)
    print("count_of_operations", count_of_operations)
    return new_array


def gauss(Ll, Uu, bb):
    iterCount = 0
    Y = []
    for i in range(0, len(bb)):
        sum = 0
        j = 0
        while j < i:
            iterCount = iterCount + 1
            sum = sum + Ll[i, j] * Y[j]
            j = j + 1
        Y.append((bb[i] - sum) / Ll[i, i])
    i = len(Y) - 1
    X = []
    while i >= 0:
        sum = 0
        j = len(Y) - 1
        ch = 0
        while j > i:
            iterCount = iterCount + 1
            sum = sum + Uu[i, j] * X[ch]
            j = j - 1
            ch = ch + 1
        X.append((Y[i] - sum) / Uu[i, i])
        i = i - 1
    X.reverse()
    return X


def Hilbertmatrix(n):
    Hilbdata = []
    Hilbindices = []
    Hilbindptr = [0]
    qty = 0
    for i in range(1, n + 1):
        for j in range(1, n + 1):
            Hilbdata.append(1 / (i + j - 1))
            Hilbindices.append(j - 1)
            qty += 1
        Hilbindptr.append(qty)
    return csr_matrix((Hilbdata, Hilbindices, Hilbindptr))


def DiagonallyDominantMatrix(n, k):
    DDMdata = []
    DDMindices = []
    DDMindptr = [0]
    qty = 0
    for i in range(1, n + 1):
        temp = []
        tempSum = 0
        for j in range(1, n + 1):
            if (j == i):
                continue
            randomNumber = random.randint(-4, 0)
            temp.append(randomNumber)
            tempSum += randomNumber
        g = 0
        for j in range(1, n + 1):
            if (j == i):
                DDMdata.append(tempSum + pow(10, -k))
                DDMindices.append(j - 1)
                qty += 1
                continue
            DDMdata.append(temp[g])
            g += 1
            DDMindices.append(j - 1)
            qty += 1
        DDMindptr.append(qty)
    return csr_matrix((DDMdata, DDMindices, DDMindptr))


def arrayForDiagonallyDominantMatrix(k):
    array = []
    for i in range(1, k + 1):
        array.append(i)
    return array


def getLU(f):
    nn = f.shape[1]
    Ldata = np.zeros(nn * nn, dtype=float)
    Lindices = np.zeros(nn * nn, dtype=int)
    Lindptr = np.zeros(nn + 1, dtype=int)
    Udata = np.zeros(nn * nn, dtype=float)
    Uindices = np.zeros(nn * nn, dtype=int)
    Uindptr = np.zeros(nn + 1, dtype=int)
    qtyOfNewL = 0
    qtyOfNewU = 0
    for i in range(0, nn):
        for j in range(0, i + 1):
            z = 0
            for k in range(0, j):
                z = z + Ldata[qtyOfNewL - j + k] * Udata[int((2 * nn - k + 1) * k / 2 + j - k + 1) - 1]  # U[k, j]
            Ldata[qtyOfNewL] = (f[i, j] - z) / 1.0
            Lindices[qtyOfNewL] = j
            qtyOfNewL += 1
        Lindptr[i + 1] = qtyOfNewL
        for j in range(i, nn):
            z = 0
            for k in range(0, i):
                z = z + Ldata[int((1 + i) * i / 2 + k)] * Udata[
                    int((2 * nn - k + 1) * k / 2 + j - k + 1) - 1]  # L[i, k] U[k, j]
            Udata[qtyOfNewU] = (f[i, j] - z) / Ldata[int((1 + i) * i / 2 + i)]  # L[i, i]
            Uindices[qtyOfNewU] = j
            qtyOfNewU += 1
        Uindptr[i + 1] = qtyOfNewU
    L = csr_matrix((Ldata, Lindices, Lindptr))
    U = csr_matrix((Udata, Uindices, Uindptr))
    result = []
    result.append(L)
    result.append(U)
    return result


# data = numpy.array([10, -7, -3, 6, 2, 5, -1, 5])
# indices = numpy.array([0, 1, 0, 1, 2, 0, 1, 2])
# indptr = numpy.array([0, 2, 5, 8])
# f = csr_matrix((data, indices, indptr))
# data = np.array([15, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 17, 19])
# indices = np.array([0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3])
# indptr = np.array([0, 4, 8, 12, 16])
# data = np.array([1, 8, 7, 3, 7, 5, 4, 6, 5])
# indices = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2])
# indptr = np.array([0, 3, 6, 9])
# A = csr_matrix((data, indices, indptr))
n = 1000
k = 8


A = DiagonallyDominantMatrix(n, k)
xStar = arrayForDiagonallyDominantMatrix(n)
F = A * xStar
LU = getLU(A)
L = LU[0]
U = LU[1]

x = gauss(L, U, F)

calcError = 0
for i in range(0, n):
    calcError = max(calcError, abs(xStar[i] - x[i]))
print("pogreshost is: ", calcError)
"""
delta = 0
for i in range(0, n):
    delta += abs(xStar[i] - x[i])
calcError = 0
for i in range(0, n):
    calcError = max(calcError, abs(xStar[i] - x[i]))
delta = delta / n
print("k is: ", k)
print("delta is: ", delta)
print("pogreshost is: ", calcError)
"""
# print(F)
# print(xStar)
# print(gauss(L, U, F))
# b = numpy.array([0, 1, 2])
# Jacobi(f, b, epsilon)
