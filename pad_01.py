# /Users/darekdajcz/Desktop/Magisterka Studia PJATK/PAD/PAD_01/Seattle2014.csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn

# zad 1
data = pd.read_csv('/Users/darekdajcz/Desktop/Magisterka Studia PJATK/PAD/PAD_01/Zadanie_1.csv', sep=';')
data = data.apply(pd.to_numeric, errors='coerce')
matrix = data.to_numpy()
rounded_matrix = np.round(matrix, 2)

np.set_printoptions(precision=2, suppress=True)
matrix_size = np.size(matrix)
matrix_size2 = matrix.size
rows, columns = matrix.shape

print('size', matrix_size)
print('size', matrix_size2)
print('rows', rows)
print('columns', columns)

mean = np.mean(matrix)
median = np.median(matrix)
variance = np.var(matrix)
print('mean', mean)
print('median', median)
print('var', variance)

matrix_no_nan = matrix[:, ~np.isnan(matrix).any(axis=0)]

print("Macierz po usunięciu kolimn z NaN:")
print(matrix_no_nan)
mean = np.mean(matrix_no_nan)
median = np.median(matrix_no_nan)
variance = np.var(matrix_no_nan)
print('mean noNan', mean)
print('median noNan', median)
print('var noNan', variance)


# zad 2
data = pd.read_csv('/Users/darekdajcz/Desktop/Magisterka Studia PJATK/PAD/PAD_01/Zadanie_2.csv', sep=';', header=None)
matrix = data.to_numpy()

eigenvalues, eigenvectors = np.linalg.eig(matrix)

print('eigenValues', eigenvalues)
print('eigenVectors', eigenvectors)

inverse_matrix = np.linalg.inv(matrix)
print('inverse_matrix', inverse_matrix)

# zad 3
dataA = pd.read_csv('/Users/darekdajcz/Desktop/Magisterka Studia PJATK/PAD/PAD_01/Zadanie_3_macierz_A.csv', sep=',',
                    header=None)
dataB = pd.read_csv('/Users/darekdajcz/Desktop/Magisterka Studia PJATK/PAD/PAD_01/Zadanie_3_macierz_B.csv', sep=',',
                    header=None)
matrixA = dataA.to_numpy()
matrixB = dataB.to_numpy()


def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    return dot_product / (norm_vec1 * norm_vec2)


num_cols_A = matrixA.shape[1]
num_cols_B = matrixB.shape[1]

print(num_cols_A)
print(num_cols_B)

similarity_matrix = np.zeros((num_cols_A, num_cols_B))
print(similarity_matrix)
print(range(num_cols_A))
print(range(num_cols_B))

for i in range(num_cols_A):
    for j in range(num_cols_B):
        print('A i:', i)
        print('B i:', j)
        print(matrixA[:, i])
        print(matrixB[:, j])
        similarity_matrix[i, j] = cosine_similarity(matrixA[i, :], matrixB[j, :])

print("Macierz podobieństwa:")
print(similarity_matrix)