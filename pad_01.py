# /Users/darekdajcz/Desktop/Magisterka Studia PJATK/PAD/PAD_01/Seattle2014.csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn

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