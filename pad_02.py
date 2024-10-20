import inline
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn

seaborn.set()

print(pd.__version__)
print(np.__version__)
data = pd.read_csv('/Users/darekdajcz/Downloads/OneDrive_1_12.10.2024/president_heights.csv')
heights = np.array(data['height(cm)'])
print(heights)

mean_height = np.mean(heights)
std_dev = np.std(heights)
min = np.min(heights)
max = np.max(heights)
x25th = np.percentile(heights, 25)
mHeght = np.median(heights)
p75 = np.percentile(heights, 75)

# ZAD 1
print("Mean height: ", mean_height)
print("Standard deviation: ", std_dev)
print("Minimum height: ", min)
print("Maximum height: ", max)
print("25th percentile: ", x25th)
print("Median: ", mHeght)
print("75th percentile: ", p75)


data2 = pd.read_csv('/Users/darekdajcz/Downloads/OneDrive_1_12.10.2024/Zadanie_2.csv', sep=';', header=None)

matrix = data2.to_numpy()
print(data2)
print('######')
print(matrix)

# ZAD 2
eigenvalues, eigenvectors = np.linalg.eig(matrix)
print("Wartości własne:")
print(eigenvalues)

print("Wektory własne:")
print(eigenvectors)

matrix2 = np.linalg.inv(matrix)
print("Macierz odwrotna:")
print(matrix2)

rainfall = pd.read_csv('/Users/darekdajcz/Downloads/OneDrive_1_12.10.2024/Seattle2014.csv')['PRCP'].values
inches = rainfall / 254.0  # 1/10mm -> inches
# var = inches.shape
plt.hist(inches, 20)
plt.show()

days_without_rain = np.sum(inches == 0)
days_with_rain = np.sum(inches > 0)
days_with_heavy_rain = np.sum(inches > 0.5)
light_rain_days = np.sum(inches < 0.2) - days_without_rain

print("Number of days without rain:", days_without_rain)
print("Number of days with rain:", days_with_rain)
print("Days with more than 0.5 inches:", days_with_heavy_rain)
print("Rainy days with < 0.2 inches:", light_rain_days)

median_rainy_days = np.median(inches[inches > 0])

days = np.arange(len(inches)) + 1
summer_days_filter = (days >= 172) & (days <= 262)
summer_rain_inches = inches[summer_days_filter]
median_summer_rain = np.median(summer_rain_inches)

argmax_summer_rain_ = np.argmax(summer_rain_inches)
max_summer_rain = np.max(summer_rain_inches)

non_summer_rain = inches[~summer_days_filter]
argmax_non_summer_rain = np.argmax(non_summer_rain)
max_non_summer_rain = np.max(non_summer_rain)

print("Median precip on rainy days in 2014 (inches):", median_rainy_days)
print("Median precip on summer days in 2014 (inches):", median_summer_rain)
print("Maximum precip on summer days in 2014 (inches):", max_summer_rain)
print("Maximum precip on non-summer rainy days (inches):", max_non_summer_rain)

A = [0, 3, 2, 5]
B = [0, 3, 1, 4]

add = np.add(A, B)
subtract = np.subtract(A, B)
multiply_A = np.multiply(A, 4)
length_B = np.linalg.norm(B)

dot = np.dot(A, B)
print('add', add)
print('subtract', subtract)
print('multiplyA', multiply_A)
print('length_B', length_B)
