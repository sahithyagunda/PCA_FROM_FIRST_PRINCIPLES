import csv
import numpy as np
import matplotlib.pyplot as plt
from utils import mean, mean_centred_data, calculate_cov_matrix, mat_vec_mult, magnitude

iris_data = []
with open("data/iris.csv", "r") as f:
    reader = csv.reader(f)
    next(reader)
    for row in reader:
        iris_data.append([float(x) for x in row[:4]])

# Step 1: Calculate the mean
mean_iris, n_iris, d_iris = mean(iris_data)
# Step 2: Mean centred data
mean_centred_iris = mean_centred_data(iris_data, mean_iris, n_iris, d_iris)
# Step 3: Covariance matrix
cov_matrix_iris = calculate_cov_matrix(mean_centred_iris, n_iris, d_iris)
# Step 4: Power Iteration for first eigenvector
v_iris = [1, 0, 0, 0]  # initial vector
for _ in range(100):
    v_new_iris = mat_vec_mult(cov_matrix_iris, v_iris)
    mag_iris = magnitude(v_new_iris)
    v_iris = [x / mag_iris for x in v_new_iris]
# Step 5: Eigen Value
lambda_value_iris = 0
temp_iris = mat_vec_mult(cov_matrix_iris, v_iris)
for i in range(d_iris):
    lambda_value_iris += v_iris[i] * temp_iris[i]
# Step 6: Deflation
new_cov_matrix_iris = [[0 for _ in range(d_iris)] for _ in range(d_iris)]
for i in range(d_iris):
    for j in range(d_iris):
        new_cov_matrix_iris[i][j] = cov_matrix_iris[i][j] - lambda_value_iris * v_iris[i] * v_iris[j]
# Power iteration to find second eigenvector
v2_iris = [1, 0, 0, 0]  # initial guess
for _ in range(100):
    v_new2_iris = mat_vec_mult(new_cov_matrix_iris, v2_iris)
    mag2_iris = magnitude(v_new2_iris)
    v2_iris = [x / mag2_iris for x in v_new2_iris]
# Step 7: Projection
z1_iris = [sum(mean_centred_iris[i][j] * v_iris[j] for j in range(d_iris)) for i in range(n_iris)]
z2_iris = [sum(mean_centred_iris[i][j] * v2_iris[j] for j in range(d_iris)) for i in range(n_iris)]
Z_iris = list(zip(z1_iris, z2_iris))

# Save Iris projection
with open("z_2d_iris.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["PC1", "PC2"])
    writer.writerows(Z_iris)

# Plot
fig = plt.figure(figsize=(12,5))
z1, z2 = zip(*Z_iris)
plt.scatter(z1, z2)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("PCA Projection of Iris Dataset (From Scratch)")
plt.grid(True)
plt.show()
fig.savefig("pca_iris_from_scratch.png")
