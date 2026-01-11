from utils import mean, mean_centred_data, calculate_cov_matrix, mat_vec_mult, magnitude 
import csv 
import numpy as np 
import matplotlib.pyplot as plt 
data = [ [78, 72, 80], [65, 60, 58], [90, 88, 92], [55, 50, 52], [82, 79, 85], [70, 68, 72], [60, 62, 61], [88, 85, 90], [73, 70, 75], [68, 65, 70] ] 
## step 1 : Calculate the mean 
mean, n, d = mean(data) 
print(mean) 
## step 2 : Mean centred data 
mean_centred = mean_centred_data(data, mean, n, d) 
print(mean_centred) 
## step 3 : Covariance matrix 
cov_matrix = calculate_cov_matrix(mean_centred, n, d) 
print(cov_matrix) 
## step 4 : Power Iteration 
v = [1,0,0] # initial vector 
for _ in range(100): 
    v_new = mat_vec_mult(cov_matrix,v) 
    mag = magnitude(v_new) 
    v = [x/mag for x in v_new] 
    print("Eigen Vector:",v) 
## step 5 : Eigen Value 
lambda_value = 0 
temp = mat_vec_mult(cov_matrix,v) 
for i in range(d): 
    lambda_value += v[i]*temp[i]
    print("Eigen Value:",lambda_value) 
## step 6 : Deflation 
new_cov_matrix = [[0 for _ in range(d)] for _ in range(d)] 
for i in range(d): 
    for j in range(d): 
        new_cov_matrix[i][j] = cov_matrix[i][j] - lambda_value * v[i] * v[j] 
        print(new_cov_matrix) 
# Power iteration to find second eigenvector 
v2 = [1, 0, 0] # initial guess 
for _ in range(100): 
    v_new = mat_vec_mult(new_cov_matrix, v2) 
    mag = magnitude(v_new) 
    v2 = [x / mag for x in v_new] # Optional: compute second eigenvalue 
    temp2 = mat_vec_mult(new_cov_matrix, v2) 
    lambda2 = sum(v2[i]*temp2[i] for i in range(d)) 
    print("Second Eigenvector:", v2) 
    print("Second Eigenvalue:", lambda2) 
#### step 7 : Projection 
z1 = [sum(mean_centred[i][j]*v[j] for j in range(d)) for i in range(n)] 
# PC1 
z2 = [sum(mean_centred[i][j]*v2[j] for j in range(d)) for i in range(n)] # PC2 
# Combine into 2D projections 
Z_2D = list(zip(z1, z2)) 
print("2D Projected Data:", Z_2D) 
# --- Save to CSV --- 
with open('z_2d.csv', 'w', newline='') as f: 
    writer = csv.writer(f) 
    writer.writerow(['PC1','PC2']) # header 
    writer.writerows(Z_2D) # write all 2D points 
### NUMPY VALIDATION 
# Convert covariance matrix to NumPy array 
cov_np = np.array(cov_matrix) # NumPy eigen decomposition 
eigvals_np, eigvecs_np = np.linalg.eig(cov_np)
 # Sort eigenvalues and eigenvectors in descending order 
idx = np.argsort(eigvals_np)[::-1] 
eigvals_np = eigvals_np[idx] 
eigvecs_np = eigvecs_np[:, idx] 
print("NumPy Eigenvalues:", eigvals_np) 
print("NumPy First Eigenvector:", eigvecs_np[:, 0]) 
# ---- Compare eigenvalues ---- 
print("Custom Eigenvalue:", lambda_value) 
print("NumPy Largest Eigenvalue:", eigvals_np[0]) 
# ---- Cosine similarity ---- 
v_custom = np.array(v) 
v_numpy = eigvecs_np[:, 0] 
cosine_similarity = np.dot(v_custom, v_numpy) / ( np.linalg.norm(v_custom) * np.linalg.norm(v_numpy) ) 
print("Cosine Similarity:", cosine_similarity) 
# ---- Explained Variance ---- 
total_variance = eigvals_np.sum() 
explained_variance_ratio = eigvals_np / total_variance 
cumulative_variance = explained_variance_ratio.cumsum() 
print("Explained Variance Ratio:", explained_variance_ratio) 
print("Cumulative Explained Variance:", cumulative_variance) 
print("Total Variance:", total_variance) 
# ---- Reconstruction from 2 PCs ---- 
W = np.column_stack((v, v2)) 
# principal components 
Z = np.column_stack((z1, z2)) 
# projected data 
mu = np.array(mean)
# Reconstruct data 
X_reconstructed = Z @ W.T + mu 
# Original data 
X_original = np.array(data) 
# Mean Squared Error 
mse = np.mean((X_original - X_reconstructed) ** 2) 
print("Reconstruction MSE:", mse)