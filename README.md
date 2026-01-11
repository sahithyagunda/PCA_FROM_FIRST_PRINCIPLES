#### MATHEMATICAL FOUNDATIONS OF PRINCIPLE COMPONENT ANALYSIS(PCA)

PThis project implements Principal Component Analysis (PCA) from first principles, without using machine learning libraries such as scikit-learn. The primary objective is to understand how PCA works mathematically.

Instead of calling built-in PCA functions, all computations are implemented manually using basic Python and linear algebra operations. These include mean centering, covariance matrix computation, eigenvector estimation using power iteration, eigenvalue calculation, deflation for multiple components, projection to lower dimensions, and reconstruction error analysis.

The implementation is evaluated on both a toy dataset and the Iris dataset to demonstrate correctness, numerical stability, and practical behavior.

The motivation behind this project is to study PCA as a linear algebra and optimization problem, and to gain insight into:

1. how variance is distributed across dimensions,

2. why eigenvectors define principal directions,

3. how dimensionality reduction affects information retention,and how PCA behaves numerically when implemented from scratch.

### FORMULATION
## 1. Mean of the Dataset

Given a dataset with n observations and d features, the mean of each feature is computed as:
                      μj​=n1​i=1∑n​xij​

The mean vector 𝜇 represents the central location of the data.

## 2. Mean Centering

To ensure that PCA captures variance correctly, the data is mean-centered by subtracting the mean from each observation:
                          Xc​=X−μ

Mean centering shifts the data so that it is centered around the origin. This step is necessary because covariance depends on deviations from the mean.

## 3. Covariance Matrix

The covariance matrix measures how features vary together. It is computed as: 
                      Σ=n−11​XcT​Xc​

Each element Σij ​represents the covariance between feature i and j
High covariance indicates strong linear dependence between features.

## 4. Eigenvalue Problem

PCA identifies directions along which the variance of the data is maximized. These directions are obtained by solving the eigenvalue equation:

                          Σv=λv
v is an eigenvector (principal direction)

𝜆 is the corresponding eigenvalue (amount of variance)

Eigenvectors with larger eigenvalues represent more important directions.

## 5. Power Iteration Method

To compute the dominant eigenvector, the power iteration algorithm is used. Starting with an initial vector,the method repeatedly applies:	​

                                vk+1​=∥Σvk​∥Σvk​​
This process converges to the eigenvector corresponding to the largest eigenvalue, assuming the matrix has a dominant eigenvalue.

## 6. Deflation for Second Principal Component

After computing the first eigenvector, the covariance matrix is deflated to remove its effect:
                                   Σ′=Σ−λ1​v1​v1T​
 Power iteration is then applied again on Σ′ to obtain the second principal component. 
 This ensures orthogonality between principal directions. This ensures orthogonality between principal directions.

## 7. Projection onto Lower-Dimensional Space

The original data is projected onto the principal components using:
                   Z=Xc​W
where:
W contains the selected eigenvectors
Z is the lower-dimensional representation of the data

In this project the data is projected from 3D to 2D using the first two principal components.

### 8. Geometric Interpretation

Principal components are orthogonal directions

Each component captures maximum remaining variance

Projection preserves structure while reducing dimensionality

## EXPERIMENTAL ANALYSIS

# Toy Dataset

A small three-dimensional dataset was used to study PCA behavior.

Computed eigenvalues:

[450.29, 3.08]

The large gap between eigenvalues indicates that most of the variance lies along a single dominant direction. This confirms that the data is effectively low-dimensional.
After projecting the data onto two principal components, reconstruction was performed and evaluated using mean squared error:
Reconstruction MSE = 0.73
The low reconstruction error shows that dimensionality reduction preserves most of the original information.

## Iris Dataset
PCA was applied to the four numerical features of the Iris dataset without using class labels.
The resulting two-dimensional projection reveals clear structure in the data, demonstrating that PCA successfully captures the dominant variance directions. The behavior closely matches that of standard PCA implementations.

### VALIDATION

To ensure correctness, the custom implementation was validated using NumPy’s eigen decomposition.

Eigenvalues closely match NumPy results

Eigenvector directions were verified using cosine similarity

Explained variance ratios are consistent

This confirms the numerical correctness of the from-scratch implementation.

## KEY TAKEAWAYS

1. PCA is fundamentally a covariance and eigen decomposition problem

2. Eigenvectors define orthogonal directions of maximum variance

3. Deflation is essential for extracting multiple components

4. Significant dimensionality reduction can be achieved with minimal loss

## CONCLUSION

This project presents a complete implementation of Principal Component Analysis from first principles. By avoiding high-level machine learning libraries, it provides a clear and detailed understanding of the mathematical foundations of PCA.

The work strengthens core knowledge in linear algebra, numerical methods, and statistical modeling, and serves as preparation for advanced machine learning and research-oriented study.