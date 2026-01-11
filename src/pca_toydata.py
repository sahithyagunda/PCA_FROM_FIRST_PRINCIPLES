import matplotlib.pyplot as plt
import csv

# Original 3D data
data = [
    [78, 72, 80],
    [65, 60, 58],
    [90, 88, 92],
    [55, 50, 52],
    [82, 79, 85],
    [70, 68, 72],
    [60, 62, 61],
    [88, 85, 90],
    [73, 70, 75],
    [68, 65, 70]
]

# ---- Read 2D PCA projections ----
Z_2D = []
with open('z_2d.csv', 'r') as f:
    reader = csv.reader(f)
    next(reader)  # skip header
    for row in reader:
        Z_2D.append([float(row[0]), float(row[1])])

# ---- Plot ----
fig = plt.figure(figsize=(12,5))

# 3D original data
ax1 = fig.add_subplot(121, projection='3d')
for point in data:
    ax1.scatter(point[0], point[1], point[2], c='b')
ax1.set_xlabel('Math')
ax1.set_ylabel('Physics')
ax1.set_zlabel('Chemistry')
ax1.set_title('Original 3D Data')

# 2D PCA projection
ax2 = fig.add_subplot(122)
for point in Z_2D:
    ax2.scatter(point[0], point[1], c='r')

ax2.set_xlabel('PC1')
ax2.set_ylabel('PC2')
ax2.set_title('2D PCA Projection')
ax2.grid(True)

plt.tight_layout()
plt.show()

# save the plot
fig.savefig('pca_projection.png')
