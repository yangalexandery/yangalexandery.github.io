import numpy as np
import matplotlib.pyplot as plt

### code used for generating Gaussian Mixture Model (image 1) ###
p_1 = 0.5
mu_1 = np.array([[5], [5]])
sig_1 = np.array([[3, 0], [0, 3]])

p_2 = 0.5
mu_2 = np.array([[14], [3]])
sig_2 = np.array([[2, 0], [0, 2]])

dim_1_r = []
dim_2_r = []

dim_1_b = []
dim_2_b = []

for i in range(1500):
	if np.random.rand() < 0.5:
		dim_1_r.append(5 + 3 * np.random.randn())
		dim_2_r.append(5 + 3 * np.random.randn())
	else:
		dim_1_b.append(12 + 2 * np.random.randn())
		dim_2_b.append(3  + 2 * np.random.randn())

plt.scatter(dim_1_r, dim_2_r, color='r', s=2)
plt.scatter(dim_1_b, dim_2_b, color='b', s=2)
plt.xlabel("$X_1$")
plt.ylabel("$X_2$")
plt.show()