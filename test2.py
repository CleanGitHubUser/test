from scipy.stats import gamma
import numpy as np
import math

nU = 1000
k = 30
mean_y = 3
shape_para = 1
a = shape_para
a_prime = shape_para
b_prime = a / math.sqrt(mean_y / k)
c = shape_para
c_prime = shape_para
d_prime = c / math.sqrt(mean_y / k)

rng = np.random.default_rng()
ksi = rng.uniform(0.15, 0.85, nU)

ksi = gamma.ppf(ksi, a_prime, b_prime)

ksi = ksi.reshape(nU, 1)

# Theta = np.tile(ksi, (1, k))

# Theta.shape

np.apply_along_axis(
    gamma.ppf,
    0,
    ksi,
    q=rng.uniform(0.15, 0.85, k),

)

Beta = np.empty(size = (nI, k))