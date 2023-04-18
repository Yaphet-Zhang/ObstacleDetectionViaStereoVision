# 一次元カルマンフィルタ: https://inzkyk.xyz/kalman_filter/one_dimensional_kalman_filters/#section:4.1

import numpy as np
from numpy.random import randn
import matplotlib.pyplot as plt
import filterpy.stats as stats

# N(10, 1): mu=10,sigma=1 (1 time)
plt.figure(1)
stats.plot_gaussian_pdf(mean=10., variance=1., xlim=(4, 16), ylim=(0, .5))
plt.grid()


# N(10, 1): mu=10,sigma=1 (500 times)
plt.figure(2)
xs = range(500)
ys = randn(500)*1. + 10.
plt.plot(xs, ys)
print(f'出力の平均は {np.mean(ys):.3f}')


plt.show()
