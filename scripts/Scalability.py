from pyclassify.numba_distance_module import distance_numba
from pyclassify.utils import distance_numpy
from time import time
import numpy as np
import matplotlib.pyplot as plt

a = np.random.rand(80)
b = np.random.rand(80)

t_numpy = []
t_numba = []
for i in range(29):
    n=2**i
    a = np.random.rand(n)
    b = np.random.rand(n)

    t_s = time()
    distance_numpy(a, b)
    t_e = time()
    t_numpy.append(t_e-t_s)

    t_s = time()
    distance_numba(a, b)
    t_e = time()
    t_numba.append(t_e-t_s)

x = [2**i for i in range(29)]
plt.figure()
plt.loglog(x, t_numpy, '-o' ,x, t_numba, '-o')
plt.legend(['numpy', 'numba'])
plt.xlabel("Vector dimension")
plt.ylabel("Time [s]")
plt.savefig("/home/gaspare/Desktop/PhD/devtools_scicomp_project_2025/logs/scalability.png")
plt.show()