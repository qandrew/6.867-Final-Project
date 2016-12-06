# A random file to test out stuff

import numpy as np
import matplotlib.pyplot as plt

def extents(f):
  delta = f[1] - f[0]
  return [f[0] - delta/2, f[-1] + delta/2]

x = np.linspace(-100, -10, 10)
y = np.array([-8, -3.0])
data = np.random.randn(y.size,x.size)
print data.shape


plt.imshow(data, aspect='auto', interpolation='none')
           # extent=extents(x) + extents(y))
plt.show()
# plt.savefig('py.png')