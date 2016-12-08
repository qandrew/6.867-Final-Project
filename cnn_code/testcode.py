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


# plt.imshow(data, aspect='auto', interpolation='none')
           # extent=extents(x) + extents(y))
# plt.show()


a = [3,4,5,10]
c = [1,2,3,4]
b = [2,3,4,5]
plt.plot(c,a,label='test')
plt.plot(b,'go')
plt.legend(loc=2)
plt.savefig(str(a[0])+'py.png')
plt.show()

# np.savetxt('featureVector/test.txt',data)

