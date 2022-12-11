import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TKAgg')

alpha = 1e-6
xs = list(range(0, 301))
ys = [1.0 + alpha * x**2 for x in xs]

fig, ax = plt.subplots()

ax.plot(xs, ys)
ax.set(xlabel='radial distance', ylabel='correction factor')
ax.set_ylim(bottom=0.9)
ax.grid()
plt.show()