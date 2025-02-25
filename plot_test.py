import matplotlib.pyplot as plt

plt.figure()
plt.xlim(0, 5)
plt.ylim(0, 5)

for i in range(0, 5):
    plt.axhspan(i, i+.2, facecolor='0.2', alpha=0.5)
    plt.axvspan(i, i+.5, facecolor='red', alpha=0.5)

plt.show()