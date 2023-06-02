"""
A empty example
===============

TODO: this is just a placeholder for now
"""
# %%
import hazardous
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-3, 3, 100)
plt.plot(x, x ** 2, label="$x^2$")
plt.plot(x, x ** 3, label="$x^3$")
_ = plt.legend()
