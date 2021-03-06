# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

greyhounds = 500
labs = 500

grep_height = 28 + 4 * np.random.randn(greyhounds)
lab_height = 24 + 4 * np.random.randn(labs)

plt.hist([grep_height,lab_height],stacked=True,color=['r','b'])
plt.show()
