#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt("/Users/pranavjay/Downloads/numpy2D.csv", delimiter = ',')
plt.imshow(data)
