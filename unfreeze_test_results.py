import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator

def read_file(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        data = file.readlines()
        lines = [list(map(float, line.strip('\n').strip('\ufeff').split(','))) for line in data]
    return lines

def plot(x, y):
    line1 = plt.plot(x, y[0], '.-', label='text', alpha=0.8)
    line2 = plt.plot(x, y[1], 'x-', label='struc', alpha=0.8)
    line3 = plt.plot(x, y[2], '*-', label='combine', alpha=0.8)
    

x = [0,6,12,18,20,22,24]
y = read_file('./unfreeze_test_results.csv')
plot(x, y)
plt.grid(alpha=0.8)
x_major_locator=MultipleLocator(3)
ax=plt.gca()
ax.xaxis.set_major_locator(x_major_locator)
plt.xlim(-0.5, 24.5)
plt.axhline(0.563, alpha=0.4, color='#2ca02c', linestyle="--")
plt.axhline(0.539, alpha=0.4, color='#ff7f0e', linestyle="--")
plt.axhline(0.521, alpha=0.4, color='#1f77b4', linestyle="--")
plt.ylabel('MRR')
plt.xlabel('Layers')
plt.legend()
plt.savefig('unfreeze_contrast_test.eps')