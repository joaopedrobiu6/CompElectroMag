import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Define the function to fit
def func(x, a, b):
    return a * x**3 + b 

N = [64, 128, 256, 512]
time = [0.1567838191986084, 0.9454758167266846, 8.802689552307129, 69.34697079658508]

time2 = [0.07398509979248047, 0.3604722023010254, 2.421337842941284, 20.19579792022705]
time3 = [0.021584033966064453, 0.1635119915008545, 1.2764191627502441, 9.937772035598755]
total_time = np.sum(np.array(time3))
# Fit the data
popt, pcov = curve_fit(func, N, time3)

# for N = 1024
time1024 = func(1024, *popt)
time2048 = func(2048, *popt)
print(time1024)
print((total_time + time1024 + time2048)/60)