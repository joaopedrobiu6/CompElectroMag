import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Define the function to fit
def func(x, a, b):
    return a * x**3 + b 

N = [64, 128, 256, 512, 1024]
time_colab = [0.2460489273071289, 1.5473787784576416, 11.57834529876709, 33.877668380737305, 244.09256601333618]

time_vscode = [0.033071279525756836, 0.20355606079101562, 1.2105348110198975, 9.677772998809814, 75.88517618179321, 816.3999180793762]
time2 = [0.09738492965698242, 0.4203982353210449, 3.260528087615967, 25.50840711593628, 197.70876908302307]
total_time = np.sum(np.array(time_vscode))


# Fit the data
popt, pcov = curve_fit(func, N, time2)

# for N = 1024
time1024 = func(1024, *popt)
time2048 = func(2048, *popt)
print(f"time 1024: {time1024}")
print(f"time 2048: {time2048/60}")
print((total_time)/60)

error_time = total_time*2
print(f"error time: {error_time/60}")