import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

#  --------------------------------------------------------------
#  Time step 1D wave equation using two time-levels f0 & f1
#  --------------------------------------------------------------
#  function [omega, s1, s2] = Wave1D(a, time, nx)
# 
#  Arguments:
#     a     = the length of the interval
#     time  = the total time interval for the simulation
#     nx    = the number of subintervals in the domain (0,a)
#  Returns:
#     omega = the angular frequencies
#     s1    = the complex Fourier transform of data at x = a/5
#     s2    = the complex Fourier transform of data at x = a/2

def Wave1D(a, time, nx):
    f0         = np.random.randn(nx) # Initialize with random numbers
    f0[0]    = 0              # Boundary condition at x = 0
    f0[nx-1]   = 0              # Boundary condition at x = a

    f1         = np.random.randn(nx) # Initialize with random numbers
    f1[0]    = 0              # Boundary condition at x = 0
    f1[nx-1]   = 0              # Boundary condition at x = a

    dx         = a/nx           # The cell size 
    d2tmax     = 1.9*dx         # The time step must satisfy
                                 # 2*dt < 2*dx for stability

    ntime = np.int64(time/d2tmax + 1)  # The number of time steps
    dt = time/(2*ntime)             # The time step

    # Initialize the coefficient matrix for updating the solution f

    A = np.zeros((nx,nx))
    for i in range(1,nx-1):
      A[i,i]   = 2*(1-(dt/dx)**2)    # Diagonal entries
      A[i,i+1] = (dt/dx)**2          # Upper diagonal entries
      A[i,i-1] = (dt/dx)**2          # Lower diagonal entries



    sign1 = np.zeros(2*ntime)
    sign2 = np.zeros(2*ntime)

    # Time step and sample the solution
    # Sample location #1 is close to the left boundary
    # Sample location #2 is at the midpoint of the domain
    for itime in range(0,ntime): # Every 'itime' means two time steps 'dt'

      f0               = A@f1 - f0          # Update
      sign1[2*itime-1] = f0[np.int64(1+nx/5)]  # Sample at location #1
      sign2[2*itime-1] = f0[np.int64(1+nx/2)]  # Sample at location #2


      f1               = A@f0 - f1          # Update               
      sign1[2*itime]   = f1[np.int64(1+nx/5)]  # Sample at location #1
      sign2[2*itime]   = f1[np.int64(1+nx/2)]  # Sample at location #2


    # Compute the discrete Fourier transform of 
    # the time-domain signals
    spectr1     = np.fft.fft(sign1) 
    spectr2     = np.fft.fft(sign2)

    # In the MATLAB implementation of the function fft(), 
    # the first half of the output corresponds to positive frequency
    s1 = spectr1[0:ntime] 
    s2 = spectr2[0:ntime]

    # Frequency vector for use with 's1' and 's2'
    omega       = (2*np.pi/time)*np.linspace(0, ntime-1, ntime)
    return omega, s1, s2

omega, s1, s2 = Wave1D(np.pi, 8000, 30)
plt.plot(omega, np.abs(s1), label='s1')
plt.show()
plt.plot(omega, np.abs(s2), label='s2')
plt.show()


