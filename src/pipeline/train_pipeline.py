from scipy.stats import qmc
import numpy as np

'''
Code to randomly sample points within the parameter space at different redshifts. This will be used by the ode solver
to then generate points for model training.

'''

n_samples = 6000 # No. of parameter combinations to sample
sampler = qmc.LatinHypercube(d=4)
sample = sampler.random(n=n_samples)


# lower_bounds = [40,0,0,0]               # H0, omega_m, y0, z
# upper_bounds = [100,1,1,3]

'''
There is a problem here though. $\Omega_m$ and $y_0$ need to follow the constraint :  omega_m^2 + y0^2 <= 1
We can go around this using a slightly clever approach.


'''


lower_bounds = [40,0,0,0]               # H0, r, theta, z
upper_bounds = [100,1,np.pi*0.5,3]  #sample theta in the first quadrant (so as to ensure omega_m and y0 are positive)


param_inputs = qmc.scale(sample,lower_bounds,upper_bounds)

r_ = sample[:,1]
theta = sample[:,2]

omega_m = r_*np.cos(theta)
y0 = r_*np.sin(theta)

param_inputs = np.column_stack([param_inputs[:,0],omega_m,y0,param_inputs[:,3]])



def ode(z,y1):
    pass

if __name__ == '__main__':
    print(param_inputs)
    print(sum(((omega_m**2 + y0**2)<=1).astype(int)) == n_samples)