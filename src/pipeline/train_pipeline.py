from scipy.stats import qmc
import numpy as np
from components.model.wsqmodel_functions import Hubble_solver,xy_solver
from time import time
from components.handling.logger import logging
import random
import os

import torch
from torch.utils.data import Dataset
'''
Code to randomly sample points within the parameter space at different redshifts. This will be used by the ode solver
to then generate points for model training.

'''

n_samples = 6000 # No. of parameter combinations to sample


def gen_params(n_samples = 6000):
    sampler = qmc.LatinHypercube(d=3)
    sample = sampler.random(n=n_samples)

    logging.info(f'Generated {n_samples} samples from parameter space')

    r'''
    There is a problem here though. $\Omega_m$ and $y_0$ need to follow the constraint :  omega_m + y0^2 <= 1
    We can go around this using a slightly clever approach.

    We use the first quadrant of a unit radius disk to emulate omega_m and y0 values.


    '''


    lower_bounds = [40,0,0]               # H0, r, theta, z
    upper_bounds = [100,1,np.pi*0.5]  #sample theta in the first quadrant (so as to ensure omega_m and y0 are positive)


    param_inputs = qmc.scale(sample,lower_bounds,upper_bounds)

    r_ = sample[:,1]  
    theta = sample[:,2]

    omega_m = (r_*np.cos(theta))**2
    y0 = r_*np.sin(theta)

    param_inputs = np.column_stack([param_inputs[:,0],omega_m,y0])
    return param_inputs

def gen_solver(param_set,savepath,num_z = 1000):
    zvals = np.linspace(0,3,num_z)
    train_vals= []
    for params in param_set:
        H0_,omega_m_,y0_ = params
        Hvals_ = Hubble_solver(params,zvals)
        for i,Hval in enumerate(Hvals_):
            train_vals.append([H0_,omega_m_,y0_,zvals[i],Hval])
    random.shuffle(train_vals)
    np.savetxt(savepath,train_vals)

    logging.info(f'Saved generated data to {savepath}')
    return np.asarray(train_vals)

class CustomDataset(Dataset):
    def __init__(self,datapath):
        self.data = np.loadtxt(datapath).astype(np.float32)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        
        real_params = self.data[index,:-1]
        real_out = self.data[index,-1]
        

        inbet_points = np.random.uniform(low = [40,0,0,0],high = [100,1,1,3],size = (4,)).astype(np.float32)
        return {
            'real_x' : torch.tensor(real_params),
            'real_y' : torch.tensor(real_out),
            'inbet_x' : torch.tensor(inbet_points,requires_grad=True)
        }


















if __name__ == '__main__':
    savepath = os.path.join(os.getcwd(),"outputs","gen_data.txt")
    param_inputs = gen_params(n_samples)
    print(param_inputs)
    # print(sum(((omega_m + y0**2)<1).astype(int)) == n_samples)

    start = time()
    Hdata = gen_solver(param_inputs,savepath)
    end = time()
    print(f'Hlen = {len(Hdata)}')
    print(end-start)
    print(Hdata[0])