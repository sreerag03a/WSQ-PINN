from scipy.stats import qmc
import numpy as np
from time import time
from components.handling.logger import logging
import random
import os


from components.model.wsqmodel_functions import solver, wsq_H_f
from components.model.model import PINN_WSQ

import torch
from torch.utils.data import Dataset,DataLoader
import torch.nn as nn
from torch import optim
'''
Code to randomly sample points within the parameter space at different redshifts. This will be used by the ode solver
to then generate points for model training.

'''

n_samples = 6000 # No. of parameter combinations to sample


def gen_params(n_samples = 600):
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

def gen_solver(param_set,savepath,num_z = 100):
    zvals = np.linspace(0,3,num_z)
    train_vals = []
    for params in param_set:
        H0_,omega_m_,y0_ = params
        x,y,la,H = solver(params,zvals)

        for i,Hval in enumerate(H):
            train_vals.append([H0_,omega_m_,y0_,zvals[i],x[i],y[i],la[i],Hval])
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
        
        real_params = self.data[index,:-4]
        real_out = self.data[index,-4:]
        

        inbet_points = np.random.uniform(low = [40,0,0,0],high = [100,1,1,3],size = (4,)).astype(np.float32) #Collocation points
        return {
            'real_x' : torch.tensor(real_params),
            'real_y' : torch.tensor(real_out),
            'inbet_x' : torch.tensor(inbet_points,requires_grad=True)
        }



def calc_phys_loss(model,inputs):
    
    predictions = model(inputs)
    

    x_preds = predictions[:,0]
    y_preds = predictions[:,1]
    lambda_preds = predictions[:,2]
    H_preds = predictions[:,3]

    return torch.mean(wsq_H_f(inputs,(x_preds,y_preds,lambda_preds,H_preds)))

def train_model(datapath,n_epochs = 50):

    device = torch.device("cuda")
    dataset = CustomDataset(datapath)

    loader = DataLoader(dataset, batch_size = 500, shuffle= True)

    model = PINN_WSQ(activation_function=nn.Tanh).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-2)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=15)

    for epoch in range(n_epochs):
        loss_epoch = 0

        for batch in loader:
            train_x = batch['real_x'].to(device)
            train_y = batch['real_y'].to(device)
            inbet_x = batch['inbet_x'].to(device)


            preds = model(train_x)

            loss_data = nn.MSELoss()(preds,train_y)

            loss_phy = calc_phys_loss(model,inbet_x)


            # epoch_progress = min(epoch / 40.0, 1.0)
            alpha = 1
            tot_loss = loss_data + (alpha * loss_phy)

            optimizer.zero_grad()
            tot_loss.backward()

            optimizer.step()
            

            loss_epoch += tot_loss
        scheduler.step(loss_epoch)

        print(f"Epoch {epoch+1}: Loss = {loss_epoch / len(loader):.4f}, learning rate : {scheduler.get_last_lr()}")
    return model










if __name__ == '__main__':
    savepath = os.path.join(os.getcwd(),"outputs","gen_data.txt")
    param_inputs = gen_params()
    # start = time()
    gen_solver(param_inputs,savepath)
    device = torch.device("cuda")
    model = train_model(savepath,1000)
    print(model(torch.tensor((70,0.3,0.822,0)).to(device)))