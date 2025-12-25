from scipy.stats import qmc
import numpy as np
from time import time
from components.handling.logger import logging
import random
import os


from components.model.wsqmodel_functions import solver
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
        x,y,h = solver(params,zvals)
        Hvals_,xvals_,yvals_ = H0_*h,x,y
        for i,Hval in enumerate(Hvals_):
            train_vals.append([H0_,omega_m_,y0_,zvals[i],Hval,xvals_[i],yvals_[i]])
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
        
        real_params = self.data[index,:-3]
        real_out = self.data[index,-3:]
        

        inbet_points = np.random.uniform(low = [40,0,0,0],high = [100,1,1,3],size = (4,)).astype(np.float32)
        return {
            'real_x' : torch.tensor(real_params,requires_grad=True),
            'real_y' : torch.tensor(real_out),
            # 'inbet_x' : torch.tensor(inbet_points,requires_grad=True)
        }



def calc_phys_loss(model,inputs):
    # H0_,omega_m_,y0_,z_ = inputs
    
    H0_ = inputs[:,0]

    y0_ = inputs[:,-2]
    z_ = inputs[:,-1]
    predictions = model(inputs)
    H_preds = predictions[:,0]

    x_preds = predictions[:,1]
    y_preds = predictions[:,2]
    h_preds = H_preds/H0_
    dH_dz_ = torch.autograd.grad(h_preds,inputs,grad_outputs=torch.ones_like(h_preds), create_graph=True)[0][:,-1]
    dx_dz = torch.autograd.grad(x_preds,inputs,grad_outputs=torch.ones_like(x_preds), create_graph=True)[0][:,-1]
    dy_dz = torch.autograd.grad(y_preds,inputs,grad_outputs=torch.ones_like(y_preds), create_graph=True)[0][:,-1]



    dH_dz_th = (3/2)*((h_preds**2)+(x_preds**2)-(y_preds**2))/(h_preds*(1+z_))
    dx_dz_th = 3*x_preds/(1+z_) - (torch.sqrt(torch.tensor(3/2))* x_preds*y_preds*(1 - (0.5* (y_preds/y0_)**2))/(h_preds*(1+z_)) )
    dy_dz_th = torch.sqrt(torch.tensor(3/2))* y_preds*y_preds*(1 - (0.5* (y_preds/y0_)**2))/(h_preds*(1+z_)) 

    return torch.mean((dH_dz_ - dH_dz_th)**2) + torch.mean((dx_dz - dx_dz_th)**2) + torch.mean((dy_dz-dy_dz_th)**2)


def train_model(datapath,n_epochs = 50):

    device = torch.device("cuda")
    dataset = CustomDataset(datapath)

    loader = DataLoader(dataset, batch_size = 100, shuffle= True)

    model = PINN_WSQ(activation_function=nn.Tanh).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(n_epochs):
        loss_epoch = 0

        for batch in loader:
            train_x = batch['real_x'].to(device)
            train_y = batch['real_y'].to(device)
            # inbet_x = batch['inbet_x'].to(device)

            H_train,x_train,y_train = train_y[:,0],train_y[:,1],train_y[:,2]

            preds = model(train_x)
            H_pred,x_pred,y_pred = preds[:,0],preds[:,1],preds[:,2]
            # loss_data = torch.mean((H_train-H_pred)**2) + torch.mean((x_train-x_pred)**2) + torch.mean((y_train-y_pred)**2)
            loss_data = nn.MSELoss()(preds,train_y)

            # print(inbet_x)
            # loss_phy=0
            loss_phy = calc_phys_loss(model,train_x)
            # tot_loss = loss_data
            tot_loss = (loss_data + 0.1*loss_phy)*1e-5

            optimizer.zero_grad()
            tot_loss.backward()
            optimizer.step()

            loss_epoch += tot_loss

        print(f"Epoch {epoch+1}: Loss = {loss_epoch / len(loader):.4f}")
    return model










if __name__ == '__main__':
    savepath = os.path.join(os.getcwd(),"outputs","gen_data.txt")
    param_inputs = gen_params()
    print(param_inputs)
    # # print(sum(((omega_m + y0**2)<1).astype(int)) == n_samples)

    # start = time()
    Hdata = gen_solver(param_inputs,savepath)
    # end = time()
    # print(f'Hlen = {len(Hdata)}')
    # print(end-start)
    # print(Hdata[0])
    device = torch.device("cuda")
    model = train_model(savepath,20)
    print(model(torch.tensor((70,0.3,0.822,0)).to(device)))