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


def calc_phys_loss(model,inputs):
    
    # x_preds,y_preds,lambda_preds,H_preds = model(inputs)
    predictions = model(inputs)

    x_preds = predictions[:,0]
    y_preds = predictions[:,1]
    lambda_preds = predictions[:,2]
    H_preds = predictions[:,3]

    return torch.mean(wsq_H_f(inputs,(x_preds,y_preds,lambda_preds,H_preds)))

def train_model(train_params = 600,n_epochs = 500, learning_r = 1e-3,batch_size = 200,colloc_times = 500):
    
    device = torch.device("cuda")
    # dataset = CustomDataset(train_params)

    # loader = DataLoader(dataset, batch_size = 600, shuffle= True)
    paramset = torch.tensor(gen_params(train_params), dtype=torch.float32).to(device)

    z0 = torch.zeros(paramset.shape[0], 1).to(device)
    train_x = torch.cat([paramset,z0], dim=1)

    del(z0)
    H0_,omega_m_,y0_ = paramset[:,0:1],paramset[:,1:2],paramset[:,2:3]
    x0_ = torch.sqrt(1 - omega_m_ - (y0_**2)).reshape(-1,1)
    la0 = torch.full_like(x0_, 0.5)
    
    train_y = torch.cat((x0_,y0_,la0,H0_),dim=1).to(device)
    
    # print(train_y.shape)
    
    model = PINN_WSQ(activation_function=nn.Tanh).to(device)
    optimizer = optim.Adam(model.parameters(), lr = learning_r)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    batches_n = int(train_params/batch_size)
    for epoch in range(n_epochs):

        loss_epoch = 0
        for i in range(batches_n):
            start,end = i*batch_size,(i+1)*batch_size
            x_train,y_train = train_x[start:end,:],train_y[start:end,:]
            params_ = paramset[start:end,:]
            z_colloc = torch.linspace(0.0, 3.0, colloc_times,dtype=torch.float32).to(device)
            z_colloc = z_colloc.repeat(batch_size).reshape(-1,1)
            colloc_x = params_.repeat_interleave(colloc_times,dim=0)

            colloc_x =  torch.cat([colloc_x,z_colloc], dim=1)

            

            colloc_x.requires_grad = True
            

            preds = model(x_train)
            tot_loss = nn.MSELoss()(preds,y_train)
            # tot_loss=0
            loss_phy = calc_phys_loss(model,colloc_x) # work in progress


            epoch_progress = min(epoch / 40.0, 1.0)
            alpha = epoch_progress*0.1
            tot_loss += (alpha * loss_phy)
            

            optimizer.zero_grad()
            tot_loss.backward()

            optimizer.step()
                

            loss_epoch += tot_loss
        if (epoch+1)%500 == 0:
            optimizer.param_groups[0]['lr'] *= 0.5
        # scheduler.step(loss_epoch)
        logging.info(f"Epoch {epoch+1}: Loss = {loss_epoch:.4f}, learning rate : {optimizer.param_groups[0]['lr']}")
        print(f"Epoch {epoch+1}: Loss = {loss_epoch:.4f}, learning rate : {optimizer.param_groups[0]['lr']}")
    return model










if __name__ == '__main__':
    # savepath = os.path.join(os.getcwd(),"outputs","gen_data.txt")
    # param_inputs = gen_params()
    # # start = time()
    # gen_solver(param_inputs,savepath)
    device = torch.device("cuda")
    model = train_model(train_params=100,n_epochs=5000,learning_r=1e-3,batch_size=10,colloc_times=100)


    print(f'Model pred : {model(torch.tensor(([70,0.3,0.822,0])).to(device))}')
    print(f'Actual : {solver((70,0.3,0.822),[0])}')
    print(f'Model pred : {model(torch.tensor(([70,0.3,0.822,3])).to(device))}')
    print(f'Actual : {solver((70,0.3,0.822),[3])}')