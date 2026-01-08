import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import torch
from components.handling.logger import logging

def wsq_ode(z,y1):
    x,y,la,H = y1
    
    dx_dz = (3*x - (np.sqrt(3/2)*la*y**2) - ((3/2)*x*(1 + x**2 - y**2)))/(1+z)

    dy_dz  = ((np.sqrt(3/2)*la*y*x) - ((3/2)*y*(1 + x**2 - y**2)))/(1+z)

    dla_dz = -np.sqrt(6)*x*la*(1-la)/(1+z)

    dH_dz = (3/2)*H*(1 + x**2 - y**2)/(1+z)


    return [dx_dz,dy_dz, dla_dz,dH_dz]



def solver(params,z,zrange = (0,3)):
    H0,omega_m_param,y0_param= params
    x0 = np.sqrt(1-omega_m_param-(y0_param**2))
    initial_conditions = [x0,y0_param,0.5,H0]
    solution = solve_ivp(wsq_ode,zrange,initial_conditions,t_eval=z)
    x,y,la,H = solution.y
    return x,y,la,H


def wsq_H_f(z,y1):
    x,y,la,H = y1
    x = x.reshape(-1,1)
    y = y.reshape(-1,1)
    la = la.reshape(-1,1)
    H = H.reshape(-1,1)
    dx_dz_ = torch.autograd.grad(x,z, grad_outputs= torch.ones_like(x), create_graph=True)[0]
    dy_dz_ = torch.autograd.grad(y,z, grad_outputs= torch.ones_like(y), create_graph=True)[0]
    dla_dz_ = torch.autograd.grad(la,z, grad_outputs= torch.ones_like(la), create_graph=True)[0]
    dH_dz_ = torch.autograd.grad(H,z, grad_outputs= torch.ones_like(H), create_graph=True)[0]

    # print('dhdz',dH_dz_)
    dx_dz_th = (3*x - (np.sqrt(3/2)*la*y**2) - ((3/2)*x*(1 + x**2 - y**2)))/(1+z)
    dy_dz_th  = ((np.sqrt(3/2)*la*y*x) - ((3/2)*y*(1 + x**2 - y**2)))/(1+z)
    dla_dz_th = -np.sqrt(6)*x*la*(1-la)/(1+z)
    dH_dz_th = (3/2)*H*(1 + x**2 - y**2)/((1+z))
    return torch.nn.MSELoss()(dx_dz_,dx_dz_th) + torch.nn.MSELoss()(dy_dz_,dy_dz_th) + torch.nn.MSELoss()(dla_dz_,dla_dz_th) + torch.nn.MSELoss()(dH_dz_,dH_dz_th)

if __name__ =='__main__':
    zvals = np.linspace(0,150,10000)
    x,y,la,H = solver((69,0.3,0.82),zvals,(0,150))
    plt.plot(zvals,H)
    plt.show()
    ode = x**2 + y**2
    plt.plot(zvals,ode,label = r'$\Omega_\text{de}$')
    plt.plot(zvals,1-ode, label = r'$\Omega_\text{m}$')
    plt.legend()
    plt.show()

    plt.plot(zvals,la)
    plt.show()