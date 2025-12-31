import numpy as np
from scipy.integrate import solve_ivp

import torch

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


def wsq_H_f(inputs,y1):
    x,y,la,H = y1

    z = inputs[:,-1]
    dx_dz_ = torch.autograd.grad(x,inputs, grad_outputs= torch.ones_like(x), create_graph=True)[0][:,-1]
    dy_dz_ = torch.autograd.grad(y,inputs, grad_outputs= torch.ones_like(y), create_graph=True)[0][:,-1]
    dla_dz_ = torch.autograd.grad(la,inputs, grad_outputs= torch.ones_like(la), create_graph=True)[0][:,-1]
    dH_dz_ = torch.autograd.grad(H,inputs, grad_outputs= torch.ones_like(H), create_graph=True)[0][:,-1]


    dH_dz_th = -(3/2)*H*(1 + x**2 - y**2)/(1+z)
    dx_dz_th = (3*x - (np.sqrt(3/2)*la*y**2) - ((3/2)*x*(1 + x**2 - y**2)))/(1+z)
    dy_dz_th  = ((np.sqrt(3/2)*la*y*x) - ((3/2)*y*(1 + x**2 - y**2)))/(1+z)
    dla_dz_th = -np.sqrt(6)*x*la*(1-la)/(1+z)

    return (dH_dz_- dH_dz_th)**2 + (dx_dz_- dx_dz_th)**2 + (dy_dz_- dy_dz_th)**2 + (dla_dz_- dla_dz_th)**2

if __name__ =='__main__':
    x,y,la,H = solver((69,0.3,0.82),[0,0.1,0.2])
    print(H)