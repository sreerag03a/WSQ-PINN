import numpy as np
from scipy.integrate import solve_ivp

import torch

def wsq_ode(z,y1,y0_param):
    x,y,h = y1
    l = y0_param**2
    
    dx_dz = (3*x/(1+z)) - ((np.sqrt(3/2)*(y**2)*(1-((1/(2*l))*(y**2)))/(h*(1+z))))

    dy_dz = np.sqrt(3/2)*x*y*(1-((1/(2*l))*(y**2)))/(h*(1+z))
    
    dh_dz = (3/2)*((h**2)+(x**2)-(y**2))/(h*(1+z))

    return [dx_dz,dy_dz,dh_dz]



def solver(params,z,zrange = (0,3)):
    H0,omega_m_param,y0_param= params
    x0 = np.sqrt(1-omega_m_param-(y0_param**2))
    initial_conditions = [x0,y0_param,1]
    solution = solve_ivp(wsq_ode,zrange,initial_conditions,t_eval=z, args = (y0_param,))
    x,y,h = solution.y
    return x,y,h
