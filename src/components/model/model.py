import torch
import torch.nn as nn


'''


'''

class PINN_WSQ(nn.Module):

    def __init__(self,activation_function = nn.Tanh,dropout = 0.2):
        super().__init__()
        self.net =  nn.Sequential(
            nn.Linear(1,128),activation_function(), # First layer with input dimension (4,) and tanh activation function
            nn.Linear(128,128), activation_function(), # Second layer with 128 inputs and tanh activation function
            nn.Linear(128,128), activation_function(),
            nn.Linear(128,4) # Outputs x(z), y(z), lambda(z), H(z)
        )


    def forward(self,z,params):
        preds = self.net(z)
        try:
            H0,omega_m0,y0 = params[:,0],params[:,1],params[:,2]
            z = z.squeeze()
            x0 = torch.sqrt(1 - omega_m0 - (y0**2))
            x_,y_,la_,H_ = preds[:,0],preds[:,1],preds[:,2],preds[:,3]
            la0 = 0.5
            x,y,la,H = x0 + z*x_,y0 + z*y_,la0 + z*la_,1.0 + z*H_
        except:
            H0,omega_m0,y0 = params
            x0 = torch.sqrt(1 - omega_m0 - (y0**2))
            x_,y_,la_,H_ = preds[0],preds[1],preds[2],preds[3]
            la0 = 0.5
            x,y,la,H = x0 + z*x0*x_,y0 + z*y0*y_,la0 + z*la0*la_, 1.0 + z*H_

        return x,y,la,H
    