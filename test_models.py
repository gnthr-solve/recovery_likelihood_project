
import torch
import torch.linalg as tla

from ebm import EnergyModel
from helper_tools import quadratic_form_batch, check_nan

"""
Concrete Energy Models for Experiments.

These are the concrete subclasses for the EnergyModel base class.
They define the parameters to be trained in the __init__ and implement the energy method.
In those cases where the gradient of the energy is analytically available, like in the MultivariateGaussianModel,
the models can override the energy_grad method.
Overriding the avg_param_grad method is also possible, but as this method normally computes the average gradients of a batch,

"""


"""
Multivariate Gaussian Model
-------------------------------------------------------------------------------------------------------------------------------------------
"""
class MultivariateGaussianModel(EnergyModel):

    def __init__(self, mu_0: torch.Tensor, Sigma_0: torch.Tensor):
        super().__init__()
        self.params['mu'] = mu_0.clone()
        self.params['Sigma'] = Sigma_0.clone()
        self.Sigma_inv = (lambda : torch.inverse(self.params['Sigma']))


    def kernel(self, x: torch.Tensor):

        # x can be of shape (d,) or (n, d)
        x = torch.unsqueeze(x, dim=0) if x.dim() == 1 else x # shape (1, d) or (n, d)

        # Reshape mu and Sigma_inv to support broadcasting
        mu = self.params['mu'].unsqueeze(0)  # Shape (1, d)
        Sigma_inv = self.Sigma_inv()

        diff = x - mu # shape (1, d) or (n, d)
        exponent = -0.5 * tla.vecdot(
            torch.matmul(diff, Sigma_inv),
            diff,
        )

        kernel_value = torch.exp(exponent)
    
        return kernel_value


    def energy(self, x: torch.Tensor):

        energy = -torch.log(self.kernel(x))
        #energy = npl.multi_dot([(x - self.mu), self.Sigma_inv, (x - self.mu)])/2

        return energy
    

    def energy_grad(self, x: torch.Tensor):

        mu = self.params['mu']  # Shape (1, d)
        Sigma_inv = self.Sigma_inv()

        grad = tla.multi_dot([(x - mu), Sigma_inv])

        return grad



"""
Simple Linear Model
-------------------------------------------------------------------------------------------------------------------------------------------
"""
class SimpleLinear(EnergyModel):

    def __init__(self, theta_0: torch.Tensor, C_0: torch.Tensor):
        super().__init__()
        self.params['theta'] = theta_0.clone()
        self.params['C'] = C_0.clone()

    
    def energy(self, x: torch.Tensor):

        theta = self.params['theta']
        C = self.params['C']
        theta = theta.unsqueeze(0)

        #result = theta @ x + C
        result = tla.vecdot(theta, x, dim = 1) + C

        return result



"""
Simple Linear Model
-------------------------------------------------------------------------------------------------------------------------------------------
"""
class VisibleBoltzmann(EnergyModel):

    def __init__(self, W_0: torch.Tensor):
        super().__init__()
        self.params['W'] = W_0.clone()

    
    def energy(self, x: torch.Tensor):

        W = self.params['W']

        x = torch.unsqueeze(x, dim=0) if x.dim() == 1 else x # shape (1, d) or (n, d)

        result = quadratic_form_batch(x, W)

        return -result/2




"""
Testing Functions
-------------------------------------------------------------------------------------------------------------------------------------------
"""
def linear_test():

    theta = torch.tensor([2, 3], dtype = torch.float32)
    C = torch.tensor(5, dtype = torch.float32)
    
    model = SimpleLinear(theta_0 = theta, C_0 = C)

    x = torch.tensor(
        [[2, 1],
         [1, 1],
         [3, 1]], 
        dtype = torch.float32
    )
    #x = torch.tensor([2, 1], dtype = torch.float32)

    print("Model execution result: \n", model(x))
    #print("Model execution result: \n", model.energy(x))
    print("Model grad w.r.t. input: \n", model.energy_grad(x))
    print("Model mean parameter grad: \n", model.avg_param_grad(x))



def mvGaussian_test():

    mu = torch.tensor([2, 2], dtype = torch.float32)
    Sigma = torch.tensor(
        [[2, 0],
         [0, 1],],
        dtype=torch.float32,
    )
    
    model = MultivariateGaussianModel(mu_0 = mu, Sigma_0 = Sigma)

    x = torch.tensor(
        [[2, 1],
         [1, 1],
         [3, 1]], 
        dtype = torch.float32
    )
    x = torch.tensor([2, 1], dtype = torch.float32)

    print("Model execution result: \n", model(x))
    #print("Model execution result: \n", model.energy(x))
    print("Model grad w.r.t. input: \n", model.energy_grad(x))
    print("Model mean parameter grad: \n", model.avg_param_grad(x))







if __name__=="__main__":

    #linear_test()
    mvGaussian_test()