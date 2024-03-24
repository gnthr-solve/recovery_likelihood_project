
import torch
import torch.linalg as tla

#from torch.nn import Softmax

from ebm import EnergyModel
from helper_tools import quadratic_form_batch, check_nan
from timing_decorators import timing_decorator

"""
Concrete Energy Models for Experiments.

These are the concrete subclasses for the EnergyModel base class.
They define the parameters to be trained in the __init__ and implement the energy method.
In those cases where the gradient of the energy is analytically available, like in the MultivariateGaussianModel,
the models can override the energy_grad method.
Overriding the avg_param_grad method is also possible, but as this method normally computes the average gradients of a batch,
one has to keep that extra detail in mind.
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
Mixture of two Gaussians Model
-------------------------------------------------------------------------------------------------------------------------------------------
"""
class SimpleGaussianMixtureModel(EnergyModel):

    def __init__(
            self,
            start_weights: torch.Tensor,
            start_mu_1: torch.Tensor, 
            start_Sigma_1: torch.Tensor, 
            start_mu_2: torch.Tensor, 
            start_Sigma_2: torch.Tensor, 
        ):
        super().__init__()

        self.params['W'] = torch.log(start_weights.clone())

        self.params['mu_1'] = start_mu_1.clone()
        self.params['Sigma_1'] = start_Sigma_1.clone()

        self.params['mu_2'] = start_mu_2.clone()
        self.params['Sigma_2'] = start_Sigma_2.clone()

        self.Sigma_inv = (lambda Sigma: torch.inverse(Sigma))
        self.dim = start_mu_1.shape[-1]

    #@timing_decorator
    def kernel(self, x: torch.Tensor):

        # x can be of shape (d,) or (n, d)
        x = torch.unsqueeze(x, dim=0) if x.dim() == 1 else x # shape (1, d) or (n, d)

        W = self.params['W'].softmax(dim = -1)

        K_1 = self.component_kernel(x = x, component = 1)
        K_2 = self.component_kernel(x = x, component = 2)

        Z_1, Z_2 = self.norm_const()

        kernel_value = W[0]* Z_2 * K_1 + W[1]* Z_1 * K_2
    
        return kernel_value
    

    #@timing_decorator
    def component_kernel(self, x: torch.Tensor, component: int):

        # Reshape mu and Sigma_inv to support broadcasting
        mu = self.params[f'mu_{component}'].unsqueeze(0)  # Shape (1, d)
        Sigma_inv = self.Sigma_inv(self.params[f'Sigma_{component}'])

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
    

    #@timing_decorator
    def norm_const(self):
        
        Z_1 = torch.sqrt((2 * torch.pi)**self.dim * tla.det(self.params['Sigma_1']))
        Z_2 = torch.sqrt((2 * torch.pi)**self.dim * tla.det(self.params['Sigma_2']))
        
        return Z_1, Z_2
        




"""
Univariate Polynomial Model without x**0 coefficient
-------------------------------------------------------------------------------------------------------------------------------------------
"""
class UnivPolynomial(EnergyModel):

    def __init__(self, start_W: torch.Tensor):
        """
        Univariate Polynomial Energy
        Weight associated powers are interpreted like their index i.e. W[i] -> W[i] * x**(i+1)
        """
        super().__init__()
        self.params['W'] = start_W.clone()

    
    def energy(self, x: torch.Tensor):

        W = self.params['W']
        #Squeeze necessary to allow sampler batches of shape (n, 1)
        x = torch.atleast_1d(x).squeeze()

        # Use torchs method to create a matching Vandermonde Matrix
        vander = torch.vander(x, W.shape[0] + 1, increasing = True)
        
        #remove 0 coefficient
        vander = vander[:, 1:]

        result = torch.matmul(vander, W)

        return result
    

    def energy_grad(self, x: torch.Tensor):
        
        W = self.params['W']
        #Squeeze necessary to allow sampler batches of shape (n, 1)
        x = torch.atleast_1d(x).squeeze()

        coeff = torch.arange(W.shape[0], dtype=x.dtype) + 1
        W = W * coeff

        vander = torch.vander(x, W.shape[0], increasing = True)

        grad = torch.matmul(vander, W)
        #grad needs to be unsqueezed, otherwise sampler batch gradient calculations malfunction
        grad = grad.unsqueeze(dim = -1)

        return grad
    
    
    def avg_param_grad(self, x: torch.Tensor):

        W = self.params['W']
        x = x.squeeze()

        vander = torch.vander(x, W.shape[0]+1, increasing = True)
        vander = vander[:, 1:]

        return {'W': torch.sum(vander, dim = 0) / x.shape[0]}
    


"""
Moderated Cosine
-------------------------------------------------------------------------------------------------------------------------------------------
Becomes more multimodal the larger W_cos gets.
"""

class ModeratedCosine(EnergyModel):

    def __init__(self, W_cos: torch.Tensor, mu: torch.Tensor):
        
        self.params = {
            'W_cos': W_cos.clone(),
            'mu': mu.clone(),
        }


    def energy(self, x: torch.Tensor):

        W_cos = self.params['W_cos']
        mu = self.params['mu']

        #Make x a tensor with dim = 2, if mu is scalar and x a batch the x values need to be stacked.
        x = torch.atleast_1d(x)
        x = x.unsqueeze(1) if mu.dim() == 0 else torch.atleast_2d(x)

        diff = x - mu

        cos_term = W_cos * torch.cos(diff)
        log_norm_term = torch.log(torch.norm(diff, p = 2, dim = 1)**2 + 1)

        return cos_term + log_norm_term



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



def univPolynomial_test():

    start_W = torch.tensor([1, 1, 1, 1], dtype = torch.float32)
    
    model = UnivPolynomial(start_W = start_W)

    x = torch.tensor(
        [-1, 1, 2, -2],
        dtype = torch.float32
    )
    x = x.unsqueeze(-1)
    print("Model execution result: \n", model(x))
    #print("Model execution result: \n", model.energy(x))
    print("Model grad w.r.t. input: \n", model.energy_grad(x))
    print("Model mean parameter grad: \n", model.avg_param_grad(x))






if __name__=="__main__":

    #linear_test()
    #mvGaussian_test()
    univPolynomial_test()