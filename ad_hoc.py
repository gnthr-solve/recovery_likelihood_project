
import torch

from hydra import initialize, compose
from hydra.utils import instantiate
from pathlib import Path
from experiment import Experiment
from training_observer import TimingObserver, ParameterObserver, LikelihoodObserver
from result_manager import ResultManager
from metrics import apply_param_metric_to_df, FrobeniusError, LpError

from norm_const import norm_const
from test_distributions import UnivPolynomial
# check computation backend to use
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("-device:", device)

W = torch.tensor([0, -1.2, -0.7, 2, 1], dtype=torch.float32)

poly_dist = UnivPolynomial(W_0=W)

poly_norm_const = norm_const(poly_dist, 1)

print(poly_norm_const)