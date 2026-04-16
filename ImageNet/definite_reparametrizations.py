import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.parametrize as parametrize
from torch.nn.utils.parametrizations import orthogonal
import os
import math

class Definite(nn.Module):
    def __init__(self, size: int, neg: float):
        assert 0 <= neg <= 1
        super().__init__()
        self.size = size
        self.n_neg = int(neg * size)
        self.L_unconstrained = nn.Parameter(torch.zeros(size, size))
        # Unconstrained diagonal for D
        self.log_diag = nn.Parameter(torch.log(torch.exp(torch.ones(self.size,))-1))
        self.register_buffer('neg', torch.ones(size, dtype=torch.float))
        self.neg[:self.n_neg] = -1
        self.reset_parameters()

    def reset_parameters(self):
        self.L_unconstrained.data = torch.zeros(self.size, self.size)
        if "INIT" in os.environ:
            init = float(os.environ["INIT"])
        else:
            init = 0.15
        # self.log_diag.data = torch.log(torch.exp(0.01*torch.ones(self.size,))-1) 
        self.log_diag.data = torch.log(torch.exp(init*torch.ones(self.size,))-1)

    def forward(self):
        """
        Returns L, D for building positive-definite transformations.
        L is lower-triangular with ones on the diagonal, D is diag(softplus(...)).
        """
        # Force L_unconstrained to be strictly lower-triangular plus Identity on diagonal
        L = torch.tril(self.L_unconstrained, diagonal=-1) + torch.eye(self.size,
                                                                      dtype=self.L_unconstrained.dtype,
                                                                      device=self.L_unconstrained.device)
        # Make sure the diagonal of D is strictly positive via softplus
        self.neg[:self.n_neg] = -1
        d = F.softplus(self.log_diag) * self.neg #+ 0.5
        return L @ torch.diag(d) @ L.transpose(0, 1)


class AntiSym(nn.Module):
    def __init__(self, size: int):
        super().__init__()
        self.size = size
        self.T = nn.Parameter(torch.zeros(size, size))
        nn.init.kaiming_uniform_(self.T, a=math.sqrt(5))  # on node3 use this
        # nn.init.orthogonal(self.T)  # node 2

    def forward(self):
        return self.T - self.T.transpose(0, 1)

def make_positive_definite(matrix: torch.Tensor, eps: float = 1e-2) -> torch.Tensor:
    """
    Given a symmetric 2D tensor (matrix), clip its eigenvalues to ensure all are >= eps.
    Returns a new matrix with only positive eigenvalues (i.e., positive definite).
    """
    assert matrix.dim() == 2 and matrix.size(0) == matrix.size(1), "Input must be a square 2D tensor"
    assert torch.allclose(matrix, matrix.T, atol=1e-6), "Input must be symmetric"

    # Eigen decomposition
    eigvals, eigvecs = torch.linalg.eigh(matrix)

    # Clip eigenvalues
    eigvals_clipped = torch.clamp(eigvals, min=eps)
    # eigvals_clipped = torch.abs(eigvals) / 2

    # Reconstruct the matrix
    matrix_pd = eigvecs @ torch.diag(eigvals_clipped) @ eigvecs.T

    return matrix_pd


class QKVParametrization(nn.Module):
    def __init__(self, size: int, wneg: float, aneg: float):
        super().__init__()
        self.size = size
        self.q = nn.Parameter(torch.zeros(size, size))
        self.Wsym_generator = Definite(size, wneg)
        self.Asym_generator = Definite(size, aneg)
        self.Tw_generator = AntiSym(size)
        self.Ta_generator = AntiSym(size)

        with torch.no_grad():

            qinit = torch.empty_like(self.q).to(self.q.device)
            kinit = torch.empty_like(self.q).to(self.q.device)
            vinit = torch.empty_like(self.q).to(self.q.device)
            nn.init.orthogonal(qinit)
            nn.init.kaiming_uniform_(kinit, a=math.sqrt(5))
            nn.init.kaiming_uniform_(vinit, a=math.sqrt(5))
            Winit = qinit @ kinit.T
            W_sym_init = 0.5 * (Winit + Winit.T)
            T_w_init   = 0.5 * (Winit - Winit.T)

            vtinit = torch.inverse(vinit.T)
            Ainit = Winit @ vtinit
            A_sym_init = 0.5 * (Ainit + Ainit.T)
            T_a_init = 0.5 * (Ainit - Ainit.T)

            self.q.data = qinit
            self.Wsym_generator.L_unconstrained.data = torch.cholesky(make_positive_definite(W_sym_init))
            self.Tw_generator.T.data = T_w_init

            self.Asym_generator.L_unconstrained.data = torch.cholesky(make_positive_definite(A_sym_init))
            self.Ta_generator.T.data = T_a_init

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        # Materialize everything exactly once
        Q = self.q + 0.0  # Force "unwrapped" param
        W = self.Wsym_generator() + self.Tw_generator() + 0.0
        Asym = self.Asym_generator() + 0.0
        antisym_a = self.Ta_generator() + 0.0

        # Now use only the unwrapped local Tensors
        K = torch.linalg.solve(Q, W).transpose(0, 1)
        V = torch.linalg.solve(Asym + antisym_a, W)
        # print('reparam')
        return torch.cat([Q, K, V], dim=0)
