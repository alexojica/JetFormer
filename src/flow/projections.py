import torch
import torch.nn as nn


class InvertibleLinear(nn.Module):
    """Invertible linear layer using LU parameterization.

    Exposes forward/inverse with log-determinant and a helper to set weights
    from a provided full-rank matrix.
    """

    def __init__(self, dim: int):
        super().__init__()
        # Initialize with a random orthonormal matrix, then take LU
        W, _ = torch.linalg.qr(torch.randn(dim, dim))
        P, L, U = torch.linalg.lu(W)
        self.register_buffer('P', P)
        self.L = nn.Parameter(L)
        self.U = nn.Parameter(U)
        # Masks: unit diagonal on L; upper-triangular on U
        # Register as buffers so they follow the module to the right device/dtype
        self.register_buffer('L_mask', torch.tril(torch.ones_like(self.L), diagonal=-1), persistent=False)
        self.register_buffer('U_mask', torch.triu(torch.ones_like(self.U), diagonal=0), persistent=False)

    def _weight(self) -> torch.Tensor:
        L = self.L * self.L_mask + torch.eye(self.L.shape[0], device=self.L.device, dtype=self.L.dtype)
        U = self.U * self.U_mask
        W = self.P @ L @ U
        return W

    def forward(self, x: torch.Tensor):
        # x: [..., D]
        W = self._weight()
        y = x @ W.t()
        logdet = torch.sum(torch.log(torch.abs(torch.diag(self.U))))
        return y, logdet

    def inverse(self, y: torch.Tensor):
        W = self._weight()
        WinvT = torch.inverse(W).t()
        x = y @ WinvT
        logdet = -torch.sum(torch.log(torch.abs(torch.diag(self.U))))
        return x, logdet

    def set_weight(self, W_new: torch.Tensor, frozen: bool = True):
        """Decompose provided W into LU and populate parameters.

        If frozen=True, gradients are disabled for L and U.
        """
        with torch.no_grad():
            P, L, U = torch.linalg.lu(W_new)
            self.P.copy_(P)
            self.L.copy_(L)
            self.U.copy_(U)
        if frozen:
            for p in self.parameters():
                p.requires_grad = False


    


__all__ = [
    'InvertibleLinear',
]


