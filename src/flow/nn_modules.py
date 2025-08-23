import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ActNorm(nn.Module):
    """An activation normalization layer that normalizes inputs using data-dependent initialization of scale and bias."""
    def __init__(self, num_features: int, eps: float = 1e-6):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.log_scale = nn.Parameter(torch.zeros(1, 1, 1, num_features), requires_grad=False)
        self.bias = nn.Parameter(torch.zeros(1, 1, 1, num_features), requires_grad=False)
        self.initialized = False

    def _initialize(self, x: torch.Tensor):
        """Initializes scale and bias based on the first batch of data."""
        if not self.training:
            return
        
        with torch.no_grad():
            # (B, H, W, C) -> mean/std over B, H, W
            mean = torch.mean(x, dim=(0, 1, 2))
            std = torch.std(x, dim=(0, 1, 2))
            
            self.log_scale.data.copy_(-torch.log(std + self.eps))
            self.bias.data.copy_(-mean)
            
            self.log_scale.requires_grad = True
            self.bias.requires_grad = True
            self.initialized = True
            print(f"ActNorm initialized for {self.num_features} features.")

    def forward(self, x: torch.Tensor):
        """Applies activation normalization and returns the transformed output and log-determinant."""
        if not self.initialized:
            self._initialize(x)

        z = (x + self.bias) * torch.exp(self.log_scale)
        
        B, H, W, C = x.shape
        logdet = H * W * torch.sum(self.log_scale)
        
        return z, logdet.expand(B)

    def inverse(self, z: torch.Tensor):
        """Applies the inverse of activation normalization."""
        x = z * torch.exp(-self.log_scale) - self.bias
        
        B, H, W, C = z.shape
        inv_logdet = -H * W * torch.sum(self.log_scale)

        return x, inv_logdet.expand(B)


class Invertible1x1Conv(nn.Module):
    """An invertible 1x1 convolution layer using LU decomposition for efficient determinant calculation."""
    def __init__(self, num_channels: int):
        super().__init__()
        self.num_channels = num_channels
        
        W, _ = torch.linalg.qr(torch.randn(num_channels, num_channels))
        P, L, U = torch.linalg.lu(W)
        
        self.register_buffer('P', P)
        self.L = nn.Parameter(L)
        self.U = nn.Parameter(U)

        self.L_mask = torch.tril(torch.ones_like(self.L), diagonal=-1)
        self.U_mask = torch.triu(torch.ones_like(self.U), diagonal=0)

    def _get_weight(self):
        """Computes the convolutional weight matrix from its LU decomposition."""
        L = self.L * self.L_mask + torch.eye(self.num_channels, device=self.L.device)
        U = self.U * self.U_mask
        W = self.P @ L @ U
        return W.view(self.num_channels, self.num_channels, 1, 1)

    def forward(self, x: torch.Tensor):
        """Applies the 1x1 convolution and returns the output and log-determinant."""
        B, H, W, C = x.shape
        W = self._get_weight()
        
        # permute to (B, C, H, W) for conv2d
        x_perm = x.permute(0, 3, 1, 2)
        z_perm = F.conv2d(x_perm, W)
        z = z_perm.permute(0, 2, 3, 1)

        logdet = H * W * torch.sum(torch.log(torch.abs(self.U.diag())))
        
        return z, logdet.expand(B)

    def inverse(self, z: torch.Tensor):
        """Applies the inverse 1x1 convolution."""
        B, H, W, C = z.shape
        W = self._get_weight()
        W_inv = torch.inverse(W.squeeze()).view(self.num_channels, self.num_channels, 1, 1)
        
        z_perm = z.permute(0, 3, 1, 2)
        x_perm = F.conv2d(z_perm, W_inv)
        x = x_perm.permute(0, 2, 3, 1)
        
        inv_logdet = -H * W * torch.sum(torch.log(torch.abs(self.U.diag())))

        return x, inv_logdet.expand(B) 