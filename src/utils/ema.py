import torch


class ExponentialMovingAverage:
    def __init__(self, model, decay: float = 0.9999):
        self.decay = float(decay)
        self.shadow = {}
        base = model
        if hasattr(base, 'module'):
            base = base.module
        for name, param in base.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.detach().clone().float()

    @torch.no_grad()
    def update(self, model):
        base = model
        if hasattr(base, 'module'):
            base = base.module
        for name, param in base.named_parameters():
            if not param.requires_grad:
                continue
            if name not in self.shadow:
                self.shadow[name] = param.detach().clone().float()
            else:
                self.shadow[name].mul_(self.decay).add_(param.detach().float(), alpha=(1.0 - self.decay))

    def state_dict(self):
        return {k: v.cpu() for k, v in self.shadow.items()}

    def load_state_dict(self, state):
        self.shadow = {k: v.clone() for k, v in state.items()}

    @torch.no_grad()
    def apply_to(self, model):
        base = model
        if hasattr(base, 'module'):
            base = base.module
        self._backup = {}
        for name, param in base.named_parameters():
            if name in self.shadow:
                self._backup[name] = param.detach().clone()
                param.data.copy_(self.shadow[name].to(param.dtype).to(param.device))

    @torch.no_grad()
    def restore(self, model):
        base = model
        if hasattr(base, 'module'):
            base = base.module
        if not hasattr(self, '_backup'):
            return
        for name, param in base.named_parameters():
            if name in self._backup:
                param.data.copy_(self._backup[name])
        self._backup = {}


