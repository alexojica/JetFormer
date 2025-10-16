import logging
import os
import time
import math
import torch
import wandb


class WBLogger:
    """Centralized Weights & Biases logging with JetFormer terminology."""
    def __init__(self, wandb_run, config):
        self.wb = wandb_run
        self.cfg = config
        self.enabled = (self.wb is not None)

    def update_summary_config(self, param_counts: dict):
        if not self.enabled:
            return
        try:
            # Derive a few stable config fields used across runs
            ps_val = int(getattr(self.cfg, 'patch_size', 16))
            self.wb.summary.update({
                # Param counts
                "model/total_params": param_counts.get('total', 0),
                "model/jet_params": param_counts.get('jet', 0),
                "model/transformer_params": param_counts.get('transformer', 0),
                # Config snapshot
                "config/image_size": int(getattr(self.cfg, 'input_size', (256, 256))[0]),
                "config/patch_size": ps_val,
                # Autoregressive token dimensionality (paper uses d)
                "config/mixtures_k": int(getattr(self.cfg, 'num_mixtures', 1024)),
                "config/image_ar_dim": int(getattr(self.cfg, 'image_ar_dim', 128)),
                # Full token dimensionality per image token (3 * p^2)
                "config/image_token_dim": int(3 * ps_val * ps_val),
                "config/num_class_tokens": int(getattr(self.cfg, 'class_token_length', 16)),
                "config/jet/num_blocks": int(getattr(self.cfg, 'jet_depth', 8)),
                "config/jet/width": int(getattr(self.cfg, 'jet_emb_dim', 512)),
                "config/jet/depth_per_block": int(getattr(self.cfg, 'jet_block_depth', 2)),
                "config/jet/num_heads": int(getattr(self.cfg, 'jet_num_heads', 8)),
                "config/ar/depth": int(getattr(self.cfg, 'n_layers', 12)),
                "config/ar/width": int(getattr(self.cfg, 'd_model', 768)),
                "config/ar/num_heads": int(getattr(self.cfg, 'n_heads', 12)),
                "config/ar/n_kv_heads": int(getattr(self.cfg, 'n_kv_heads', 1)),
                "config/rope": True,
            })
        except Exception:
            pass

    def _param_norm(self, model_or_module) -> float:
        try:
            total = 0.0
            for p in model_or_module.parameters():
                if p.requires_grad:
                    total += float(torch.sum(p.data.float() * p.data.float()))
            return float(total) ** 0.5
        except Exception:
            return float('nan')

    def _grad_norm(self, model) -> float:
        try:
            total = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    g = p.grad.data
                    total += float(torch.sum(g * g))
            return float(total) ** 0.5
        except Exception:
            return float('nan')

    def log_train_step(self, model, optimizer, out: dict, step: int, epoch: int, batch_time: float, log_grads: bool = False):
        if not self.enabled:
            return
        text_ce = float(out.get('text_loss', 0.0))
        text_ppl = float(math.exp(min(30.0, text_ce))) if text_ce > 0 else 0.0
        # Resolve base module for submodule-specific grad norms (computed only when requested)
        grad_metrics = {}
        grad_hists = {}
        if log_grads:
            base = model
            if hasattr(base, 'module'):
                base = base.module
            try:
                grad_norm_transformer = self._grad_norm(base.transformer) if hasattr(base, 'transformer') else float('nan')
            except Exception:
                grad_norm_transformer = float('nan')
            try:
                grad_norm_jet = self._grad_norm(base.jet) if hasattr(base, 'jet') else float('nan')
            except Exception:
                grad_norm_jet = float('nan')
            grad_metrics = {
                "diag/optim/grad_norm": self._grad_norm(model),
                "diag/optim/grad_norm_transformer": grad_norm_transformer,
                "diag/optim/grad_norm_jet": grad_norm_jet,
                # L2 norm of parameters
                "diag/optim/l2_params": self._param_norm(model),
                "diag/optim/l2_params_transformer": self._param_norm(base.transformer) if hasattr(base, 'transformer') else 0.0,
                "diag/optim/l2_params_flow": self._param_norm(base.jet) if hasattr(base, 'jet') else 0.0,
            }
            # Grad histograms
            def _concat_grads(module):
                try:
                    grads = []
                    for p in module.parameters():
                        if p.grad is not None:
                            g = p.grad.detach()
                            if g.is_sparse:
                                g = g.coalesce().values()
                            grads.append(g.reshape(-1))
                    if len(grads) == 0:
                        return None
                    return torch.cat(grads)
                except Exception:
                    return None

            def _add_hist(tag: str, module):
                t = _concat_grads(module)
                if t is None or t.numel() == 0:
                    return
                t_f = t.float()
                try:
                    grad_hists[f"train/grads/{tag}"] = wandb.Histogram(t_f.detach().cpu().numpy())
                except Exception:
                    # Fallback: skip histogram if construction fails
                    pass
                try:
                    grad_metrics[f"train/grads/{tag}_mean"] = float(t_f.mean().item())
                    grad_metrics[f"train/grads/{tag}_std"] = float(t_f.std(unbiased=False).item())
                    grad_metrics[f"train/grads/{tag}_max_abs"] = float(t_f.abs().max().item())
                except Exception:
                    pass

            _add_hist("model", model)
            if hasattr(base, 'transformer'):
                _add_hist("transformer", base.transformer)
            if hasattr(base, 'jet'):
                _add_hist("jet", base.jet)
        # Prefer explicit NLL (nats) if provided; otherwise derive from BPD if possible
        payload = {
            # Loss components
            "loss/total": float(out.get('loss', 0.0)),
            "loss/image": float(out.get('image_loss', 0.0)),
            "loss/text": float(out.get('text_loss', 0.0)),
            # Bits/dim (paper-consistent): total, ar, flow
            "bpd/total": float(out.get('image_bpd_total', out.get('bpd', 0.0))),
            "bpd/ar": float(out.get('ar_bpd_component', 0.0)),
            "bpd/flow": float(out.get('flow_bpd_component', 0.0)),
            # Text metrics (CE)
            "text/ce": text_ce,
            "text/ppl": text_ppl,
            "text/ce_denom": float(out.get('text_ce_denom', float('nan'))),
            "text/ce_prefix": float(out.get('nll_text_prefix', float('nan'))),
            "text/ce_suffix": float(out.get('nll_text_suffix', float('nan'))),
            # Image metrics (BPD)
            "bpd/image_prefix": float(out.get('nll_image_prefix', float('nan'))),
            "bpd/image_suffix": float(out.get('nll_image_suffix', float('nan'))),
            # Curriculum & noise
            "diag/sigma_rgb": float(out.get('sigma_rgb', 0.0)),
            "diag/latent_noise_std": float(getattr(self.cfg, 'latent_noise_std', 0.3)),
            # AR/flow diagnostics
            "diag/ar/gmm_entropy_nats": float(out.get('gmm_entropy_nats', float('nan'))),
            "diag/ar/gmm_log_scales_mean": float(out.get('gmm_log_scales_mean', float('nan'))),
            "diag/ar/gmm_log_scales_std": float(out.get('gmm_log_scales_std', float('nan'))),
            "diag/ar/image_logits_rms": float(out.get('image_logits_rms', float('nan'))),
            "diag/flow/logdet_per_patch": float(out.get('flow_logdet_per_patch', float('nan'))),
            "diag/latent/ar_hat_tokens_rms": float(out.get('ar_hat_tokens_rms', float('nan'))),
            "diag/latent/residual_tokens_rms": float(out.get('residual_tokens_rms', float('nan'))),
            "diag/sanity/gmm_small_scales_rate": float(out.get('gmm_small_scales_rate', 0.0)),
            "diag/text_first_rate": float(out.get('text_first_rate', float('nan'))),
            # Optimization / dynamics
            "diag/optim/lr": optimizer.param_groups[0]['lr'] if hasattr(optimizer, 'param_groups') else 0.0,
            "diag/optim/beta2": optimizer.param_groups[0].get('betas', (None, 0.95))[1] if hasattr(optimizer, 'param_groups') else 0.95,
            "diag/optim/wd": optimizer.param_groups[0].get('weight_decay', 1e-4) if hasattr(optimizer, 'param_groups') else 1e-4,
            # Timing & housekeeping
            "perf/batch_time": batch_time,
            "step": step,
            "epoch": epoch,
        }
        if log_grads and (grad_metrics or grad_hists):
            # Add parameter norms for other components
            try:
                p_other = grad_metrics.get("diag/optim/l2_params", 0.0)**2
                p_transformer = grad_metrics.get("diag/optim/l2_params_transformer", 0.0)**2
                p_flow = grad_metrics.get("diag/optim/l2_params_flow", 0.0)**2
                p_other = max(0.0, p_other - p_transformer - p_flow) ** 0.5
                grad_metrics["diag/optim/l2_params_other"] = p_other
            except Exception:
                grad_metrics["diag/optim/l2_params_other"] = 0.0
            payload.update(grad_metrics)
            payload.update(grad_hists)
        try:
            # Use explicit step so W&B curves are aligned with optimizer steps
            self.wb.log(payload, step=int(step))
        except Exception:
            pass

    def log_validation_epoch(self, model, v_total: float, v_text: float, v_img_bpd: float, v_flow_bpd: float, epoch: int, step: int):
        if not self.enabled:
            return
        text_ppl = float(math.exp(min(30.0, v_text))) if v_text > 0 else 0.0
        payload = {
            # Main validation loss
            "val/loss": v_total,
            # BPD components
            "val/bpd/total": v_img_bpd,
            "val/bpd/ar": (v_img_bpd - v_flow_bpd),
            "val/bpd/flow": v_flow_bpd,
            # Text metrics
            "val/text/ce": v_text,
            "val/text/ppl": text_ppl,
            # Housekeeping
            'epoch': epoch,
            'global_step': step,
        }
        try:
            # Log a single point per validation by committing exactly once with the optimizer step index
            self.wb.log(payload, step=int(step))
        except Exception:
            pass

def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        level = os.environ.get("JETFORMER_LOG_LEVEL", "INFO").upper()
        logger.setLevel(getattr(logging, level, logging.INFO))
        handler = logging.StreamHandler()
        fmt = logging.Formatter('[%(asctime)s] %(name)s %(levelname)s: %(message)s')
        handler.setFormatter(fmt)
        logger.addHandler(handler)
        logger.propagate = False
    return logger


