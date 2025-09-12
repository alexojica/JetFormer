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
                "train/grad_norm": self._grad_norm(model),
                "train/grad_norm_transformer": grad_norm_transformer,
                "train/grad_norm_jet": grad_norm_jet,
            }
            # Gradient histograms and summary stats
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
            # Bits/dim first (paper-consistent): total, ar, flow
            "train/total_bpd": float(out.get('image_bpd_total', out.get('bpd', 0.0))),
            "train/ar_bpd": float(out.get('ar_bpd_component', 0.0)),
            "train/flow_bpd": float(out.get('flow_bpd_component', 0.0)),
            # NLL (nats) next: total, ar, flow(-logdet)
            "train/nll_nats": float(out.get('total_nll_nats', float('nan'))),
            "train/ar_nll_nats": float(out.get('ar_nll_nats', float('nan'))),
            "train/flow_neg_logdet_nats": float(out.get('flow_neg_logdet_nats', float('nan'))),
            # Other log-likelihoods (nats)
            "train/ar_log_pz_nats": float(out.get('ar_log_pz_nats', float('nan'))),
            "train/log_px_nats": float(out.get('total_log_px_nats', float('nan'))),
            # Text
            "train/text_ce": text_ce,
            "train/text_ppl": text_ppl,
            "train/text_loss_masked": float(out.get('text_loss_masked', text_ce)),
            "train/image_loss_masked": float(out.get('image_loss_masked', out.get('image_bpd_total', 0.0))),
            "train/text_loss_unmasked": float(out.get('text_loss_unmasked', float('nan'))),
            "train/text_ce_denom": float(out.get('text_ce_denom', float('nan'))),
            # Curriculum & noise
            "train/sigma_rgb": float(out.get('sigma_rgb', 0.0)),
            "train/sigma_rgb_final": float(getattr(self.cfg, 'rgb_sigma_final', 3.0)),
            "train/latent_noise_std": float(getattr(self.cfg, 'latent_noise_std', 0.3)),
            "train/cfg_drop_prob": float(getattr(self.cfg, 'cfg_drop_prob', 0.1)),
            # AR/flow diagnostics
            "train/gmm_entropy_nats": float(out.get('gmm_entropy_nats', float('nan'))),
            "train/gmm_log_scales_mean": float(out.get('gmm_log_scales_mean', float('nan'))),
            "train/gmm_log_scales_std": float(out.get('gmm_log_scales_std', float('nan'))),
            "train/ar_hat_tokens_rms": float(out.get('ar_hat_tokens_rms', float('nan'))),
            "train/residual_tokens_rms": float(out.get('residual_tokens_rms', float('nan'))),
            "train/text_first_rate": float(out.get('text_first_rate', float('nan'))),
            "train/flow_logdet_per_patch": float(out.get('flow_logdet_per_patch', float('nan'))),
            "train/image_logits_rms": float(out.get('image_logits_rms', float('nan'))),
            # Optimization / dynamics
            "train/lr": optimizer.param_groups[0]['lr'] if hasattr(optimizer, 'param_groups') else 0.0,
            "train/optimizer_beta2": optimizer.param_groups[0].get('betas', (None, 0.95))[1] if hasattr(optimizer, 'param_groups') else 0.95,
            "train/weight_decay": optimizer.param_groups[0].get('weight_decay', 1e-4) if hasattr(optimizer, 'param_groups') else 1e-4,
            "train/dropout": float(getattr(self.cfg, 'dropout', 0.1)),
            # Timing
            "train/batch_time": batch_time,
            # Housekeeping
            "train/step": step,
            "train/epoch": epoch,
            # Sanity
            "sanity/gmm_small_scales_rate": float(out.get('gmm_small_scales_rate', 0.0)),
        }
        if log_grads and (grad_metrics or grad_hists):
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
        # Recover NLL in nats from BPD using actual model input size
        base = model
        if hasattr(base, 'module'):
            base = base.module
        C = 3
        try:
            H, W = int(base.input_size[0]), int(base.input_size[1])
        except Exception:
            H = int(getattr(self.cfg, 'input_size', (256,256))[0])
            W = int(getattr(self.cfg, 'input_size', (256,256))[1])
        dim_x = float(C * H * W)
        val_total_nll = float(v_img_bpd) * math.log(2.0) * dim_x
        val_flow_neg_logdet = float(v_flow_bpd) * math.log(2.0) * dim_x
        val_ar_nll = val_total_nll - val_flow_neg_logdet
        payload = {
            # Bits/dim first
            'val/total_bpd': v_img_bpd,
            'val/ar_bpd': (v_img_bpd - v_flow_bpd),
            'val/flow_bpd': v_flow_bpd,
            # NLL (nats) next
            'val/nll_nats': val_total_nll,
            'val/ar_nll_nats': val_ar_nll,
            'val/flow_neg_logdet_nats': val_flow_neg_logdet,
            # Text
            'val/text_ce': v_text,
            'val/text_ppl': text_ppl,
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


