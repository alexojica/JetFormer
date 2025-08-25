import time
import math
import torch


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
            self.wb.summary.update({
                # Param counts
                "model/total_params": param_counts.get('total', 0),
                "model/jet_params": param_counts.get('jet', 0),
                "model/transformer_params": param_counts.get('transformer', 0),
                # Config snapshot
                "config/image_size": int(getattr(self.cfg, 'input_size', (256, 256))[0]),
                "config/patch_size": int(getattr(self.cfg, 'patch_size', 16)),
                "config/factored_dims": int(getattr(self.cfg, 'image_ar_dim', 128)),
                "config/mixtures_k": int(getattr(self.cfg, 'num_mixtures', 1024)),
                "config/image_ar_dim": int(getattr(self.cfg, 'image_ar_dim', 128)),
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
                "config/params_total": param_counts.get('total', 0),
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

    def log_train_step(self, model, optimizer, out: dict, step: int, epoch: int, batch_time: float):
        if not self.enabled:
            return
        text_ce = float(out.get('text_loss', 0.0))
        text_ppl = float(math.exp(min(30.0, text_ce))) if text_ce > 0 else 0.0
        payload = {
            # Core likelihood metrics
            "train/total_bpd": float(out.get('image_bpd_total', 0.0)),
            # Alias matching paper terminology
            "train/nll_bpd": float(out.get('image_bpd_total', 0.0)),
            "train/flow_bpd": float(out.get('flow_bpd_component', 0.0)),
            "train/ar_bpd": float(out.get('ar_bpd_component', 0.0)),
            # Raw AR log-likelihood (nats)
            "train/image_loglik": float(out.get('image_loglik_nats', 0.0)),
            # Text
            "train/text_ce": text_ce,
            "train/text_ppl": text_ppl,
            # Curriculum & noise
            "train/sigma_rgb": float(out.get('sigma_rgb', 0.0)),
            "train/sigma_rgb_final": float(getattr(self.cfg, 'rgb_sigma_final', 3.0)),
            "train/latent_noise_std": float(getattr(self.cfg, 'latent_noise_std', 0.3)),
            "train/cfg_drop_prob": float(getattr(self.cfg, 'cfg_drop_prob', 0.1)),
            # Optimization / dynamics
            "train/lr": optimizer.param_groups[0]['lr'] if hasattr(optimizer, 'param_groups') else 0.0,
            "train/grad_norm": self._grad_norm(model),
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
        try:
            self.wb.log(payload)
        except Exception:
            pass

    def log_validation_epoch(self, v_total: float, v_text: float, v_img_bpd: float, v_flow_bpd: float, epoch: int, step: int):
        if not self.enabled:
            return
        text_ppl = float(math.exp(min(30.0, v_text))) if v_text > 0 else 0.0
        payload = {
            # Bits/dim
            'val/total_bpd': v_img_bpd,
            # Alias matching paper terminology
            'val/nll_bpd': v_img_bpd,
            'val/flow_bpd': v_flow_bpd,
            'val/ar_bpd': max(0.0, v_img_bpd - v_flow_bpd),
            # Text
            'val/text_ce': v_text,
            'val/text_ppl': text_ppl,
            # Housekeeping
            'epoch': epoch,
            'global_step': step,
        }
        try:
            self.wb.log(payload)
        except Exception:
            pass

 