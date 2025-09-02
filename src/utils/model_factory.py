from types import SimpleNamespace
from typing import Any, Dict, Tuple

import torch

from src.jetformer import JetFormer


def _get_from_ns_or_dict(config: SimpleNamespace | Dict[str, Any], key: str, default=None):
    if isinstance(config, dict):
        return config.get(key, default)
    try:
        return getattr(config, key)
    except Exception:
        return default


def build_jetformer_from_config(config: SimpleNamespace | Dict[str, Any], device: torch.device) -> JetFormer:
    """Construct JetFormer from a SimpleNamespace or dict config on the given device.

    Accepts both dict- and attribute-style configs. Handles tuple conversion for input_size.
    """
    param_names = [
        'vocab_size','d_model','n_heads','n_kv_heads','n_layers','d_ff','max_seq_len','num_mixtures','dropout',
        'jet_depth','jet_block_depth','jet_emb_dim','jet_num_heads','patch_size','image_ar_dim','use_bfloat16_img_head',
        'num_classes','class_token_length','latent_projection','latent_proj_matrix_path','pre_latent_projection',
        'pre_latent_proj_matrix_path','pre_factor_dim','flow_actnorm','flow_invertible_dense',
        'grad_checkpoint_transformer','flow_grad_checkpoint'
    ]
    kwargs: Dict[str, Any] = {}
    for name in param_names:
        val = _get_from_ns_or_dict(config, name, None)
        if name == 'pre_factor_dim':
            v = val
            if isinstance(v, str):
                vl = v.strip().lower()
                if vl in {"none", "null", "false", ""}:
                    v = None
                else:
                    try:
                        v = int(v)
                    except Exception:
                        v = None
            elif isinstance(v, float):
                try:
                    v = int(v)
                except Exception:
                    v = None
            if isinstance(v, int) and v <= 0:
                v = None
            if v is not None:
                kwargs[name] = v
        elif val is not None:
            kwargs[name] = val
    inp = _get_from_ns_or_dict(config, 'input_size', None)
    if inp is not None:
        kwargs['input_size'] = tuple(inp)
    model = JetFormer(**kwargs).to(device)
    return model


