from types import SimpleNamespace
from typing import Any, Dict, Tuple

import torch

from src.jetformer import JetFormer
try:
    from src.utils.logging import get_logger as _get_logger
    _mf_logger = _get_logger(__name__)
except Exception:
    _mf_logger = None


def _get_from_ns_or_dict(config: SimpleNamespace | Dict[str, Any], key: str, default=None):
    """Safely get a value from a SimpleNamespace or dict, supporting dot notation."""
    if hasattr(config, 'get') and callable(getattr(config, 'get')):
        return config.get(key, default)
    
    # Fallback for plain objects or dicts without a .get method for nested keys
    try:
        keys = key.split('.')
        val = config
        for k in keys:
            if isinstance(val, dict):
                val = val.get(k)
                if val is None: return default
            elif isinstance(val, SimpleNamespace):
                val = getattr(val, k, None)
                if val is None: return default
            else:
                return default
        return val
    except (AttributeError, KeyError):
        return default
        

def build_jetformer_from_config(config: SimpleNamespace | Dict[str, Any], device: torch.device) -> JetFormer:
    """Construct JetFormer from a nested SimpleNamespace or dict config."""
    
    # Helper to resolve a config value, preferring the nested SimpleNamespace first
    def _get(key, default=None):
        return _get_from_ns_or_dict(config, key, default)

    # --- Model Parameters ---
    model_cfg = _get('model', {})
    kwargs = {
        'd_model': _get_from_ns_or_dict(model_cfg, 'width'),
        'n_layers': _get_from_ns_or_dict(model_cfg, 'depth'),
        'd_ff': _get_from_ns_or_dict(model_cfg, 'mlp_dim'),
        'n_heads': _get_from_ns_or_dict(model_cfg, 'num_heads'),
        'n_kv_heads': _get_from_ns_or_dict(model_cfg, 'num_kv_heads'),
        'vocab_size': _get_from_ns_or_dict(model_cfg, 'vocab_size'),
        'bos_id': _get_from_ns_or_dict(model_cfg, 'bos_id'),
        'boi_id': _get_from_ns_or_dict(model_cfg, 'boi_id'),
        'nolabel_id': _get_from_ns_or_dict(model_cfg, 'nolabel_id'),
        'num_mixtures': _get_from_ns_or_dict(model_cfg, 'num_mixtures'),
        'dropout': _get_from_ns_or_dict(model_cfg, 'dropout'),
        'use_bfloat16_img_head': _get_from_ns_or_dict(model_cfg, 'head_dtype', 'fp32') == 'bfloat16',
        'num_vocab_repeats': _get_from_ns_or_dict(model_cfg, 'num_vocab_repeats', 1),
        'scale_tol': _get_from_ns_or_dict(model_cfg, 'scale_tol', 1e-6),
        'causal_mask_on_prefix': _get_from_ns_or_dict(model_cfg, 'causal_mask_on_prefix', True),
        'untie_output_vocab': _get_from_ns_or_dict(model_cfg, 'untie_output_vocab', False),
        'per_modality_final_norm': _get_from_ns_or_dict(model_cfg, 'per_modality_final_norm', False),
        'right_align_inputs': _get_from_ns_or_dict(model_cfg, 'right_align_inputs', True),
        'strict_special_ids': _get_from_ns_or_dict(model_cfg, 'strict_special_ids', True),
        'use_boi_token': _get_from_ns_or_dict(model_cfg, 'use_boi_token', True),
        'max_seq_len': _get_from_ns_or_dict(model_cfg, 'max_seq_len'),
        'rope_skip_pad': _get_from_ns_or_dict(model_cfg, 'rope_skip_pad', True),
        'grad_checkpoint_transformer': _get_from_ns_or_dict(model_cfg, 'remat_policy', 'none') != 'none',
    }

    # --- Input and PCA/Adaptor dependent params ---
    input_cfg = _get('input', {})
    pca_model_cfg = _get('patch_pca.model', {})
    adaptor_cfg = _get('adaptor', {})
    adaptor_model_cfg = _get_from_ns_or_dict(adaptor_cfg, 'model', {})
    
    kwargs['input_size'] = tuple(_get_from_ns_or_dict(input_cfg, 'input_size'))
    kwargs['patch_size'] = int(_get_from_ns_or_dict(pca_model_cfg, 'patch_size'))
    kwargs['image_ar_dim'] = int(_get_from_ns_or_dict(pca_model_cfg, 'codeword_dim'))
    kwargs['num_classes'] = _get_from_ns_or_dict(input_cfg, 'num_classes')
    kwargs['class_token_length'] = _get_from_ns_or_dict(input_cfg, 'class_token_length')

    # --- Jet/Flow Parameters ---
    kwargs['jet_depth'] = _get_from_ns_or_dict(adaptor_model_cfg, 'depth')
    kwargs['jet_block_depth'] = _get_from_ns_or_dict(adaptor_model_cfg, 'block_depth')
    kwargs['jet_emb_dim'] = _get_from_ns_or_dict(adaptor_model_cfg, 'emb_dim')
    kwargs['jet_num_heads'] = _get_from_ns_or_dict(adaptor_model_cfg, 'num_heads')
    kwargs['flow_actnorm'] = _get_from_ns_or_dict(adaptor_model_cfg, 'actnorm', False)
    kwargs['flow_invertible_dense'] = _get_from_ns_or_dict(adaptor_model_cfg, 'invertible_dense', False)
    kwargs['flow_grad_checkpoint'] = _get_from_ns_or_dict(adaptor_model_cfg, 'flow_grad_checkpoint', False)

    # Filter out None values before passing to constructor
    final_kwargs = {k: v for k, v in kwargs.items() if v is not None}
    
    model = JetFormer(**final_kwargs).to(device)
    
    # Attach training mode
    model.training_mode = _get('jetformer_training_mode', 'pca')
    
    # Attach PatchPCA and Adaptor
    try:
        from src.latents.patch_pca import PatchPCA
        if _get('patch_pca') is not None:
            model.patch_pca = PatchPCA(**_get('patch_pca.model', {})).to(device)
    except Exception:
        pass # Fail gracefully if import or instantiation fails
        
    try:
        from src.latents.jet_adaptor import build_adaptor
        if _get('use_adaptor', False):
            H, W = kwargs['input_size']
            ps = kwargs['patch_size']
            grid_h, grid_w = H // ps, W // ps
            
            # The dimension for the adaptor is the full patch token dimension
            full_token_dim = 3 * ps * ps

            adaptor_kind = _get_from_ns_or_dict(adaptor_cfg, 'kind', 'jet')
            # Pass adaptor_model_cfg as kwargs to build_adaptor
            model.adaptor = build_adaptor(
                kind=adaptor_kind, 
                grid_h=grid_h, 
                grid_w=grid_w, 
                dim=full_token_dim, 
                **adaptor_model_cfg
            ).to(device)
            
            # For latent_noise_dim
            model._latent_noise_dim = _get_from_ns_or_dict(adaptor_cfg, 'latent_noise_dim', 0)
    except Exception:
        pass
        
    # Alias model.jet to the correct flow module
    try:
        if model.training_mode == 'pca' and getattr(model, 'adaptor', None) is not None:
            model.jet = getattr(model.adaptor, 'flow', model.adaptor)
            model.jet_is_latent = True
        else:
            model.jet_is_latent = bool(_get('model.pre_factor_dim') is not None)
    except Exception:
        pass
        
    return model


