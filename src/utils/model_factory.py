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
        'patch_size','image_ar_dim','use_bfloat16_img_head',
        'num_classes','class_token_length','latent_projection','latent_proj_matrix_path','pre_latent_projection',
        'pre_latent_proj_matrix_path','pre_factor_dim',
        'grad_checkpoint_transformer','flow_grad_checkpoint',
        # New parity flags
        'use_boi_token','causal_mask_on_prefix','untie_output_vocab','per_modality_final_norm',
        'num_vocab_repeats','bos_id','boi_id','nolabel_id','scale_tol','right_align_inputs'
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

    # Handle jet config from a nested block
    jet_cfg = _get_from_ns_or_dict(config, 'jet', {})
    jet_params_for_constructor = {
        'jet_depth': _get_from_ns_or_dict(jet_cfg, 'depth'),
        'jet_block_depth': _get_from_ns_or_dict(jet_cfg, 'block_depth'),
        'jet_emb_dim': _get_from_ns_or_dict(jet_cfg, 'emb_dim'),
        'jet_num_heads': _get_from_ns_or_dict(jet_cfg, 'num_heads'),
        'flow_actnorm': _get_from_ns_or_dict(jet_cfg, 'actnorm'),
        'flow_invertible_dense': _get_from_ns_or_dict(jet_cfg, 'invertible_dense'),
    }
    kwargs.update({k: v for k, v in jet_params_for_constructor.items() if v is not None})

    # Enforce strict special tokens: require ids when flag true; else warn
    bos_id = _get_from_ns_or_dict(config, 'bos_id', None)
    nolabel_id = _get_from_ns_or_dict(config, 'nolabel_id', None)
    boi_id = _get_from_ns_or_dict(config, 'boi_id', None)
    ssi_cfg = _get_from_ns_or_dict(config, 'strict_special_ids', True)
    if ssi_cfg and not (isinstance(bos_id, int) and isinstance(nolabel_id, int)):
        # Hard default to strict parity; if missing ids, set safe defaults and continue
        bos_id = 0 if bos_id is None else bos_id
        nolabel_id = 0 if nolabel_id is None else nolabel_id
        try:
            if _mf_logger is not None:
                _mf_logger.warning("strict_special_ids=True but missing bos_id/nolabel_id; defaulting missing ids to 0 for parity.")
        except Exception:
            pass
    kwargs['bos_id'] = bos_id
    kwargs['nolabel_id'] = nolabel_id
    if boi_id is not None:
        kwargs['boi_id'] = boi_id
    # Pass through explicit intent to use BOI token (default True for parity) and enforce under strict mode
    use_boi = _get_from_ns_or_dict(config, 'use_boi_token', True)
    kwargs['use_boi_token'] = bool(use_boi)
    kwargs['strict_special_ids'] = bool(ssi_cfg)

    # Optional dtype parsing for heads/embeddings
    def _to_torch_dtype(v):
        if v is None:
            return None
        if isinstance(v, torch.dtype):
            return v
        if isinstance(v, str):
            s = v.strip().lower()
            return {
                'fp32': torch.float32,
                'float32': torch.float32,
                'f32': torch.float32,
                'bf16': torch.bfloat16,
                'bfloat16': torch.bfloat16,
                'fp16': torch.float16,
                'float16': torch.float16,
            }.get(s, None)
        return None

    hd = _get_from_ns_or_dict(config, 'head_dtype', None)
    ed = _get_from_ns_or_dict(config, 'embed_dtype', None)
    if hd is not None:
        kwargs['head_dtype'] = _to_torch_dtype(hd)
    if ed is not None:
        kwargs['embed_dtype'] = _to_torch_dtype(ed)
    # Attention logits softcap
    sc = _get_from_ns_or_dict(config, 'attn_logits_softcap', None)
    if sc is not None:
        try:
            kwargs['attn_logits_softcap'] = float(sc)
        except Exception:
            kwargs['attn_logits_softcap'] = None
    # Multivariate head flags
    mv = _get_from_ns_or_dict(config, 'multivariate', None)
    if mv is not None:
        kwargs['multivariate'] = bool(mv)
    mv_d = _get_from_ns_or_dict(config, 'out_dim', None)
    if mv_d is not None:
        kwargs['multivariate_out_dim'] = int(mv_d)
    # Optional rope_skip_pad exposure (default False)
    rsp = _get_from_ns_or_dict(config, 'rope_skip_pad', None)
    if rsp is not None:
        kwargs['rope_skip_pad'] = bool(rsp)
    else:
        # Default to True for JAX parity in masked positions
        kwargs['rope_skip_pad'] = True
    # Optional strict_special_ids exposure
    ssi = _get_from_ns_or_dict(config, 'strict_special_ids', None)
    if ssi is not None:
        kwargs['strict_special_ids'] = bool(ssi)
    inp = _get_from_ns_or_dict(config, 'input_size', None)
    if inp is not None:
        kwargs['input_size'] = tuple(inp)
    model = JetFormer(**kwargs).to(device)

    # Training-mode gate: default to PCA+Adaptor (paper path) unless explicitly set to 'legacy'
    training_mode = str(_get_from_ns_or_dict(config, 'jetformer_training_mode', 'pca') or 'pca').lower()

    # Attach PatchPCA and Adaptor either from nested blocks or sensible defaults when training_mode=='pca'
    try:
        from src.latents.patch_pca import PatchPCA
        pca_cfg = _get_from_ns_or_dict(config, 'patch_pca', None)
        if training_mode == 'pca' and not isinstance(pca_cfg, (dict, SimpleNamespace)):
            # Build with defaults if no explicit block provided
            pca_cfg = {
                'pca_init_file': None,
                'whiten': True,
                'noise_std': 0.0,
                'add_dequant_noise': False,
                'input_size': _get_from_ns_or_dict(config, 'input_size', (256, 256)),
                'patch_size': _get_from_ns_or_dict(config, 'patch_size', 16),
                'depth_to_seq': 1,
                'skip_pca': False,
            }
        if isinstance(pca_cfg, (dict, SimpleNamespace)):
            pcakw = dict(pca_cfg) if isinstance(pca_cfg, dict) else {k: getattr(pca_cfg, k) for k in dir(pca_cfg) if not k.startswith('_')}
            model.patch_pca = PatchPCA(
                pca_init_file=pcakw.get('pca_init_file'),
                whiten=bool(pcakw.get('whiten', True)),
                noise_std=float(pcakw.get('noise_std', 0.0)),
                add_dequant_noise=bool(pcakw.get('add_dequant_noise', False)),
                input_size=tuple(pcakw.get('input_size', getattr(config, 'input_size', (256, 256)))),
                patch_size=int(pcakw.get('patch_size', getattr(config, 'patch_size', 16))),
                depth_to_seq=int(pcakw.get('depth_to_seq', 1)),
                skip_pca=bool(pcakw.get('skip_pca', False)),
            ).to(device)
    except Exception:
        pass

    try:
        from src.latents.jet_adaptor import build_adaptor
        adaptor_cfg = _get_from_ns_or_dict(config, 'adaptor', None)
        H, W = kwargs.get('input_size', tuple(_get_from_ns_or_dict(config, 'input_size', (256, 256))))
        ps = int(kwargs.get('patch_size', _get_from_ns_or_dict(config, 'patch_size', 16)))
        grid_h, grid_w = (H // ps, W // ps)
        # For adaptor over patch latents, the channel dim must match full patch token dim
        full_token_dim = 3 * ps * ps

        # Build default adaptor when training_mode=='pca' and none provided
        if training_mode == 'pca' and not isinstance(adaptor_cfg, (dict, SimpleNamespace)):
            adaptor_cfg = {'kind': 'jet', 'depth': 8, 'block_depth': 2, 'emb_dim': 256, 'num_heads': 4}

        if isinstance(adaptor_cfg, (dict, SimpleNamespace)):
            ak = dict(adaptor_cfg) if isinstance(adaptor_cfg, dict) else {k: getattr(adaptor_cfg, k) for k in dir(adaptor_cfg) if not k.startswith('_')}
            kind = str(ak.get('kind', 'none') or 'none').lower()
            if kind != 'none':
                dim = int(ak.get('dim', full_token_dim))
                # Pass through latent_noise_dim
                model._latent_noise_dim = int(ak.get('latent_noise_dim', _get_from_ns_or_dict(config, 'latent_noise_dim', 0)))

                # Use shared jet config for adaptor params
                jet_params_for_adaptor = dict(jet_cfg) if isinstance(jet_cfg, dict) else {k: getattr(jet_cfg, k) for k in dir(jet_cfg) if not k.startswith('_')}

                model.adaptor = build_adaptor(kind, grid_h, grid_w, dim, **jet_params_for_adaptor)
                model.adaptor = model.adaptor.to(device)
        # If explicitly legacy, ensure no adaptor by default
        if training_mode != 'pca' and not isinstance(adaptor_cfg, (dict, SimpleNamespace)):
            model.adaptor = None
    except Exception:
        pass
    return model


