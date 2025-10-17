from typing import Dict, Any
from types import SimpleNamespace


def _dict_to_sns(d: Dict[str, Any]) -> SimpleNamespace:
    """Recursively convert a dictionary to a SimpleNamespace."""
    if not isinstance(d, dict):
        return d
    return SimpleNamespace(**{k: _dict_to_sns(v) for k, v in d.items()})


def get_nested_config(config: Dict[str, Any]) -> SimpleNamespace:
    """Converts a dictionary to a nested SimpleNamespace, allowing attribute access."""
    sns = _dict_to_sns(config)
    # Add a .get() method for safe access with defaults, mimicking dict behavior
    def _sns_get(namespace, key, default=None):
        parts = key.split('.')
        curr = namespace
        for part in parts:
            if not isinstance(curr, SimpleNamespace) or not hasattr(curr, part):
                return default
            curr = getattr(curr, part)
        return curr
    
    # Bind the get method to the root namespace instance
    import types
    sns.get = types.MethodType(lambda self, key, default=None: _sns_get(self, key, default), sns)
    return sns


def migrate_and_normalize_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Migrates flat-key config to a JAX-like nested structure and normalizes keys."""
    
    # This is where the magic happens. We will build a new, structured config.
    new_cfg = {}

    # Top-level training settings
    new_cfg['num_epochs'] = cfg.get('num_epochs', cfg.get('epochs', cfg.get('total_epochs')))
    new_cfg['torch_compile'] = cfg.get('torch_compile', False)
    training_mode = cfg.get('jetformer_training_mode', 'pca')
    new_cfg['jetformer_training_mode'] = training_mode
    new_cfg['advanced_metrics'] = cfg.get('advanced_metrics', True)
    new_cfg['batch_size'] = cfg.get('batch_size')
    new_cfg['grad_accum_steps'] = cfg.get('grad_accum_steps', 1)

    # Input/Dataset
    new_cfg['input'] = {
        'dataset': cfg.get('dataset'),
        'input_size': cfg.get('input_size'),
        'num_classes': cfg.get('num_classes'),
        'num_workers': cfg.get('num_workers'),
        'max_samples': cfg.get('max_samples'),
        'random_flip_prob': cfg.get('random_flip_prob'),
        'class_token_length': cfg.get('class_token_length'),
        'ignore_pad': cfg.get('ignore_pad', False)
    }

    # Model
    new_cfg['model'] = {
        'width': cfg.get('d_model'),
        'depth': cfg.get('n_layers'),
        'mlp_dim': cfg.get('d_ff'),
        'num_heads': cfg.get('n_heads'),
        'num_kv_heads': cfg.get('n_kv_heads'),
        'vocab_size': cfg.get('vocab_size'),
        'bos_id': cfg.get('bos_id'),
        'boi_id': cfg.get('boi_id'),
        'nolabel_id': cfg.get('nolabel_id'),
        'num_mixtures': cfg.get('num_mixtures'),
        'dropout': cfg.get('dropout'),
        'drop_labels_probability': cfg.get('cfg_drop_prob'),
        'head_dtype': 'bfloat16' if cfg.get('use_bfloat16_img_head') else 'float32',
        'remat_policy': 'checkpoint' if cfg.get('grad_checkpoint_transformer') else 'none',
        'num_vocab_repeats': cfg.get('num_vocab_repeats'),
        'scale_tol': cfg.get('scale_tol', 1e-6),
        'causal_mask_on_prefix': cfg.get('causal_mask_on_prefix', True),
        'untie_output_vocab': cfg.get('untie_output_vocab', False),
        'per_modality_final_norm': cfg.get('per_modality_final_norm', False),
        'right_align_inputs': cfg.get('right_align_inputs', True),
        'strict_special_ids': cfg.get('strict_special_ids', True),
        'use_boi_token': cfg.get('use_boi_token', True),
        'max_seq_len': cfg.get('max_seq_len'), # For RoPE
        'rope_skip_pad': cfg.get('rope_skip_pad', True)
    }

    # Patch PCA
    pca_cfg = cfg.get('patch_pca', {})
    new_cfg['patch_pca'] = {
        'model': {
            'depth_to_seq': pca_cfg.get('depth_to_seq', 1),
            'input_size': cfg.get('input_size'),
            'patch_size': cfg.get('patch_size'),
            'codeword_dim': cfg.get('image_ar_dim'),
            'noise_std': pca_cfg.get('noise_std', 0.0),
            'add_dequant_noise': pca_cfg.get('add_dequant_noise', False),
            'skip_pca': pca_cfg.get('skip_pca', False),
            'pca_init_file': pca_cfg.get('pca_init_file'),
            'whiten': pca_cfg.get('whiten', True),
        }
    }
    new_cfg['model']['out_dim'] = new_cfg['patch_pca']['model']['codeword_dim']


    # Adaptor/Jet
    jet_cfg = cfg.get('jet', {})
    adaptor_cfg = cfg.get('adaptor', {})
    new_cfg['use_adaptor'] = training_mode == 'pca' or cfg.get('use_adaptor', False)
    new_cfg['adaptor'] = {
        'model': {
            'depth': jet_cfg.get('depth'),
            'block_depth': jet_cfg.get('block_depth'),
            'emb_dim': jet_cfg.get('emb_dim'),
            'num_heads': jet_cfg.get('num_heads'),
            'actnorm': jet_cfg.get('actnorm', False),
            'invertible_dense': jet_cfg.get('invertible_dense', False),
            'flow_grad_checkpoint': cfg.get('flow_grad_checkpoint', False)
        },
        'latent_noise_dim': adaptor_cfg.get('latent_noise_dim', cfg.get('latent_noise_dim', 0)),
        'kind': adaptor_cfg.get('kind', 'jet') if new_cfg['use_adaptor'] else 'none'
    }

    # Optimizer & Schedule
    new_cfg['optimizer'] = {
        'lr': cfg.get('learning_rate', cfg.get('lr')),
        'wd': cfg.get('weight_decay', cfg.get('wd')),
        'b1': cfg.get('opt_b1'),
        'b2': cfg.get('opt_b2'),
        'grad_clip_norm': cfg.get('grad_clip_norm', 1.0),
    }
    
    ema_cfg = cfg.get('ema', {})
    new_cfg['ema_decay'] = ema_cfg.get('decay', 0.0) if ema_cfg.get('enabled', False) else 0.0

    new_cfg['schedule'] = {
        'warmup_percent': cfg.get('warmup_percent', cfg.get('warmup_pct')),
        'decay_type': 'cosine' if cfg.get('use_cosine', True) else 'linear'
    }

    # Training curriculum
    new_cfg['training'] = {
        'input_noise_std': cfg.get('input_noise_std', 0.0),
        'noise_scale': cfg.get('noise_scale'),
        'noise_min': cfg.get('noise_min', 0.0),
        'text_prefix_prob': cfg.get('text_prefix_prob', 0.5),
        'loss_on_prefix': cfg.get('loss_on_prefix', True),
        'stop_grad_nvp_prefix': cfg.get('stop_grad_nvp_prefix', False),
        'rgb_noise_on_image_prefix': cfg.get('rgb_noise_on_image_prefix', True),
        'text_loss_weight': cfg.get('text_loss_weight', 1.0),
        'image_loss_weight': cfg.get('image_loss_weight', 1.0),
    }

    # Sampling / Eval
    new_cfg['sampling'] = {
        'cfg_strength': cfg.get('cfg_strength', 4.0),
        'cfg_mode': cfg.get('cfg_mode', 'interp'),
    }

    new_cfg['eval'] = {
        'sample_every_epochs': cfg.get('sample_every_epochs', 0),
        'sample_every_batches': cfg.get('sample_every_batches', 0),
        'val_every_epochs': cfg.get('val_every_epochs', 1),
        'eval_no_rgb_noise': cfg.get('eval_no_rgb_noise', True),
        'fid_every_epochs': cfg.get('fid_every_epochs', 0),
        'is_every_epochs': cfg.get('is_every_epochs', 0),
        'fid_is_num_samples': cfg.get('fid_is_num_samples', 0),
    }

    # W&B
    new_cfg['wandb'] = {
        'enabled': cfg.get('wandb', False),
        'offline': cfg.get('wandb_offline', False),
        'project': cfg.get('wandb_project'),
        'run_name': cfg.get('wandb_run_name'),
        'tags': cfg.get('wandb_tags', []),
        'run_id': cfg.get('wandb_run_id')
    }

    # Accelerator
    new_cfg['accelerator'] = {
        'name': cfg.get('accelerator', 'auto'),
        'device': cfg.get('device', 'auto'),
        'precision': cfg.get('precision', 'bf16'),
        'distributed': cfg.get('distributed', False)
    }
    
    # Remove None values to avoid clutter
    def clean_nones(d):
        if not isinstance(d, dict):
            return d
        return {k: clean_nones(v) for k, v in d.items() if v is not None}

    return clean_nones(new_cfg)


