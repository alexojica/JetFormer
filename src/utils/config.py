from typing import Dict, Any


ALIASES = {
    # Training hyperparameters
    'lr': 'learning_rate',
    'epochs': 'num_epochs',
    'total_epochs': 'num_epochs',
    # Gradient clipping
    'grad_clip': 'grad_clip_norm',
    # Validation cadence
    'val_interval': 'val_every_epochs',
    # Optimizer / schedule
    'wd': 'weight_decay',
    'weightdecay': 'weight_decay',
    'warmup': 'warmup_percent',
    'warmup_pct': 'warmup_percent',
    'cosine': 'use_cosine',
    # CFG/drop labels naming parity
    'drop_labels_probability': 'cfg_drop_prob',
}


def normalize_config_keys(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Return a shallow-copied config dict with known aliases normalized.

    - Maps lr->learning_rate, epochs/total_epochs->num_epochs
    - Ensures types are basic Python types for YAML/argparse compatibility
    """
    out = dict(cfg or {})
    for alias, canon in ALIASES.items():
        if alias in out and canon not in out:
            out[canon] = out[alias]
    # Coerce some defaults if missing
    if 'num_epochs' not in out and 'epochs' in out:
        out['num_epochs'] = out['epochs']
    # Normalize boolean-like strings for use_cosine et al.
    for key in ['use_cosine']:
        if key in out and isinstance(out[key], str):
            v = out[key].lower()
            out[key] = v in ('1','true','yes','y')
    return out


