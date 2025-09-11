import itertools
import argparse
import os
import yaml
from pathlib import Path
import collections
import sys

# Import the main training function to call it directly
from ..train_flow import main as train_main

def _maybe_generate_study_plots(study_name: str, entity: str = None):
    try:
        import importlib
        mod = importlib.import_module(__package__ + '.plot_utils')
        _gen = getattr(mod, 'generate_study_plots', None)
        if _gen is None:
            raise AttributeError('generate_study_plots missing')
    except Exception:
        print("Plot generation skipped: plot_utils not available.")
        return
    _gen(study_name, entity=entity)

def main():
    """Parses arguments, generates and executes a series of training runs for an ablation study."""
    parser = argparse.ArgumentParser(description="Launch a Jet ablation study sequentially.")
    parser.add_argument("--dry", action="store_true", help="Print commands without executing")
    parser.add_argument("--study", type=str, default="vit_depth_sweep", help="Name of the study to run from the YAML config.")
    parser.add_argument("--sweep_cfg", type=Path, default=Path(__file__).with_name("ablations.yaml"), help="Path to YAML defining sweeps")
    # Allow overriding a few key baseline parameters from the CLI for quick tests
    parser.add_argument("--epochs", type=int, default=None, help="Override number of training epochs for all runs in the study.")
    parser.add_argument("--batch_size", type=int, default=None, help="Override batch size.")
    parser.add_argument("--dataset_subset_size", type=int, default=None, help="Use a random subset of the training data of this size.")
    parser.add_argument("--accelerator", type=str, default="auto", choices=["auto", "gpu", "tpu"], help="Acceleration backend to use for all runs.")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"], help="Device to use for GPU backend.")
    parser.add_argument("--wandb_entity", type=str, default=None, help="Weights & Biases entity (user or team name).")
    parser.add_argument("--num_workers", type=int, default=None, help="Override number of DataLoader workers.")
    parser.add_argument("--num_sample_images", type=int, default=None, help="Override number of images to sample at the end of each training run.")

    args = parser.parse_args()

    print(f"Loading study '{args.study}' from config file: {args.sweep_cfg}")
    cfg = _load_yaml(args.sweep_cfg)
    if args.study not in cfg:
        raise ValueError(f"Study '{args.study}' not found in {args.sweep_cfg}. Available studies: {list(cfg.keys())}")
    
    study_cfg = cfg[args.study]

    # Apply CLI overrides to the study's baseline config
    if args.epochs is not None:
        study_cfg['epochs'] = args.epochs
    if args.batch_size is not None:
        study_cfg['batch_size'] = args.batch_size
    if args.dataset_subset_size is not None:
        study_cfg['subset_size'] = args.dataset_subset_size
    if args.num_sample_images is not None:
        study_cfg['num_sample_images'] = args.num_sample_images

    # Add runner-level CLI args to fixed_params so they are passed to train.py
    fixed_params = {k: v for k, v in study_cfg.items() if not isinstance(v, list)}
    fixed_params['accelerator'] = args.accelerator
    fixed_params['device'] = args.device
    if args.num_workers is not None:
        fixed_params['num_workers'] = args.num_workers

    sweep_params = {k: v for k, v in study_cfg.items() if isinstance(v, list)}

    if not sweep_params:
        print("Warning: The selected study has no parameters to sweep (no lists found). Generating a single run.")
        param_combinations = [{}]
    else:
        # Generate Cartesian product of all swept parameter values
        keys, values = zip(*sweep_params.items())
        param_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

    print(f"Generating {len(param_combinations)} runs for study '{args.study}'...")
    
    # Sort runs to execute the largest models first, providing early feedback on VRAM limits.
    def get_model_size_heuristic(run_params):
        """A simple heuristic to estimate model size for sorting."""
        # Combine a run's specific combo with the fixed baseline params
        full_params = collections.ChainMap(run_params, fixed_params)
        
        couplings = full_params.get("coupling_layers", 1)
        
        # Determine parameters based on the backbone for the current run
        if full_params.get("backbone_kind") == "cnn":
            depth = full_params.get("cnn_block_depth", 1)
            dim = full_params.get("cnn_embed_dim", 1)
        else:  # 'vit' is the default
            depth = full_params.get("vit_depth", 1)
            dim = full_params.get("vit_embed_dim", 1)
            
        # Heuristic: complexity/memory scales with Layers * Depth * Dim^2
        return couplings * depth * (dim ** 2)

    param_combinations.sort(key=get_model_size_heuristic, reverse=True)

    original_argv = sys.argv.copy()

    for i, combo in enumerate(param_combinations):
        run_params = collections.ChainMap(combo, fixed_params)

        # Conditionally apply settings based on the backbone.
        # This is more robust than managing this in the YAML for all cases.
        if run_params.get("backbone_kind") == "cnn":
            # CNNs can be unstable with FP16; prefer TF32/BF16. Also restrict to channel-only couplings.
            run_params = run_params.new_child({
                "grad_clip_norm": run_params.get("grad_clip_norm", 1.0),
                "precision": run_params.get("precision", "tf32"),
                "ratio_M": 0,
            })
        else:  # For ViT and other potential backbones
            # ViT is stable; no grad clipping needed by default.
            run_params = run_params.new_child({
                "grad_clip_norm": run_params.get("grad_clip_norm", 0.0),
            })

        # Construct a unique, descriptive run name
        run_name_parts = [args.study]
        for key, value in combo.items():
            key_abbr = {
                "vit_depth": "vd", "cnn_block_depth": "cd",
                "vit_embed_dim": "ve", "cnn_embed_dim": "ce",
                "backbone_kind": "bk", "ratio_M": "M",
                "spatial_mode": "sm", "masking_mode": "mm",
                "actnorm": "an", "invertible_dense": "id"
            }.get(key, key)
            run_name_parts.append(f"{key_abbr}{value}")
        run_name = "_".join(run_name_parts)

        # Build the argument list for train_main
        cmd_args = ["flow/train.py"] # sys.argv[0] is the script name
        for key, value in run_params.items():
            arg_name = {
                "coupling_layers": "model_depth",
                "vit_depth": "model_block_depth",
                "cnn_block_depth": "model_block_depth",
                "vit_embed_dim": "model_emb_dim",
                "cnn_embed_dim": "model_emb_dim",
                "num_heads": "model_num_heads",
                "backbone_kind": "model_backbone",
                "ratio_M": "model_channel_repeat",
                "spatial_mode": "model_spatial_mode",
                "masking_mode": "model_masking_mode",
                "actnorm": "model_actnorm",
                "invertible_dense": "model_invertible_dense",
                "grad_checkpoint": "model_grad_checkpoint",
                "subset_size": "dataset_subset_size",
                "precision": "precision",
            }.get(key, key)
            
            # For argparse, boolean flags are handled by their presence.
            # Convert Python bools to a format argparse understands from strings.
            if value is None:
                continue
            if isinstance(value, bool):
                 cmd_args.append(f"--{arg_name}={str(value)}")
            else:
                 cmd_args.append(f"--{arg_name}={value}")


        cmd_args.append(f"--wandb_run_name={run_name}")
        cmd_args.append(f"--wandb_tags={args.study}")

        if 'subset_size' in run_params and run_params['subset_size'] is not None:
             cmd_args.append(f"--dataset_subset_size={run_params['subset_size']}")

        print("\n" + "="*80)
        print(f"Starting Run {i+1}/{len(param_combinations)}: {run_name}")
        print("Arguments:", " ".join(cmd_args[1:]))
        print("="*80)

        if args.dry:
            continue

        try:
            # Temporarily replace sys.argv to call train_main
            sys.argv = cmd_args
            train_main()
        except Exception as e:
            print(f"\nERROR: Run {run_name} failed with exception: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # Restore original sys.argv
            sys.argv = original_argv
    
    if args.dry:
        print(f"\nDRY RUN COMPLETE. Would have launched {len(param_combinations)} runs.")
        return

    print(f"\nAll {len(param_combinations)} runs for study '{args.study}' have completed.")

    # ------------------------------------------------------------------
    # Post-study analysis and plotting
    # ------------------------------------------------------------------
    print(f"\nGenerating plots for study '{args.study}'...")
    _maybe_generate_study_plots(args.study, entity=args.wandb_entity)
    print("Plot generation complete. Check your WandB project for a new run with the results.")


def _load_yaml(path: Path):
    """Loads a YAML file, raising an error if it contains duplicate keys."""
    class NoDuplicateLoader(yaml.SafeLoader):
        def _check_duplicate_key(self, key_node):
            if key_node.value in self.processing_duplicates:
                raise yaml.constructor.ConstructorError(f"Duplicate key '{key_node.value}' found at line {key_node.start_mark.line + 1}")
            self.processing_duplicates.append(key_node.value)

        def construct_mapping(self, node, deep=False):
            self.processing_duplicates = []
            return super().construct_mapping(node, deep)
    
    with open(path, "r") as f:
        return yaml.load(f, Loader=NoDuplicateLoader)


if __name__ == "__main__":
    main() 