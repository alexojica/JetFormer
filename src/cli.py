import argparse
import sys


def main():
    parser = argparse.ArgumentParser(description="Unified JetFormer CLI")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("train", help="Train JetFormer (YAML + CLI overrides)")
    sub.add_parser("train-flow", help="Train Jet Flow model (CLI)")
    sub.add_parser("eval", help="Evaluation and sampling")
    sub.add_parser("ablation", help="Run ablations")

    # passthrough args after subcommand
    args, rest = parser.parse_known_args()

    if args.cmd == "train":
        from src.train import main as train_main
        sys.argv = [sys.argv[0]] + rest
        train_main()
    elif args.cmd == "train-flow":
        from src.flow.train import main as flow_main
        sys.argv = [sys.argv[0]] + rest
        flow_main()
    elif args.cmd == "eval":
        from src.eval.run_eval import main as eval_main
        sys.argv = [sys.argv[0]] + rest
        eval_main()
    elif args.cmd == "ablation":
        from src.flow.ablation_runner import main as ablation_main
        sys.argv = [sys.argv[0]] + rest
        ablation_main()


if __name__ == "__main__":
    main()


