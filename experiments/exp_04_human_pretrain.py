"""Experiment 4: HuRo robot-movement pretraining, robot fine-tuning, then evaluation."""

import argparse
import os
from pathlib import Path
import shlex
import subprocess
import sys


HUMAN_RUN = Path("results/04_human_pretrain")
ROBOT_RUN = Path("results/04_robot_finetune")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["human", "robot", "eval"])
    parser.add_argument("--steps", type=int, help="Default: 500 human steps or 5,000 robot steps")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--checkpoint", type=str, help="Override the starting model or evaluation checkpoint")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--dataset", type=Path, help="Override the training dataset folder")
    parser.add_argument("--task-ids", type=int, nargs="+", default=[0, 2, 8], help="LIBERO task IDs to evaluate")
    parser.add_argument("--episodes", type=int, default=50, help="Episodes per task")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    os.environ.setdefault("HF_HUB_CACHE", str(Path("models/huggingface").resolve()))

    if args.mode == "eval":
        args.output_dir = args.output_dir or Path("results/04_robot_finetune_eval")
        args.checkpoint = args.checkpoint or (ROBOT_RUN / "checkpoints/last/pretrained_model")
        os.environ.setdefault("MUJOCO_GL", "egl")

        # Evaluate the selected bowl tasks with the same 342-action budget.
        command = [
            sys.executable, "-m", "lerobot.scripts.lerobot_eval",
            f"--policy.path={args.checkpoint}",
            "--env.type=libero",
            "--env.task=libero_spatial",
            # 0: between plate and ramekin; 2: table centre; 8: next to plate.
            f"--env.task_ids={args.task_ids}",
            "--env.episode_length=342",
            "--eval.batch_size=10",
            f"--eval.n_episodes={args.episodes}",
            f"--seed={args.seed}",
            f"--output_dir={args.output_dir}",
        ]
        if args.output_dir.exists() and any(args.output_dir.iterdir()):
            parser.error("Output directory is not empty; choose a new --output-dir")
        args.output_dir.mkdir(parents=True, exist_ok=True)
    else:
        human = args.mode == "human"
        dataset_name = "huro_bowl_placement_15" if human else "bowl_placement_3tasks"
        dataset = args.dataset or Path("data/lerobot") / dataset_name
        checkpoint = args.checkpoint or (
            "lerobot/smolvla_base" if human else str(HUMAN_RUN / "checkpoints/last/pretrained_model")
        )
        output = args.output_dir or (HUMAN_RUN if human else ROBOT_RUN)
        steps = args.steps if args.steps is not None else (500 if human else 5_000)
        command = [
            sys.executable, "-m", "lerobot.scripts.lerobot_train",
            f"--policy.path={checkpoint}",
            # Allex and LIBERO have different cameras, states and action meanings.
            # LeRobot rebuilds their feature definitions and normalization statistics.
            "--policy.input_features=null",
            "--policy.push_to_hub=false", "--policy.device=cuda",
            "--policy.n_action_steps=1",
            f"--dataset.repo_id=local/{dataset.name}", f"--dataset.root={dataset}",
            f"--steps={steps}", f"--batch_size={args.batch_size}",
            f"--save_freq={steps}",
            "--policy.optimizer_lr=0.0001",
            # Match experiment 3 and the successful earlier experiment 4 runs.
            "--policy.scheduler_warmup_steps=50",
            "--policy.scheduler_decay_steps=30000",
            "--policy.scheduler_decay_lr=0.0000025",
            f"--seed={args.seed}", f"--output_dir={output}",
        ]

    print(shlex.join(command), flush=True)
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
