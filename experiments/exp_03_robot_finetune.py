"""Experiment 3: fine-tune SmolVLA base on robot bowl-placement demonstrations."""

import argparse
import os
from pathlib import Path
import shlex
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["train", "eval"])
    parser.add_argument("--dataset", type=Path, default=Path("data/lerobot/bowl_placement_3tasks"))
    parser.add_argument("--steps", type=int, default=5_000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--episodes", type=int, default=50, help="Episodes per task")
    parser.add_argument("--checkpoint", type=Path, default=Path(
        "results/03_robot_finetune/checkpoints/last/pretrained_model"
    ))
    args = parser.parse_args()

    # Let LeRobot download the model into the project folder when needed.
    os.environ.setdefault("HF_HUB_CACHE", str(Path("models/huggingface").resolve()))

    if args.mode == "train":
        args.output_dir = args.output_dir or Path("results/03_robot_finetune")
        command = [
            sys.executable, "-m", "lerobot.scripts.lerobot_train",
            "--policy.path=lerobot/smolvla_base",
            # Read the camera names and robot-state shape from our LIBERO dataset.
            "--policy.input_features=null",
            "--policy.push_to_hub=false",
            "--policy.device=cuda",
            # During evaluation, observe the scene again after each action.
            "--policy.n_action_steps=1",
            "--dataset.repo_id=local/bowl_placement_3tasks",
            f"--dataset.root={args.dataset}",
            f"--steps={args.steps}",
            f"--save_freq={args.steps}",
            # Match experiment 4's LIBERO learning-rate schedule explicitly.
            "--policy.optimizer_lr=0.0001",
            "--policy.scheduler_warmup_steps=50",
            "--policy.scheduler_decay_steps=30000",
            "--policy.scheduler_decay_lr=0.0000025",
            f"--batch_size={args.batch_size}",
            f"--seed={args.seed}",
            f"--output_dir={args.output_dir}",
        ]
    else:
        args.output_dir = args.output_dir or Path("results/03_robot_finetune_eval")
        os.environ.setdefault("MUJOCO_GL", "egl")

        # Evaluate all three bowl tasks with the same 342-action budget.
        command = [
            sys.executable, "-m", "lerobot.scripts.lerobot_eval",
            f"--policy.path={args.checkpoint}",
            "--env.type=libero",
            "--env.task=libero_spatial",
            # 0: between plate and ramekin; 2: table centre; 8: next to plate.
            "--env.task_ids=[0,2,8]",
            "--env.episode_length=342",
            "--eval.batch_size=10",
            f"--eval.n_episodes={args.episodes}",
            f"--seed={args.seed}",
            f"--output_dir={args.output_dir}",
        ]
        if args.output_dir.exists() and any(args.output_dir.iterdir()):
            parser.error("Output directory is not empty; choose a new --output-dir")
        args.output_dir.mkdir(parents=True, exist_ok=True)

    print(shlex.join(command), flush=True)
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
