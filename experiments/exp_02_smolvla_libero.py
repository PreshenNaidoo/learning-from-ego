"""Experiment 2: evaluate the published LIBERO SmolVLA model without fine-tuning."""

import argparse
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", "--trials", type=int, default=50, help="Episodes per task")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-steps", type=int, default=342)
    parser.add_argument("--output-dir", type=Path, default=Path("results/02_smolvla_libero"))
    args = parser.parse_args()

    # Avoid replacing results from an earlier run.
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error("Output directory is not empty; choose a new --output-dir")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Render without a window and keep downloaded models in the project folder.
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("HF_HUB_CACHE", str(Path("models/huggingface").resolve()))

    # Evaluate the same three bowl tasks as experiments 3 and 4.
    # Allow 342 actions, matching the full hand-coded sequence.
    command = [
        sys.executable, "-m", "lerobot.scripts.lerobot_eval",
        "--policy.path=HuggingFaceVLA/smolvla_libero",
        "--policy.device=cuda",
        "--env.type=libero",
        "--env.task=libero_spatial",
        # 0: between plate and ramekin; 2: table centre; 8: next to plate.
        "--env.task_ids=[0,2,8]",
        f"--env.episode_length={args.max_steps}",
        "--eval.batch_size=10",
        f"--eval.n_episodes={args.episodes}",
        f"--seed={args.seed}",
        f"--output_dir={args.output_dir}",
    ]

    # LeRobot loads the model, runs the episodes, and saves scores and recordings.
    subprocess.run(command, check=True)
    print(f"Results: {args.output_dir / 'eval_info.json'}")


if __name__ == "__main__":
    main()
