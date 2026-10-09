"""Experiment 1: evaluate the fixed, hand-coded movement sequence."""

import argparse
import json
import os
from pathlib import Path

import numpy as np

if __package__:
    from .video import combine_views, save_mp4
else:
    from video import combine_views, save_mp4


BASELINE_SEQUENCE = [
    {
        "label": "open gripper",
        "action": [0, 0, 0, 0, 0, 0, -1],
        "steps": 10,
    },

    {
        "label": "move forward",
        "action": [0.2, 0, 0, 0, 0, 0, -1],
        "steps": 50,
    },

    {
        "label": "move right",
        "action": [0, 0.2, 0, 0, 0, 0, -1],
        "steps": 62,
    },

    {
        "label": "move down",
        "action": [0, 0, -0.2, 0, 0, 0, -1],
        "steps": 90,
    },

    {
        "label": "close gripper",
        "action": [0, 0, 0, 0, 0, 0, 1],
        "steps": 15,
    },

    {
        "label": "lift",
        "action": [0, 0, 0.2, 0, 0, 0, 1],
        "steps": 30,
    },

    {
        "label": "move forward",
        "action": [0.2, 0, 0, 0, 0, 0, 1],
        "steps": 50,
    },

    {
        "label": "move down",
        "action": [0, 0, -0.2, 0, 0, 0, 1],
        "steps": 20,
    },

    {
        "label": "release bowl",
        "action": [0, 0, 0, 0, 0, 0, -1],
        "steps": 15,
    },
]


def run_trial(env, seed, video_filename):
    observation, info = env.reset(seed=seed)

    frames = []
    success = False
    done = False

    for phase in BASELINE_SEQUENCE:
        action = np.array([phase["action"]], dtype=np.float32)

        for _ in range(phase["steps"]):
            observation, reward, terminated, truncated, info = env.step(action)

            external = observation["pixels"]["image"][0].copy()
            wrist = observation["pixels"]["image2"][0].copy()

            frames.append(combine_views(external, wrist, label=phase["label"]))

            # LIBERO tells us whether the actual task objective is satisfied.
            success = bool(info["is_success"][0])
            done = success or bool(terminated[0]) or bool(truncated[0])
            if done:
                break

        if done:
            break

    save_mp4(frames, video_filename, fps=20)

    return success

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", "--episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42, help="First trial seed")
    parser.add_argument("--output-dir", type=Path, default=Path("results/01_baseline"))
    args = parser.parse_args()
    if args.trials < 1:
        parser.error("--trials must be at least 1")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error("Output directory is not empty; choose a new --output-dir")

    os.environ.setdefault("MUJOCO_GL", "egl")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Import simulation packages only when running the experiment.
    import gymnasium as gym
    from lerobot.envs.libero import create_libero_envs
    from lerobot.utils.random_utils import set_seed

    set_seed(args.seed)

    envs = create_libero_envs(
        task="libero_spatial",
        n_envs=1,
        episode_length=342,
        gym_kwargs={"task_ids": [0]},
        camera_name="agentview_image,robot0_eye_in_hand_image",
        env_cls=gym.vector.SyncVectorEnv,
    )
    env = envs["libero_spatial"][0]
    results = []
    try:
        for trial in range(args.trials):
            seed = args.seed + trial
            success = run_trial(env, seed=seed, video_filename=args.output_dir / f"baseline_trial_{trial:02d}.mp4")
            results.append(success)
            status = "SUCCESS" if success else "FAIL"
            print(f"Trial {trial:02d} | seed={seed} | {status}")

        # Save the score and individual outcomes for comparison with model runs.
        report = {
            "task": "libero_spatial",
            "task_id": 0,
            "episode_length": 342,
            "n_episodes": args.trials,
            "seed": args.seed,
            "seeds": list(range(args.seed, args.seed + args.trials)),
            "successes": results,
            "pc_success": 100 * sum(results) / args.trials,
        }
        (args.output_dir / "eval_info.json").write_text(json.dumps(report, indent=2) + "\n")
        print(f"Successes: {sum(results)}/{args.trials}")
        print(f"Success rate: {100 * sum(results) / args.trials:.1f}%")
    finally:
        env.close()


if __name__ == "__main__":
    main()
