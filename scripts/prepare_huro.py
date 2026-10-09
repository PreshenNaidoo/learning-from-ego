"""Convert HuRo's rendered Allex demonstrations for SmolVLA pretraining."""

import argparse
import json
from pathlib import Path

import av
import numpy as np
import pyarrow.parquet as pq
from lerobot.datasets.lerobot_dataset import LeRobotDataset


SOURCE = Path("results/04_huro_all/clips_lerobot/allex/192x342")
DESTINATION = Path("data/lerobot/huro_bowl_placement")


def prepare(source, destination, clips=None):
    if destination.exists():
        raise FileExistsError(f"Destination already exists: {destination}")

    info = json.loads((source / "meta/info.json").read_text())
    episodes = [json.loads(line) for line in (source / "meta/episodes.jsonl").read_text().splitlines()]
    tasks = {row["task_index"]: row["task"] for row in (
        json.loads(line) for line in (source / "meta/tasks.jsonl").read_text().splitlines()
    )}

    # demo_01 has overlapping segments. Keep the longest segment per recording
    # so that one recording is not counted twice. Equal lengths keep the first.
    selected = {}
    for episode in episodes:
        clip = episode["clip_id"]
        if clips is not None and clip not in clips:
            continue
        if clip not in selected or episode["length"] > selected[clip]["length"]:
            selected[clip] = episode

    if clips is not None:
        missing = set(clips) - selected.keys()
        if missing:
            raise ValueError(f"Recordings not found in HuRo export: {sorted(missing)}")

    # HuRo tracked only the right hand in this pilot. Keep its seven arm joints
    # and fifteen finger joints; do not invent LIBERO gripper commands.
    state_names = info["features"]["observation.state"]["names"]
    action_names = info["features"]["action"]["names"]
    indices = [i for i, name in enumerate(state_names) if name.startswith("R_")]
    assert len(indices) == 22, "Expected 22 Allex right-arm and hand joints"
    features = {
        "observation.images.image": {
            "dtype": "image", "shape": tuple(info["features"]["observation.images.zed_left"]["shape"]),
            "names": ["height", "width", "channels"],
        },
        "observation.state": {
            "dtype": "float32", "shape": (22,), "names": [state_names[i] for i in indices],
        },
        "action": {
            "dtype": "float32", "shape": (22,), "names": [action_names[i] for i in indices],
        },
    }
    dataset = LeRobotDataset.create(
        repo_id="local/huro_bowl_placement", root=destination, fps=info["fps"],
        robot_type="allex_right", features=features, use_videos=False,
    )
    provenance = []
    for episode in selected.values():
        index = episode["episode_index"]
        chunk = index // info["chunks_size"]
        paths = {"episode_index": index, "episode_chunk": chunk,
                 "video_key": "observation.images.zed_left"}
        rows = pq.read_table(source / info["data_path"].format(**paths)).to_pylist()
        video = source / info["video_path"].format(**paths)
        count = 0
        with av.open(str(video)) as container:
            for frame in container.decode(video=0):
                row = rows[count]
                state = np.asarray(row["observation.state"], dtype=np.float32)[indices]
                action = np.asarray(row["action"], dtype=np.float32)[indices]
                if not np.isfinite(state).all() or not np.isfinite(action).all():
                    raise ValueError(f"Non-finite joint values in {video}, frame {count}")
                dataset.add_frame({
                    "observation.images.image": frame.to_ndarray(format="rgb24"),
                    "observation.state": state,
                    "action": action,
                    "task": tasks[int(row["task_index"])],
                })
                count += 1
        if count != len(rows):
            raise ValueError(f"Video/table length mismatch: {video}: {count} vs {len(rows)}")
        dataset.save_episode()
        provenance.append(episode)
        print(f"Converted {episode['clip_id']}/{episode['seg_name']}: {count} frames", flush=True)

    # LeRobot writes its current dataset format and computes normalization stats.
    dataset.finalize()
    (destination / "source_episodes.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(f"Saved {len(provenance)} episodes to {destination}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DESTINATION)
    parser.add_argument("--clips", nargs="+", help="Recording names without .mp4; default: all recordings")
    args = parser.parse_args()
    prepare(args.source, args.output_dir, args.clips)


if __name__ == "__main__":
    main()
