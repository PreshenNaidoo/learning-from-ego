"""Keep the original three HuRo demonstrations and add the other new recordings."""

import argparse
from io import BytesIO
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from PIL import Image
from lerobot.datasets.lerobot_dataset import LeRobotDataset


ORIGINAL_CLIPS = {"demo_01", "demo_02", "demo_03"}


def prepare(original, new, destination):
    if destination.exists():
        raise FileExistsError(f"Destination already exists: {destination}")

    original_info = json.loads((original / "meta/info.json").read_text())
    new_info = json.loads((new / "meta/info.json").read_text())
    feature_names = ["observation.images.image", "observation.state", "action"]
    features = {name: original_info["features"][name] for name in feature_names}
    if original_info["fps"] != new_info["fps"] or any(
        features[name] != new_info["features"][name] for name in feature_names
    ):
        raise ValueError("The datasets must use the same frame rate and feature definitions")

    original_episodes = json.loads((original / "source_episodes.json").read_text())
    if {episode["clip_id"] for episode in original_episodes} != ORIGINAL_CLIPS:
        raise ValueError("The original dataset must contain demo_01, demo_02 and demo_03")

    dataset = LeRobotDataset.create(
        repo_id=f"local/{destination.name}", root=destination,
        fps=original_info["fps"], robot_type="allex_right",
        features=features, use_videos=False,
    )
    provenance = []
    seen = set()
    total_frames = 0

    # Copy whole episodes, with their images, movements and actual task labels.
    for source in (original, new):
        episodes = json.loads((source / "source_episodes.json").read_text())
        tasks = pq.read_table(source / "meta/tasks.parquet").to_pylist()
        instructions = {row["task_index"]: row["task"] for row in tasks}
        rows_by_episode = {}
        for file in sorted((source / "data").rglob("*.parquet")):
            for row in pq.read_table(file).to_pylist():
                rows_by_episode.setdefault(row["episode_index"], []).append(row)

        # Converted episode IDs follow provenance order, not the original HuRo IDs.
        for episode_index, episode in enumerate(episodes):
            clip = episode["clip_id"]
            if source == new and clip in ORIGINAL_CLIPS:
                continue
            if clip in seen:
                raise ValueError(f"Duplicate recording: {clip}")
            rows = sorted(rows_by_episode[episode_index], key=lambda row: row["frame_index"])
            for row in rows:
                with Image.open(BytesIO(row["observation.images.image"]["bytes"])) as image:
                    pixels = np.array(image.convert("RGB"))
                dataset.add_frame({
                    "observation.images.image": pixels,
                    "observation.state": np.asarray(row["observation.state"], dtype=np.float32),
                    "action": np.asarray(row["action"], dtype=np.float32),
                    "task": instructions[row["task_index"]],
                })
            dataset.save_episode()
            provenance.append({**episode, "source_dataset": str(source)})
            seen.add(clip)
            total_frames += len(rows)
            print(f"Added {clip} from {source.name}: {len(rows)} frames", flush=True)

    # Recompute statistics over the combined data instead of copying old statistics.
    dataset.finalize()
    (destination / "source_episodes.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(f"Saved {len(provenance)} episodes and {total_frames} frames to {destination}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", type=Path, default=Path("data/lerobot/huro_bowl_placement"))
    parser.add_argument("--new", type=Path, default=Path("data/lerobot/huro_bowl_placement_15"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/lerobot/huro_original3_plus_new"))
    args = parser.parse_args()
    prepare(args.original, args.new, args.output_dir)


if __name__ == "__main__":
    main()
