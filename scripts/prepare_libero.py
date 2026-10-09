"""Combine LIBERO Spatial tasks 0, 2 and 8 for robot fine-tuning."""

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset


# Task IDs 0, 2 and 8 in LIBERO Spatial's default order.
TASKS = [
    "pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate",
    "pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate",
    "pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate",
]
SOURCE = Path("data/libero/libero_spatial")
DESTINATION = Path("data/lerobot/bowl_placement_3tasks")


def prepare(source, destination):
    # Keep an existing dataset safe from being overwritten.
    if destination.exists():
        raise FileExistsError(f"Destination already exists: {destination}")

    dataset = None
    total_episodes = 0
    for task in TASKS:
        with h5py.File(source / f"{task}_demo.hdf5", "r") as file:
            demonstrations = file["data"]

            # Sort by episode number: demo_2 should come before demo_10.
            episode_names = sorted(
                demonstrations,
                key=lambda name: int(name.split("_")[-1]),
            )

            task_info = json.loads(demonstrations.attrs["problem_info"])
            instruction = task_info["language_instruction"]
            if dataset is None:
                environment = json.loads(demonstrations.attrs["env_args"])["env_kwargs"]
                fps = environment["control_freq"]

                first_episode = demonstrations[episode_names[0]]
                image_shape = first_episode["obs/agentview_rgb"].shape[1:]

                # Tell LeRobot what each training example contains.
                features = {
                    "observation.images.image": {
                        "dtype": "image",
                        "shape": image_shape,
                        "names": ["height", "width", "channels"],
                    },
                    "observation.images.image2": {
                        "dtype": "image",
                        "shape": image_shape,
                        "names": ["height", "width", "channels"],
                    },
                    "observation.state": {
                        "dtype": "float32",
                        "shape": (8,),
                        "names": None,
                    },
                    "action": {
                        "dtype": "float32",
                        "shape": (7,),
                        "names": None,
                    },
                }

                dataset = LeRobotDataset.create(
                    repo_id="local/bowl_placement_3tasks",
                    root=destination,
                    fps=fps,
                    robot_type="panda",
                    features=features,
                    use_videos=False,
                )

            for name in episode_names:
                episode = demonstrations[name]
                observations = episode["obs"]
                actions = episode["actions"][:].astype(np.float32)

                # The robot state describes where the gripper is, which way it
                # points, and how far apart its two fingers are: 3 + 3 + 2 values.
                positions = observations["ee_pos"][:]
                orientations = observations["ee_ori"][:]
                gripper_positions = observations["gripper_states"][:]
                states = np.concatenate(
                    [positions, orientations, gripper_positions], axis=1
                ).astype(np.float32)

                for frame_index, action in enumerate(actions):
                    external_image = observations["agentview_rgb"][frame_index]
                    wrist_image = observations["eye_in_hand_rgb"][frame_index]

                    # LeRobot rotates LIBERO camera images by 180 degrees during
                    # evaluation. Training images need the same orientation.
                    external_image = np.rot90(external_image, k=2).copy()
                    wrist_image = np.rot90(wrist_image, k=2).copy()

                    dataset.add_frame({
                        "observation.images.image": external_image,
                        "observation.images.image2": wrist_image,
                        "observation.state": states[frame_index],
                        "action": action,
                        "task": instruction,
                    })

                dataset.save_episode()
                print(f"Converted {task}/{name}: {len(actions)} frames", flush=True)

            total_episodes += len(episode_names)
            print(f"{task}: {len(episode_names)} demonstrations", flush=True)

    dataset.finalize()
    print(f"Saved {total_episodes} demonstrations across {len(TASKS)} tasks to {destination}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE, help="Folder containing the LIBERO Spatial HDF5 files")
    parser.add_argument("--output-dir", type=Path, default=DESTINATION)
    args = parser.parse_args()
    prepare(args.source, args.output_dir)


if __name__ == "__main__":
    main()
