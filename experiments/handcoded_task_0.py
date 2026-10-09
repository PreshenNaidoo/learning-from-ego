import os

os.environ.setdefault("MUJOCO_GL", "egl")

import gymnasium as gym

from lerobot.envs.libero import create_libero_envs
from lerobot.utils.random_utils import set_seed

if __package__:
    from .video import run_sequence
else:
    from video import run_sequence

set_seed(42)


envs = create_libero_envs(
    task="libero_spatial",
    n_envs=1,
    episode_length=342,
    gym_kwargs={"task_ids": [0]},
    camera_name="agentview_image,robot0_eye_in_hand_image",
    env_cls=gym.vector.SyncVectorEnv,
)

env = envs["libero_spatial"][0]


sequence = [
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


# Tune this sequence on one starting scene before evaluating across episodes.
try:
    run_sequence(
        env,
        sequence,
        filename="videos/handcoded_task_0.mp4",
        fps=20,
    )
finally:
    env.close()
