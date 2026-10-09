from pathlib import Path

import cv2
import numpy as np
import imageio.v2 as imageio


def combine_views(external, wrist, label=None):
    """
    Put the external and wrist camera views side-by-side.
    external, wrist:
        numpy arrays shaped (H, W, 3), RGB

    label:
        optional text shown at the top
    """

    # Make sure both images have the same height
    if external.shape[0] != wrist.shape[0]:
        wrist = cv2.resize(wrist, (wrist.shape[1], external.shape[0]))

    combined = np.concatenate([external, wrist], axis=1)

    # OpenCV expects BGR when drawing text
    combined_bgr = cv2.cvtColor(combined, cv2.COLOR_RGB2BGR)

    # Camera labels
    cv2.putText(
        combined_bgr,
        "External view",
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        combined_bgr,
        "Wrist view",
        (external.shape[1] + 10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    if label:
        cv2.putText(
            combined_bgr,
            label,
            (10, combined.shape[0] - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    return cv2.cvtColor(combined_bgr, cv2.COLOR_BGR2RGB)


def save_mp4(frames, filename, fps=15):
    """
    Save RGB frames as an MP4.
    """

    Path(filename).parent.mkdir(parents=True, exist_ok=True)

    writer = imageio.get_writer(filename,  fps=fps, codec="libx264")

    for frame in frames:
        writer.append_data(frame)

    writer.close()



def run_sequence(
    env,
    sequence,
    filename,
    fps=15,
    start_pause=5,
    end_pause=5,
):
    """
    Run a sequence of actions and save one side-by-side video.
    sequence format:
    [
        {
            "label": "move forward",
            "action": [0.2, 0, 0, 0, 0, 0, -1],
            "steps": 20,
        },
        ...
    ]
    """

    observation, info = env.reset(seed=42)

    frames = []
    success = False
    done = False
    action_count = 0

    # Add a few still frames at the beginning
    for _ in range(start_pause):
        external = observation["pixels"]["image"][0].copy()
        wrist = observation["pixels"]["image2"][0].copy()

        frames.append(combine_views(external, wrist, label="start"))

    for phase in sequence:
        label = phase["label"]
        action_values = phase["action"]
        steps = phase["steps"]

        action = np.array([action_values], dtype=np.float32)

        print(f"{label}: action={action_values}, steps={steps}")

        for _ in range(steps):
            observation, reward, terminated, truncated, info = env.step(action)

            external = observation["pixels"]["image"][0].copy()
            wrist = observation["pixels"]["image2"][0].copy()

            frames.append(combine_views(external, wrist, label=label))
            action_count += 1

            # Stop before another step can reset an episode that has ended.
            success = bool(info["is_success"][0])
            done = success or bool(terminated[0]) or bool(truncated[0])
            if done:
                break

        if done:
            break

    # Add a few still frames at the end
    for _ in range(end_pause):
        external = observation["pixels"]["image"][0].copy()
        wrist = observation["pixels"]["image2"][0].copy()

        frames.append(combine_views(external, wrist, label="end"))

    save_mp4(frames, filename, fps=fps)

    status = "SUCCESS" if success else "FAIL"
    print(f"{status} | seed=42 | actions={action_count}")
    print(f"Saved video to {filename}")

    return observation
