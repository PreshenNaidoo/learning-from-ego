# Setup and running

[Back to the project](../README.md)

Follow this guide in order to run the main three-task comparison and the two
reference experiments. The HuRo guide covers video processing and brings you
back here for dataset preparation and training. Ablations are not included.

The commands use one Linux GPU machine for the whole workflow, with the project
at `/workspace/learning-from-ego`. Commands run on that machine unless marked
**on your laptop**. They assume root access, as on the cloud pod used here.
On another Linux machine, use `sudo` for package installation and change the
`/workspace` paths if needed.

You need an NVIDIA GPU with at least 24 GB memory and about 150–200 GB of disk
space. This project used an RTX 3090. HuRo alone used about 70 GB during setup.
Use storage that supports normal Linux permissions and symbolic links.

The repository contains code and documentation. Recordings, prepared datasets
and checkpoints are not included. You can run Experiments 01–03 using public
models and LIBERO data. For Experiment 04, download
[my 15 original smartphone recordings](https://drive.google.com/drive/folders/1_qfCVt0DsNNct47sBkigbv2mTtqOswg7?usp=sharing)
to follow the same 15-video experiment, or use your own recordings. Extract the
download if Google Drive packages it as a ZIP and keep the original MP4 filenames.
The HuRo guide below explains how to copy them to the GPU machine and process them.
Reprocessing the recordings can produce different outputs, so exact scores may vary.

## 1. Clone the project and start tmux

```bash
apt update
apt install -y git curl wget tmux ffmpeg build-essential cmake ninja-build \
  pkg-config libglib2.0-0 libegl1 libopengl0 libgl1 libglu1-mesa
mkdir -p /workspace
cd /workspace
git clone https://github.com/PreshenNaidoo/learning-from-ego.git
cd learning-from-ego
tmux new -s learning-from-ego
```

Run the remaining commands inside tmux so work continues if SSH disconnects.
Press **Ctrl+B**, then **D** to detach. After reconnecting, run:

```bash
tmux attach -t learning-from-ego
```

Stopping or terminating the machine still stops the work. Check your provider's
storage policy before terminating it.

## 2. Create the training environment

Use Conda for both environments. `smolvla` holds this project's Python 3.12
packages. HuRo will create its own environment later. If Conda is already
installed, source its `etc/profile.d/conda.sh` instead of installing it again.
For a fresh machine:

```bash
curl -fL https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh \
  -o /tmp/Miniforge3.sh &&
bash /tmp/Miniforge3.sh -b -p /workspace/miniforge3 &&
source /workspace/miniforge3/etc/profile.d/conda.sh
```

Create and activate the training environment:

```bash
conda create -y -n smolvla python=3.12
conda activate smolvla
cd /workspace/learning-from-ego
export HF_HOME=/workspace/.hf_cache
python -m pip install -r requirements.txt
export MUJOCO_GL=egl
python -c "import torch; print(torch.zeros(1, device='cuda'))"
```

The last command should print a tensor on `cuda:0`. Resolve GPU access problems
before starting a long installation or training run. Seeing the card in
`nvidia-smi` alone does not confirm Python can use it.

## 3. Download and prepare the LIBERO demonstrations

Download the original HDF5 demonstrations from the
[LIBERO dataset](https://huggingface.co/datasets/yifengzhu-hf/LIBERO-datasets).
This downloads LIBERO Spatial. The preparation script selects tasks 0, 2 and 8,
with 50 demonstrations per task, and combines them into one training dataset.
You do not need a separate checkout of LIBERO's source code.

```bash
python - <<'PYTHON'
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="yifengzhu-hf/LIBERO-datasets",
    repo_type="dataset",
    allow_patterns="libero_spatial/*",
    local_dir="data/libero",
)
PYTHON

python scripts/prepare_libero.py
```

The prepared dataset is `data/lerobot/bowl_placement_3tasks`. Both Experiments 03
and 04 use it for robot fine-tuning.

## 4. Process your recordings and prepare the human dataset

For Experiment 04, follow the [HuRo guide](huro_pilot.md) now. It covers uploading
your MP4s, installing HuRo, processing them and inspecting the rendered videos.
Return here once all ten stages have finished and you have checked the outputs.
If you only want Experiments 01–03, skip the HuRo guide and the Experiment 04 commands.

Back on the GPU machine, switch from HuRo to the training environment:

```bash
source /workspace/miniforge3/etc/profile.d/conda.sh
conda activate smolvla
cd /workspace/learning-from-ego
export HF_HOME=/workspace/.hf_cache
export MUJOCO_GL=egl
ls results/04_huro_all/clips_lerobot/allex/
```

The export from my recordings was in `192x342`. If yours has another resolution,
use that folder in `--source` below. Prepare the dataset on this same machine:

```bash
python scripts/prepare_huro.py \
  --source results/04_huro_all/clips_lerobot/allex/192x342 \
  --output-dir data/lerobot/huro_bowl_placement_15
```

This converts HuRo's export into the training format, keeps the longest segment
from each recording and selects right-arm and hand movement. It does not turn
those movements into LIBERO robot actions. Check the printed summary and
`source_episodes.json` in the output folder to see which recordings were included.

The name `15` follows this project's dataset naming. It does not require exactly
15 recordings or mean every recording survived processing. Both preparation
scripts refuse to overwrite an existing dataset.

## 5. Train Experiments 03 and 04

Use new output folders for each run. These commands reproduce the main study's
training setup. They do not reproduce the earlier task-0-only 88% experiment.

```bash
python experiments/exp_03_robot_finetune.py train \
  --dataset data/lerobot/bowl_placement_3tasks \
  --steps 6000 --output-dir results/03_robot_only_6000

python experiments/exp_04_human_pretrain.py human \
  --dataset data/lerobot/huro_bowl_placement_15 \
  --steps 500 --output-dir results/04_human_15videos_500

python experiments/exp_04_human_pretrain.py robot \
  --dataset data/lerobot/bowl_placement_3tasks \
  --steps 6000 \
  --checkpoint results/04_human_15videos_500/checkpoints/last/pretrained_model \
  --output-dir results/04_human_500_robot_6000
```

Only the final checkpoint is saved. Once training finishes:

```bash
python experiments/exp_03_robot_finetune.py eval \
  --checkpoint results/03_robot_only_6000/checkpoints/last/pretrained_model \
  --output-dir results/03_robot_only_6000_eval

python experiments/exp_04_human_pretrain.py eval \
  --task-ids 0 2 8 \
  --checkpoint results/04_human_500_robot_6000/checkpoints/last/pretrained_model \
  --output-dir results/04_human_500_robot_6000_eval
```
  
Both default to 50 episodes per task and seed 42. Scores and per-episode outcomes
are written to `eval_info.json`, alongside evaluation recordings.

## 6. Run Experiments 01 and 02

Experiment 01 runs the hand-coded baseline on task 0. Experiment 02 evaluates
the published SmolVLA LIBERO model on tasks 0, 2 and 8 without further training:

```bash
python experiments/exp_01_baseline.py --trials 50 --seed 42
python experiments/exp_02_smolvla_libero.py \
  --episodes 50 --seed 42 --output-dir results/02_smolvla_libero_3tasks
```

## 7. Read or download the results

Each evaluation folder contains `eval_info.json` with success rates and episode
outcomes, alongside recorded evaluation videos. Training folders contain the
final model under `checkpoints/last/pretrained_model`.

To download the evaluation results, run this **on your laptop**. Replace
`POD_IP`, `POD_PORT` and `/path/to/ssh_private_key` with your connection details.
The destination is a new `results` folder in your current directory.

```bash
mkdir -p results
for folder in 01_baseline 02_smolvla_libero_3tasks 03_robot_only_6000_eval 04_human_500_robot_6000_eval; do
  scp -r -P POD_PORT -i /path/to/ssh_private_key \
    "root@POD_IP:/workspace/learning-from-ego/results/$folder" results/
done
```

Only include folders for experiments you ran. Download checkpoints separately
if you want to keep the trained models before removing the GPU machine.
