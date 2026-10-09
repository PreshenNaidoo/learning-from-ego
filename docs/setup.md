# Setup and running

[Back to the project](../README.md)

Use Linux and Python 3.12 for preparation, training and evaluation. Run commands
from the project root. GPU training and simulation need a working NVIDIA setup.

```bash
apt update
apt install -y libegl1 libopengl0 libgl1 libglu1-mesa

python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
export MUJOCO_GL=egl
```

HuRo uses a separate Conda environment. See the [HuRo guide](huro_pilot.md)
for video processing and the installation fixes found during this project.

### Prepare the main datasets

Put the three selected task HDF5 files in `data/libero/libero_spatial/`.
Their names are listed explicitly in `scripts/prepare_libero.py`.
After HuRo completes, convert its export in the project environment:

```bash
python scripts/prepare_libero.py
python scripts/prepare_huro.py \
  --source results/04_huro_all/clips_lerobot/allex/192x342 \
  --output-dir data/lerobot/huro_bowl_placement_15
```

These create `bowl_placement_3tasks` and `huro_bowl_placement_15` under
`data/lerobot/`. Existing datasets are not overwritten. The name `15` refers to
the source recordings, not the number that survived processing.

### Run the 500 → 6,000 comparison

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

Run the reference methods separately:

```bash
python experiments/exp_01_baseline.py --trials 50 --seed 42
python experiments/exp_02_smolvla_libero.py \
  --episodes 50 --seed 42 --output-dir results/02_smolvla_libero_3tasks
```
