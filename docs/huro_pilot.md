# HuRo reconstruction pilot

Replace `POD_IP`, `POD_PORT`, `/path/to/ssh_private_key` and
`/path/to/mano_v1_2/models/` with your own connection and file details.
The commands use `/workspace/learning-from-ego` as the project folder on the pod.

Run all ten HuRo stages on the phone recordings in `data/personal_demonstrations/`: reconstruct hands and
camera movement, convert human motion into robot motion (retargeting), replace
the human arms visually, and export a dataset. Use HuRo's supplied Allex robot
for reconstruction and rendering. This does not convert the scene to LIBERO or train SmolVLA.

## First installation and trial

The first trial completed on the original three recordings. It produced four rendered
segments (two overlap in demo_01), with 716 frames marked valid, and exported an
Allex LeRobot V2 dataset. The videos were visually reviewed by the project author
and looked good. These counts describe the first trial, not the expanded dataset.
See the README for the subsequent training results.

Installation required cuDNN 9.8.0.87. The first rendering attempt hung because
its log reported missing `libGLU.so.1` alongside shader errors. Installing
`libglu1-mesa` and resuming stages 9–10 completed the run. This package is now
included below. Disk use was reported at about 70 GB before the final retry.

The initial missing-MANO error was resolved by uploading both hand model files
to the paths in section 4.

## 1. Prepare the pod

Use Ubuntu 24.04 and an RTX 3090 with 24 GB GPU memory. Run these commands
**inside the pod**, logged in as root:

```bash
nvidia-smi
df -h / /workspace
```

The driver must support CUDA 12.8. `nvidia-smi` reporting CUDA 13.0 is fine:
HuRo installs its own CUDA 12.8 toolkit. HuRo documents driver 580.65.06 as tested
and warns that branches newer than R580 break rendering.
See [HuRo setup](https://github.com/3587jjh/HuRo/blob/main/setup/README.md).

The direct installation below requires a filesystem that supports normal Linux permissions
and links. On Runpod, a Network Volume survives pod termination; an ordinary
pod Volume Disk survives stopping but is deleted on termination.

```bash
apt update
apt install -y tmux git curl wget ffmpeg build-essential cmake ninja-build \
  pkg-config libglib2.0-0
apt install -y libegl1 libopengl0 libgl1 libglu1-mesa
mkdir -p /workspace/learning-from-ego/data/personal_demonstrations
tmux new -s huro
```

`libglu1-mesa` supplies `libGLU.so.1`, which Isaac Sim needs for rendering.
Keep it in the installation even though the import checks can pass without it.

Run the remaining pod commands inside tmux. Press **Ctrl+B**, then **D** to
detach. After reconnecting by SSH, return with:

```bash
tmux attach -t huro
```
 
tmux protects against SSH disconnections, not pod termination. It cannot protect
an installation already running outside its session.

## 2. Copy the recordings

Run this in **your laptop's WSL terminal**, not inside the pod. The whole project
is not needed. The full pipeline uses HuRo's own runner:

```bash
scp -P POD_PORT -i /path/to/ssh_private_key \
  /home/preshen/Projects/learning-from-ego/data/personal_demonstrations/*.mp4 \
  root@POD_IP:/workspace/learning-from-ego/data/personal_demonstrations/
```

Update the IP and port in every transfer command when changing pods. `scp` uses
uppercase `-P` for its port.

## 3. Install HuRo

Back **inside the pod's tmux session**, install Miniforge if Conda is not already
available. These are first-install commands; skip them on an existing installation.

```bash
curl -fL https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh \
  -o /tmp/Miniforge3.sh &&
bash /tmp/Miniforge3.sh -b -p /workspace/miniforge3 &&
source /workspace/miniforge3/etc/profile.d/conda.sh
```

Clone the reviewed version and build it for the RTX 3090 only. The `8.6` setting
avoids compiling for all seven supported GPU architectures. Installation can
still take several hours, but can take about an hour if you setup for a single gpu model.

```bash
cd /workspace &&
git clone https://github.com/3587jjh/HuRo.git &&
cd HuRo &&
git checkout aeaae17f0fd51cc325b030bbef79ceb68a25199e &&
git submodule update --init --recursive &&
TORCH_CUDA_ARCH_LIST=8.6 bash setup/setup.sh env
```

The command above installs the environment and packages. Before compiling and
checking them, apply the cuDNN fix from our first run. cuDNN is NVIDIA's GPU
calculation library; JAX needs 9.8 or newer, but the package installation left
9.7.1 installed. Splitting setup into these steps avoids that known check failure:

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate huro
python -m pip install --no-deps "nvidia-cudnn-cu12==9.8.0.87"
cd /workspace/HuRo
env -u LD_LIBRARY_PATH TORCH_CUDA_ARCH_LIST=8.6 bash setup/setup.sh build &&
env -u LD_LIBRARY_PATH bash setup/setup.sh check
```

This deliberately overrides PyTorch's older cuDNN pin, as anticipated in HuRo's
installer. The component checks passed afterward. `env -u LD_LIBRARY_PATH`
removes the pod's CUDA library-path override for that command only. `check`
does not rebuild the installation.

Download the model weights and read the final report for any failures:

```bash
bash setup/download.sh
```

Already downloaded files are skipped. MANO requires a separate manual download
below. The stage-6 Qwen model (about 18 GB) downloads on first use; stages 1–5
do not need it. For later stages, keep its cache on persistent storage:

```bash
export HF_HOME=/workspace/.hf_cache
```

To check the rendering components, read NVIDIA's licence:

```bash
cat "$CONDA_PREFIX/lib/python3.11/site-packages/isaacsim/LICENSE.txt"
```

If you accept its terms:

```bash
export OMNI_KIT_ACCEPT_EULA=Y
env -u LD_LIBRARY_PATH bash setup/setup.sh check
```

Our successful check ended with `overlay OK` and `== all checks passed ==`.
The deprecation warnings did not prevent those checks from passing.

## 4. Add the MANO hand models

On your laptop, register at [MANO](https://mano.is.tue.mpg.de/) and download
**Models & Code → mano_v1_2.zip**. Extract it. Only `MANO_RIGHT.pkl` and
`MANO_LEFT.pkl` are needed. The Python 2.7 note applies to the bundled example
code; HuRo keeps its Python 3.11 environment.

**Inside the pod**, create the destination folders:

```bash
mkdir -p /workspace/HuRo/submodules/hawor/_DATA/data/mano
mkdir -p /workspace/HuRo/submodules/hawor/_DATA/data_left/mano_left
```

**On your laptop, in WSL**, copy the files from the Windows Downloads folder:

```bash
scp -P POD_PORT -i /path/to/ssh_private_key \
  /path/to/mano_v1_2/models/MANO_RIGHT.pkl \
  root@POD_IP:/workspace/HuRo/submodules/hawor/_DATA/data/mano/

scp -P POD_PORT -i /path/to/ssh_private_key \
  /path/to/mano_v1_2/models/MANO_LEFT.pkl \
  root@POD_IP:/workspace/HuRo/submodules/hawor/_DATA/data_left/mano_left/
```

**Inside the pod**, confirm both files arrived:

```bash
ls -lh /workspace/HuRo/submodules/hawor/_DATA/data/mano/MANO_RIGHT.pkl \
  /workspace/HuRo/submodules/hawor/_DATA/data_left/mano_left/MANO_LEFT.pkl
```

These are data files, not commands to execute. If either is missing, fix the
transfer before starting the runner. Passing installation checks does not confirm
these files are present.

## 5. Prepare all recordings

**Inside the pod**, convert the recordings to 30 fps with a 256-pixel shorter side:

```bash
cd /workspace/learning-from-ego
mkdir -p results/04_huro_all/clips

for video in data/personal_demonstrations/*.mp4; do
  output="results/04_huro_all/clips/$(basename "$video")"
  [ -f "$output" ] && continue
  ffmpeg -nostdin -n -i "$video" \
    -vf "fps=30,scale='if(gt(iw,ih),-2,256)':'if(gt(iw,ih),256,-2)'" \
    -c:v libx264 -crf 18 -an \
    "$output" || break
done
```

The loop processes every MP4 in the folder and skips already prepared clips, so
adding recordings does not stop it at the first existing file. `-n` also prevents
overwriting clips. Delete an incomplete prepared clip before retrying it. The original `demo_01` is about ten
seconds long; HuRo recommends roughly 30 seconds or more for camera calibration.
Short clips may fail. Looping the recording does not add the missing information.

## 6. Run all ten stages

If every recording has a prepared MP4 in `results/04_huro_all/clips`, skip
section 5. If you added recordings, run section 5 again to prepare the new files. Those MP4s are inputs and should look like the original recordings.
First complete section 4's check that both MANO files are present.

Inside the pod's tmux session:

```bash
source /workspace/miniforge3/etc/profile.d/conda.sh
conda activate huro
cd /workspace/HuRo
export HF_HOME=/workspace/.hf_cache
export OMNI_KIT_ACCEPT_EULA=Y
```

The licence variable records the acceptance made during installation. Set the
input folder and run all stages on GPU 0 using Allex:

```bash
sed -i \
  -e 's|^INPUT_DIR=.*|INPUT_DIR="/workspace/learning-from-ego/results/04_huro_all/clips"|' \
  -e 's/^GPUS=.*/GPUS=(0)/' \
  -e 's/^ROBOT=.*/ROBOT="allex"/' \
  -e 's/^FIRST_STAGE=.*/FIRST_STAGE=1/' \
  -e 's/^LAST_STAGE=.*/LAST_STAGE=10/' \
  run_pipeline.sh

set -o pipefail
env -u LD_LIBRARY_PATH bash run_pipeline.sh 2>&1 | tee -a /workspace/learning-from-ego/results/04_huro_all/full_pipeline.log
```

Stage 6 downloads the roughly 18 GB Qwen model on first use.
Stages 7–9 remove human arms, calculate robot movement, and render the robot.
Stage 10 exports the dataset.

Rerunning skips cached work. A dropped clip can have a `.done` marker without
usable outputs, so a final completion message does not guarantee every recording
produced a robot video. Inspect the log and output videos. After changing an input
or reconstruction settings, use a fresh input directory to avoid stale results.

### Resume after a rendering failure

If stages 1–8 completed and stage 9 failed, keep their outputs. In the same
activated environment, after fixing the reported error, run:

```bash
cd /workspace/HuRo
export HF_HOME=/workspace/.hf_cache
export OMNI_KIT_ACCEPT_EULA=Y
sed -i 's/^FIRST_STAGE=.*/FIRST_STAGE=9/' run_pipeline.sh
set -o pipefail
env -u LD_LIBRARY_PATH bash run_pipeline.sh 2>&1 | tee -a /workspace/learning-from-ego/results/04_huro_all/render_retry.log
```

This resumes rendering and dataset export using the input folder configured
above. For a new set of recordings, use the full section 6 settings again to
reset `FIRST_STAGE=1`.

Our first renderer hung during warmup and reported shader errors plus missing
`libGLU.so.1`. Installing `libglu1-mesa` fixed that run. If it happens again,
check the detailed logs under:

```text
/workspace/miniforge3/envs/huro/lib/python3.11/site-packages/isaacsim/kit/logs/Kit/Isaac-Sim Python/5.1/
```

Import checks alone do not confirm successful rendering; check that the output
videos exist and can be played.

## 7. Download and inspect the results

Run this **on your laptop in WSL**:

```bash
mkdir -p /home/preshen/Projects/learning-from-ego/results/04_huro_all

scp -r -P POD_PORT -i /path/to/ssh_private_key \
  root@POD_IP:/workspace/learning-from-ego/results/04_huro_all/. \
  /home/preshen/Projects/learning-from-ego/results/04_huro_all/
```

The videos to view are under:

```text
results/04_huro_all/clips_chunked/allex/overlay/video/
```

Each recording has a subfolder containing the rendered manipulation segments.
These videos show Allex in the recorded scene, not the LIBERO robot or desk.

Other outputs:

- `clips/`: prepared input videos, visually similar to the originals.
- `clips_hand/` and `clips_extr/`: estimated hand and camera measurements.
- `clips_chunked/original/video/`: extracted segments and videos with arms removed.
- `clips_chunked/allex/annot/`: calculated robot movements.
- `clips_lerobot/allex/`: exported datasets grouped by image resolution.
- `full_pipeline.log`: output and errors from the full run.

Check that the robot follows the hand motion and that the bowl and plate remain
visible. Once you are happy with the clips, continue below.

## 8. Prepare the training dataset locally

Run this **on your laptop in WSL, after HuRo has finished and the outputs have
been downloaded**. Use the project's `.venv`, not the HuRo Conda environment.
This step prepares data; it does not train a model or require the cloud GPU.

```bash
cd /home/preshen/Projects/learning-from-ego
source .venv/bin/activate

python scripts/prepare_huro.py \
  --source results/04_huro_all/clips_lerobot/allex/192x342 \
  --output-dir data/lerobot/huro_bowl_placement_15
```

The script converts HuRo's export into the format used by our training code and
computes the statistics used to scale the model inputs and actions. It keeps
the longest segment from each recording and uses the right-arm and hand movement.
It does not convert these movements into LIBERO actions.

The new folder preserves the original three-video dataset. The `15` refers to
the number of source recordings, not a guarantee of 15 usable training episodes.
Read the converter's summary and `source_episodes.json` in the output folder to
see which recordings were included. The script refuses to overwrite an existing
output folder; use a new name if you need to repeat the conversion.

## 9. Copy the prepared dataset to the training pod

After conversion succeeds, create the destination folder **on the pod**:

```bash
mkdir -p /workspace/learning-from-ego/data/lerobot
```

Then copy the dataset **from your laptop's WSL terminal**. If training on a
different pod, update the IP and port first.

```bash
scp -r -P POD_PORT -i /path/to/ssh_private_key \
  /home/preshen/Projects/learning-from-ego/data/lerobot/huro_bowl_placement_15 \
  root@POD_IP:/workspace/learning-from-ego/data/lerobot/
```

Copy to a fresh destination folder so that files from different conversions do
not get mixed together. The training pod needs the prepared dataset, project
training code, and training environment; the dataset transfer alone does not
set those up.

## 10. Copy the updated training script before training

`experiments/exp_04_human_pretrain.py` now selects
`huro_bowl_placement_15` for its human-data stage. Copy the updated script to the
training pod. Both experiments still use `bowl_placement` for LIBERO fine-tuning.

Use fresh results folders for the expanded-data experiment. Run the human-data
stage first, then start LIBERO fine-tuning from that new checkpoint. Finally,
evaluate the fine-tuned model using the same settings as experiment 3.
The README describes the matched comparison; its existing commands refer to
the earlier run, so update the dataset and checkpoint/output paths before using
them for this run.
