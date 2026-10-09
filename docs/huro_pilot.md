# Process phone recordings with HuRo

Start with [Setup and running](setup.md). Complete sections 1–3 there before
following this guide. You should have the repository cloned, system packages
installed, a working GPU and the `smolvla` Conda environment.

This guide uses the same GPU machine, project folder and tmux session. HuRo
creates a separate `huro` environment for reconstructing human movement and
rendering the Allex robot in your scene. After processing, return to the setup
guide to convert the output and train the VLA.

These installation commands were used with an RTX 3090 and NVIDIA driver
580.65.06. The GPU architecture setting `8.6` below is for that card. For another
GPU, consult [HuRo's setup guide](https://github.com/3587jjh/HuRo/blob/aeaae17f0fd51cc325b030bbef79ceb68a25199e/setup/README.md).
The `libglu1-mesa` package installed in the setup guide is needed for rendering.

## 1. Copy your recordings

First create the destination **on the GPU machine**, in the tmux session from
the setup guide:

```bash
mkdir -p /workspace/learning-from-ego/data/personal_demonstrations
```

Download [my 15 original smartphone recordings](https://drive.google.com/drive/folders/1_qfCVt0DsNNct47sBkigbv2mTtqOswg7?usp=sharing)
on your laptop. Extract the ZIP if needed and keep the original filenames.
Use the folder containing the `.mp4` files as `/path/to/recordings` in the command
below. This copies the recordings into `data/personal_demonstrations/` on the GPU
machine, ready for section 4.

You can also use your own first-person videos of lifting a bowl and placing it
on a plate. Keep your right hand and the objects visible, use consistent video
orientation and save each recording as a separate `.mp4` with a unique name.

Run this **on your laptop**, using a Linux, macOS or WSL terminal. Replace the
connection placeholders and `/path/to/recordings` with your own details:

```bash
scp -P POD_PORT -i /path/to/ssh_private_key \
  /path/to/recordings/*.mp4 \
  root@POD_IP:/workspace/learning-from-ego/data/personal_demonstrations/
```

Update the IP and port in every transfer command when changing pods. `scp` uses
uppercase `-P` for its port.

## 2. Install HuRo

Back **inside the GPU machine's tmux session**, use the Conda installation from
the setup guide:

```bash
source /workspace/miniforge3/etc/profile.d/conda.sh
conda deactivate
export HF_HOME=/workspace/.hf_cache
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

## 3. Add the MANO hand models

On your laptop, register at [MANO](https://mano.is.tue.mpg.de/) and download
**Models & Code → mano_v1_2.zip**. Extract it. Only `MANO_RIGHT.pkl` and
`MANO_LEFT.pkl` are needed. The Python 2.7 note applies to the bundled example
code; HuRo keeps its Python 3.11 environment.

**Inside the pod**, create the destination folders:

```bash
mkdir -p /workspace/HuRo/submodules/hawor/_DATA/data/mano
mkdir -p /workspace/HuRo/submodules/hawor/_DATA/data_left/mano_left
```

**On your laptop, in WSL**, copy the files from the extracted models folder:

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

## 4. Prepare all recordings

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

## 5. Run all ten stages

If every recording has a prepared MP4 in `results/04_huro_all/clips`, skip
section 4. If you added recordings, run section 4 again to prepare the new files. Those MP4s are inputs and should look like the original recordings.
First complete section 3's check that both MANO files are present.

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
above. For a new set of recordings, use the full section 5 settings again to
reset `FIRST_STAGE=1`.

Our first renderer hung during warmup and reported shader errors plus missing
`libGLU.so.1`. Installing `libglu1-mesa` fixed that run. If it happens again,
check the detailed logs under:

```text
/workspace/miniforge3/envs/huro/lib/python3.11/site-packages/isaacsim/kit/logs/Kit/Isaac-Sim Python/5.1/
```

Import checks alone do not confirm successful rendering; check that the output
videos exist and can be played.

## 6. Inspect the results

Run this **on your laptop in WSL**:

```bash
mkdir -p ./results/04_huro_all

scp -r -P POD_PORT -i /path/to/ssh_private_key \
  root@POD_IP:/workspace/learning-from-ego/results/04_huro_all/. \
  ./results/04_huro_all/
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

## 7. Return to dataset preparation and training

Keep the HuRo outputs on the GPU machine. Downloading them is only for inspection,
so there is no need to upload them again.

Continue at [Setup, section 4: prepare the human dataset](setup.md#4-process-your-recordings-and-prepare-the-human-dataset).
That section switches to `smolvla` and runs `prepare_huro.py`. The following
sections train and evaluate both models using the same LIBERO dataset.
