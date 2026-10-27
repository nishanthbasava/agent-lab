# mini-vla

A minimal vision-language-action (VLA) model built from scratch for learning.
Flow-matching action expert (DiT-style) over 50-step action chunks, conditioned
on vision/language tokens, evaluated closed-loop in LIBERO.

See `NOTES.md` for a running summary of each piece as it's built.

## Layout

```
data/          raw LIBERO HDF5 demos + normalization stats (gitignored)
minivla/       the package: dataset, vision encoder, action expert, flow matching
scripts/       runnable entry points; each step of the build ends with one
third_party/   LIBERO clone (gitignored)
```

## Setup

```bash
uv venv --python 3.10 .venv
source .venv/bin/activate
uv pip install -r requirements.txt
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git third_party/LIBERO
touch third_party/LIBERO/libero/__init__.py   # upstream relies on legacy editable installs
uv pip install -e third_party/LIBERO --no-deps
```

macOS only: mujoco 2.3.7 dlopens the wrong OpenGL framework path. Patch it in the venv:

```bash
sed -i '' "s|'/System/Library/OpenGL.framework/OpenGL'|'/System/Library/Frameworks/OpenGL.framework/OpenGL'|" \
  .venv/lib/python3.10/site-packages/mujoco/cgl/cgl.py
```

Sanity check (downloads one 485MB task file, replays demo 0, writes `data/replay.gif`):

```bash
python -c "from huggingface_hub import hf_hub_download; hf_hub_download('yifengzhu-hf/LIBERO-datasets', 'libero_spatial/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate_demo.hdf5', repo_type='dataset', local_dir='data')"
python scripts/replay_demo.py data/libero_spatial/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate_demo.hdf5
```
