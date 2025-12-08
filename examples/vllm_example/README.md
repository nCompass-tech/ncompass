# VLLM Example

# Setup commands:
```bash
python -m nc_pkg --build --run
```

# Shutdown commands:
```bash
python -m nc_pkg --down
```

# Run commands (with sudo):

### nsys run command
```bash
NCOMPASS_CACHE_DIR=<>\
NCOMPASS_PROFILER_TYPE=<>\
  ncompass profile -- /opt/venv/bin/python main.py --nsys
```

### torch profiler run command
```bash
VLLM_TORCH_PROFILER_DIR=.torch_traces\
NCOMPASS_CACHE_DIR=<>\
NCOMPASS_PROFILER_TYPE=<>\
  /opt/venv/bin/python main.py --torch
```

## Run commands (without sudo): 

To run without sudo, you need to edit the Dockerfile to not have the last line (the one that sets
user). This way the container is root, so you don't have to use sudo with nsys

### nsys run command
```bash
NCOMPASS_CACHE_DIR=<>\
NCOMPASS_PROFILER_TYPE=<>\
  ncompass profile --no-sudo -- python main.py --nsys
```

### torch profiler run command
```bash
VLLM_TORCH_PROFILER_DIR=.torch_traces\
NCOMPASS_CACHE_DIR=<>\
NCOMPASS_PROFILER_TYPE=<>\
  python main.py --torch
```

# Notes on development
Because of the way the ncompass SDK is now built, we can't install it in development mode (i.e.
with -e).

Basically, we've added two files `ncompass.pth` and `ncompass_init.py` which get added to the pip
package which deal with doing the rewrites by calling `enable_rewrites`. `.pth` files (if found in
`..../site-packages/*.pth`) are called on startup of each python process. This way, we don't need
to enforce things like `enable_rewrites` needs to be called at the module level and not inside
functions etc.

But the build process for packaging `*.pth` files (using `setup.py`) means that we can't get `-e`
builds to put those files somewhere in the PYTHONPATH. 

To see the logic of `enable_rewrites`, look at `ncompass_init.py`. We basically construct the path
to the `config.json` using `NCOMPASS_CACHE_DIR` and `NCOMPASS_PROFILER_TYPE`. There's error
handling to ensure both of them need to be specified if either one is.

If there's an error that occurs in `ncompass_init.py`, default python behavior is to not cause the
program to crash, which means that it'll throw a bunch of error messages, but won't stop execution
of the user's program.
