# VLLM Example

# Docker based build

This example has a Dockerfile based setup that installs vLLM 0.1.12 based on a git LFS tracked .whl
file. This also installs nsys into the docker container so that you have all the necessary
dependencies. However, you don't have to use this to run vLLM, you can just use your own version of
the vLLM code base that you're running as is and ensure that you have the following dependencies:
- nsight systems (if you want to run profiling with nsys)
- the ncompass SDK installed:
  -  You can take a look at the "Notes on developemnt" section below to understand why we can't
  install the ncomapss SDK with the -e flag, but if you want to install the SDK from src code, just
  run `pip install ../../` (without -e).

If you want to use the docker based system we provide, run the following:
## Setup commands:
```bash
python -m nc_pkg --build --run
```

## Shutdown commands:
```bash
python -m nc_pkg --down
```

# Run commands (with sudo):

### nsys run command
```bash
NCOMPASS_CACHE_DIR=<>\
NCOMPASS_PROFILER_TYPE=<>\
  ncompass profile -- <absoulte path to python executable> main.py --nsys
```

### torch profiler run command
```bash
VLLM_TORCH_PROFILER_DIR=.torch_traces\
NCOMPASS_CACHE_DIR=<>\
NCOMPASS_PROFILER_TYPE=<>\
  <absoulte path to python executable> main.py --torch
```

## Run commands (without sudo): 

If using the docker build setup, to run without sudo, you need to edit the Dockerfile 
to not have the last line (the one that sets user). This way the container is root, 
so you don't have to use sudo with nsys

### nsys run command
```bash
NCOMPASS_CACHE_DIR=<path to top directory that contains the .cache/ dir>\
NCOMPASS_PROFILER_TYPE=NVTX\
  ncompass profile --no-sudo -- python main.py --nsys
```

### torch profiler run command
```bash
VLLM_TORCH_PROFILER_DIR=.torch_traces\
NCOMPASS_CACHE_DIR=<path to top directory that contains the .cache/ dir>\
NCOMPASS_PROFILER_TYPE=Torch\
  python main.py --torch
```

# Notes on development
Because of the way the ncompass SDK is now built, we can't install it in development mode (i.e.
with -e).

Basically, we've added two files `ncompass.pth` and `ncompass_init.py` which get added to the pip
package which deal with doing the rewrites by calling `enable_rewrites`. `.pth` files (if found in
`..../site-packages/*.pth`) are called on startup of each python process. This way, we don't need
to enforce things like: `enable_rewrites` needs to be called at the module level and not inside
functions etc.

But the build process for packaging `*.pth` files (using `setup.py`) means that we can't get `-e`
builds to put those files somewhere in the PYTHONPATH. 

To see the logic of `enable_rewrites`, look at `ncompass_init.py`. We basically construct the path
to the `config.json` using `NCOMPASS_CACHE_DIR` and `NCOMPASS_PROFILER_TYPE`. There's error
handling to ensure both of them need to be specified if either one is.

If there's an error that occurs in `ncompass_init.py`, default python behavior is to not cause the
program to crash, which means that it'll throw a bunch of error messages, but won't stop execution
of the user's program.
