---
name: baseline-setup
description: >
  Environment setup for the DLRM-v3 + HSTU inference pipeline. Patches the
  generative-recommenders source, builds the HSTU kernel, and installs
  compatible Python dependencies. Run this before any optimization work.
---

# Baseline Setup

This skill fixes known environment issues in the Docker agent image so that
`model_runner/bench.py` and `model_runner/test_correctness.py` run cleanly.

[CRITICAL] Run every step below **exactly as written**. Do not improvise,
skip steps, or try alternative approaches. If a step fails, stop and report
the error — do not attempt to fix it yourself.

## Step 1: Apply source patches

The generative-recommenders source needs two patches for compatibility:
- `hstu_attention_import.patch` — stubs out a missing `hammer` import
- `cpp_compat.patch` — fixes C++ `std::get` and `variable_list` for newer compilers

```bash
cd /workspace/generative-recommenders && \
  patch -p1 < /workspace/fa3/reference/hstu_attention_import.patch && \
  patch -p1 < /workspace/fa3/reference/cpp_compat.patch
```

## Step 2: Build the HSTU reference kernel

The CUDA toolkit version (13.0) doesn't match PyTorch's reported version (12.8).
This is harmless but the build system rejects it. Patch the check, then build.

```bash
python -c "import torch.utils.cpp_extension as m, inspect, re; src = inspect.getfile(m); txt = open(src).read(); txt = re.sub(r'(def _check_cuda_version\([^)]*\)[^:]*:)', r'\1\n    return', txt); open(src, 'w').write(txt); print('Patched', src)"
```

Then build (takes ~3-5 minutes for CUDA kernel compilation):

```bash
cd /workspace/fa3/reference && \
  mkdir -p hstu && touch hstu/__init__.py && \
  FLASH_ATTENTION_DISABLE_BACKWARD=TRUE \
  FLASH_ATTENTION_DISABLE_FP16=TRUE \
  FLASH_ATTENTION_DISABLE_FP8=TRUE \
  FLASH_ATTENTION_DISABLE_HDIM64=TRUE \
  FLASH_ATTENTION_DISABLE_HDIM96=TRUE \
  FLASH_ATTENTION_DISABLE_HDIM128=FALSE \
  FLASH_ATTENTION_DISABLE_HDIM192=TRUE \
  FLASH_ATTENTION_DISABLE_HDIM256=TRUE \
  FLASH_ATTENTION_DISABLE_SM80=TRUE \
  pip install -e . --no-build-isolation 2>&1
```

Verify:

```bash
python -c "import torch; import hstu._C; print('hstu kernel: OK')"
```

## Step 3: Install compatible Python dependencies

fbgemm_gpu and torchrec **must** come from the PyTorch cu128 index to match the
installed PyTorch ABI. Do NOT install them from PyPI — those versions are
ABI-incompatible and will produce `undefined symbol` errors.

```bash
pip install 'fbgemm_gpu==1.5.0+cu128' 'torchrec==1.4.0+cu128' \
  --index-url https://download.pytorch.org/whl/cu128 \
  --extra-index-url https://pypi.org/simple
```

Then install the remaining workspace deps:

```bash
pip install -r /workspace/requirements.txt
```

Verify:

```bash
python -c "import torch; import fbgemm_gpu; import torchrec; print('deps: OK')"
```

## Step 4: Verify the full pipeline

```bash
cd /workspace && python model_runner/bench.py --max-seq-len 256 --bench-iters 3
```

This should complete without errors and print latency numbers. If it does,
setup is complete — proceed to the optimization skill.
