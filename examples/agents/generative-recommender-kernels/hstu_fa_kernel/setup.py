"""
Build script for the from-scratch HSTU attention kernel (hstu_ai_optimized package).
Based on fa3/setup.py.
Registers ops under torch.ops.hstu_ai_optimized namespace.
"""

import itertools
import os
import platform
import subprocess
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path

import torch
from packaging.version import parse, Version
from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDA_HOME, CUDAExtension

# HACK: monkey patch pytorch's _write_ninja_file to pass correct arch flags
# for sm90a .cu files. This is essential for Hopper sm90a support.
from torch.utils.cpp_extension import (
    _is_cuda_file,
    _join_cuda_home,
    _join_rocm_home,
    _maybe_write,
    COMMON_HIP_FLAGS,
    get_cxx_compiler,
    IS_HIP_EXTENSION,
    IS_WINDOWS,
    SUBPROCESS_DECODE_ARGS,
)


@dataclass(frozen=True)
class BuildConfig:
    """Frozen configuration for the HSTU kernel build."""
    this_dir: str = os.path.dirname(os.path.abspath(__file__))
    package_name: str = "hstu_ai_optimized"
    force_build: bool = os.getenv("FLASH_ATTENTION_FORCE_BUILD", "FALSE") == "TRUE"
    skip_cuda_build: bool = os.getenv("FLASH_ATTENTION_SKIP_CUDA_BUILD", "FALSE") == "TRUE"
    force_cxx11_abi: bool = os.getenv("FLASH_ATTENTION_FORCE_CXX11_ABI", "FALSE") == "TRUE"
    disable_backward: bool = True  # Always forward-only for scratch kernel
    disable_fp16: bool = os.getenv("FLASH_ATTENTION_DISABLE_FP16", "TRUE") == "TRUE"
    disable_hdim64: bool = os.getenv("FLASH_ATTENTION_DISABLE_HDIM64", "TRUE") == "TRUE"
    disable_hdim96: bool = os.getenv("FLASH_ATTENTION_DISABLE_HDIM96", "TRUE") == "TRUE"
    disable_hdim128: bool = os.getenv("FLASH_ATTENTION_DISABLE_HDIM128", "FALSE") == "TRUE"
    disable_hdim192: bool = os.getenv("FLASH_ATTENTION_DISABLE_HDIM192", "TRUE") == "TRUE"
    disable_hdim256: bool = os.getenv("FLASH_ATTENTION_DISABLE_HDIM256", "TRUE") == "TRUE"
    disable_sm8x: bool = True  # Always SM90-only


def _sanitize_flags(flags):
    """Strip whitespace from each flag, treating None as empty."""
    if flags is None:
        return []
    return [flag.strip() for flag in flags]


def _ninja_config_block(with_cuda, cuda_dlink_post_cflags):
    """Build the ninja config block (version, compiler, nvcc paths)."""
    compiler = get_cxx_compiler()
    config = ["ninja_required_version = 1.3", f"cxx = {compiler}"]
    if with_cuda or cuda_dlink_post_cflags:
        if IS_HIP_EXTENSION:
            nvcc = _join_rocm_home("bin", "hipcc")
        else:
            nvcc = _join_cuda_home("bin", "nvcc")
        nvcc_from_env = os.getenv("PYTORCH_NVCC", nvcc)
        config.append(f"nvcc_from_env = {nvcc_from_env}")
        config.append(f"nvcc = {nvcc}")
    return config


def _ninja_flags_block(cflags, post_cflags, cuda_cflags, cuda_post_cflags,
                       cuda_dlink_post_cflags, ldflags, with_cuda):
    """Build the ninja variable block for all compiler/linker flags."""
    if IS_HIP_EXTENSION:
        post_cflags = COMMON_HIP_FLAGS + post_cflags
    flags = [
        f"cflags = {' '.join(cflags)}",
        f"post_cflags = {' '.join(post_cflags)}",
    ]
    if with_cuda:
        flags.append(f"cuda_cflags = {' '.join(cuda_cflags)}")
        flags.append(f"cuda_post_cflags = {' '.join(cuda_post_cflags)}")
        cuda_post_cflags_sm80 = [
            s if s != "arch=compute_90a,code=sm_90a" else "arch=compute_80,code=sm_80"
            for s in cuda_post_cflags
        ]
        flags.append(f"cuda_post_cflags_sm80 = {' '.join(cuda_post_cflags_sm80)}")
        cuda_post_cflags_sm80_sm90 = cuda_post_cflags + [
            "-gencode",
            "arch=compute_80,code=sm_80",
        ]
        flags.append(
            f"cuda_post_cflags_sm80_sm90 = {' '.join(cuda_post_cflags_sm80_sm90)}"
        )
        cuda_post_cflags_sm100 = [
            s
            if s != "arch=compute_90a,code=sm_90a"
            else "arch=compute_100a,code=sm_100a"
            for s in cuda_post_cflags
        ]
        flags.append(f"cuda_post_cflags_sm100 = {' '.join(cuda_post_cflags_sm100)}")
    flags.append(f"cuda_dlink_post_cflags = {' '.join(cuda_dlink_post_cflags)}")
    flags.append(f"ldflags = {' '.join(ldflags)}")
    return flags


def _ninja_cxx_compile_rule():
    """Build the C++ compile rule block."""
    compile_rule = ["rule compile"]
    if IS_WINDOWS:
        compile_rule.append(
            "  command = cl /showIncludes $cflags -c $in /Fo$out $post_cflags"
        )
        compile_rule.append("  deps = msvc")
    else:
        compile_rule.append(
            "  command = $cxx -MMD -MF $out.d $cflags -c $in -o $out $post_cflags"
        )
        compile_rule.append("  depfile = $out.d")
        compile_rule.append("  deps = gcc")
    return compile_rule


def _ninja_cuda_compile_rules():
    """Build all CUDA compile rule blocks (sm90, sm80, sm80_sm90, sm100).

    Returns (base_rule, sm80_rule, sm80_sm90_rule, sm100_rule).
    """
    cuda_compile_rule = ["rule cuda_compile"]
    nvcc_gendeps = ""
    if (
        torch.version.cuda is not None
        and os.getenv("TORCH_EXTENSION_SKIP_NVCC_GEN_DEPENDENCIES", "0") != "1"
    ):
        cuda_compile_rule.append("  depfile = $out.d")
        cuda_compile_rule.append("  deps = gcc")
        nvcc_gendeps = (
            "--generate-dependencies-with-compile --dependency-output $out.d"
        )

    dep_lines = cuda_compile_rule[1:]

    def _make_variant(rule_name, post_cflags_var):
        return (
            [f"rule {rule_name}"]
            + dep_lines
            + [f"  command = $nvcc_from_env {nvcc_gendeps} $cuda_cflags -c $in -o $out ${post_cflags_var}"]
        )

    cuda_compile_rule_sm80 = _make_variant("cuda_compile_sm80", "cuda_post_cflags_sm80")
    cuda_compile_rule_sm80_sm90 = _make_variant("cuda_compile_sm80_sm90", "cuda_post_cflags_sm80_sm90")
    cuda_compile_rule_sm100 = _make_variant("cuda_compile_sm100", "cuda_post_cflags_sm100")

    cuda_compile_rule.append(
        f"  command = $nvcc_from_env {nvcc_gendeps} $cuda_cflags -c $in -o $out $cuda_post_cflags"
    )

    return cuda_compile_rule, cuda_compile_rule_sm80, cuda_compile_rule_sm80_sm90, cuda_compile_rule_sm100


def _ninja_build_statements(sources, objects, with_cuda):
    """Generate ninja build statements mapping sources to objects."""
    build = []
    for source_file, object_file in zip(sources, objects):
        is_cuda_source = _is_cuda_file(source_file) and with_cuda
        if is_cuda_source:
            if source_file.endswith("_sm90.cu"):
                rule = "cuda_compile"
            elif source_file.endswith("_sm80.cu"):
                rule = "cuda_compile_sm80"
            elif source_file.endswith("_sm100.cu"):
                rule = "cuda_compile_sm100"
            else:
                rule = "cuda_compile_sm80_sm90"
        else:
            rule = "compile"
        if IS_WINDOWS:
            source_file = source_file.replace(":", "$:")
            object_file = object_file.replace(":", "$:")
        source_file = source_file.replace(" ", "$ ")
        object_file = object_file.replace(" ", "$ ")
        build.append(f"build {object_file}: {rule} {source_file}")
    return build


def _ninja_devlink_block(cuda_dlink_post_cflags, objects):
    """Generate device-link rule and build statement if needed.

    Returns (devlink_rule, devlink_build, extra_objects).
    """
    if cuda_dlink_post_cflags:
        devlink_out = os.path.join(os.path.dirname(objects[0]), "dlink.o")
        devlink_rule = [
            "rule cuda_devlink",
            "  command = $nvcc $in -o $out $cuda_dlink_post_cflags",
        ]
        devlink_build = [f"build {devlink_out}: cuda_devlink {' '.join(objects)}"]
        return devlink_rule, devlink_build, [devlink_out]
    return [], [], []


def _ninja_link_block(library_target, objects):
    """Generate link rule, build statement, and default target.

    Returns (link_rule, link_build, default).
    """
    if library_target is None:
        return [], [], []

    link_rule = ["rule link"]
    if IS_WINDOWS:
        cl_paths = (
            subprocess.check_output(["where", "cl"])
            .decode(*SUBPROCESS_DECODE_ARGS)
            .split("\r\n")
        )
        if len(cl_paths) >= 1:
            cl_path = os.path.dirname(cl_paths[0]).replace(":", "$:")
        else:
            raise RuntimeError("MSVC is required to load C++ extensions")
        link_rule.append(
            f'  command = "{cl_path}/link.exe" $in /nologo $ldflags /out:$out'
        )
    else:
        link_rule.append("  command = $cxx $in $ldflags -o $out")

    link_build = [f"build {library_target}: link {' '.join(objects)}"]
    default = [f"default {library_target}"]
    return link_rule, link_build, default


def _write_ninja_file(
    path,
    cflags,
    post_cflags,
    cuda_cflags,
    cuda_post_cflags,
    cuda_dlink_post_cflags,
    sources,
    objects,
    ldflags,
    library_target,
    with_cuda,
    **kwargs,
) -> None:
    r"""Write a ninja file that does the desired compiling and linking.

    `path`: Where to write this file
    `cflags`: list of flags to pass to $cxx. Can be None.
    `post_cflags`: list of flags to append to the $cxx invocation. Can be None.
    `cuda_cflags`: list of flags to pass to $nvcc. Can be None.
    `cuda_post_cflags`: list of flags to append to the $nvcc invocation. Can be None.
    `sources`: list of paths to source files
    `objects`: list of desired paths to objects, one per source.
    `ldflags`: list of flags to pass to linker. Can be None.
    `library_target`: Name of the output library. Can be None; in that case,
                      we do no linking.
    `with_cuda`: If we should be compiling with CUDA.
    """
    cflags = _sanitize_flags(cflags)
    post_cflags = _sanitize_flags(post_cflags)
    cuda_cflags = _sanitize_flags(cuda_cflags)
    cuda_post_cflags = _sanitize_flags(cuda_post_cflags)
    cuda_dlink_post_cflags = _sanitize_flags(cuda_dlink_post_cflags)
    ldflags = _sanitize_flags(ldflags)

    assert len(sources) == len(objects)
    assert len(sources) > 0

    sources = [os.path.abspath(file) for file in sources]

    config = _ninja_config_block(with_cuda, cuda_dlink_post_cflags)
    flags = _ninja_flags_block(cflags, post_cflags, cuda_cflags, cuda_post_cflags,
                               cuda_dlink_post_cflags, ldflags, with_cuda)
    compile_rule = _ninja_cxx_compile_rule()

    blocks = [config, flags, compile_rule]

    if with_cuda:
        cuda_rules = _ninja_cuda_compile_rules()
        blocks.extend(cuda_rules)

    build = _ninja_build_statements(sources, objects, with_cuda)
    devlink_rule, devlink_build, extra_objects = _ninja_devlink_block(
        cuda_dlink_post_cflags, objects
    )
    objects = objects + extra_objects
    link_rule, link_build, default = _ninja_link_block(library_target, objects)

    blocks += [devlink_rule, link_rule, build, devlink_build, link_build, default]
    content = "\n\n".join("\n".join(b) for b in blocks)
    content += "\n"
    _maybe_write(path, content)


# Monkey patching
torch.utils.cpp_extension._write_ninja_file = _write_ninja_file
# Bypass CUDA version mismatch check (container nvcc may be newer than PyTorch's build CUDA)
torch.utils.cpp_extension._check_cuda_version = lambda *args, **kwargs: None


def get_platform():
    if sys.platform.startswith("linux"):
        return "linux_x86_64"
    elif sys.platform == "darwin":
        mac_version = ".".join(platform.mac_ver()[0].split(".")[:2])
        return f"macosx_{mac_version}_x86_64"
    elif sys.platform == "win32":
        return "win_amd64"
    else:
        raise ValueError("Unsupported platform: {}".format(sys.platform))


def get_cuda_bare_metal_version(cuda_dir):
    raw_output = subprocess.check_output(
        [cuda_dir + "/bin/nvcc", "-V"], universal_newlines=True
    )
    output = raw_output.split()
    release_idx = output.index("release") + 1
    bare_metal_version = parse(output[release_idx].split(",")[0])
    return raw_output, bare_metal_version


def check_if_cuda_home_none(global_option: str) -> None:
    if CUDA_HOME is not None:
        return
    warnings.warn(
        f"{global_option} was requested, but nvcc was not found.  Are you sure your environment has nvcc available?"
    )


def nvcc_threads_args():
    nvcc_threads = os.getenv("NVCC_THREADS") or "4"
    return ["--threads", nvcc_threads]


def validate_cuda_version():
    """Check CUDA home is set and version is >= 12.3."""
    check_if_cuda_home_none("hstu_ai_optimized")
    _, bare_metal_version = get_cuda_bare_metal_version(CUDA_HOME)
    if bare_metal_version < Version("12.3"):
        raise RuntimeError(
            f"FlashAttention-3 is only supported on CUDA 12.3 and above, get {bare_metal_version} from {CUDA_HOME}"
        )


def get_feature_args(cfg: BuildConfig):
    """Build the list of -D preprocessor flags from the config."""
    return (
        []
        + (["-DFLASHATTENTION_DISABLE_BACKWARD"] if cfg.disable_backward else [])
        + (["-DFLASHATTENTION_DISABLE_FP16"] if cfg.disable_fp16 else [])
        + ["-DFLASHATTENTION_DISABLE_FP8"]
        + (["-DFLASHATTENTION_DISABLE_HDIM64"] if cfg.disable_hdim64 else [])
        + (["-DFLASHATTENTION_DISABLE_HDIM96"] if cfg.disable_hdim96 else [])
        + (["-DFLASHATTENTION_DISABLE_HDIM128"] if cfg.disable_hdim128 else [])
        + (["-DFLASHATTENTION_DISABLE_HDIM192"] if cfg.disable_hdim192 else [])
        + (["-DFLASHATTENTION_DISABLE_HDIM256"] if cfg.disable_hdim256 else [])
        + (["-DFLASHATTENTION_DISABLE_SM8x"] if cfg.disable_sm8x else [])
    )


def get_kernel_sources(cfg: BuildConfig):
    """Enumerate all .cpp and .cu source files for the extension."""
    dtypes = ["bf16"] + (["fp16"] if not cfg.disable_fp16 else [])
    head_dimensions = (
        []
        + ([64] if not cfg.disable_hdim64 else [])
        + ([96] if not cfg.disable_hdim96 else [])
        + ([128] if not cfg.disable_hdim128 else [])
        + ([192] if not cfg.disable_hdim192 else [])
        + ([256] if not cfg.disable_hdim256 else [])
    )

    softmax_variants = ["softmaxtrue", "softmaxfalse"]
    sources_fwd_sm90 = [
        f"kernel/instantiations/flash_fwd_hdim{hdim}_{dtype}_{softmax}_sm90.cu"
        for hdim, dtype, softmax in itertools.product(head_dimensions, dtypes, softmax_variants)
    ]

    return (
        [
            "kernel/flash_api.cpp",
            "kernel/flash_api_cpu.cpp",
            "kernel/flash_common.cpp",
            "kernel/flash_common_cpu.cpp",
        ]
        + sources_fwd_sm90
    )


def get_nvcc_flags():
    """Return the base nvcc compiler flags."""
    nvcc_flags = [
        "-O3",
        "-std=c++17",
        "--ftemplate-backtrace-limit=0",
        "--use_fast_math",
        "--resource-usage",
        "-lineinfo",
        "-DCUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED",
        "-DCUTLASS_ENABLE_GDC_FOR_SM90",
        "-DCUTLASS_DEBUG_TRACE_LEVEL=0",
        "-DNDEBUG",
        "-Xfatbin",
        "-compress-all",
    ]
    if get_platform() == "win_amd64":
        nvcc_flags.extend([
            "-D_USE_MATH_DEFINES",
            "-Xcompiler=/Zc:__cplusplus",
        ])
    return nvcc_flags


def get_include_dirs(cfg: BuildConfig):
    """Return include directories for kernel headers and cutlass."""
    repo_root = Path(cfg.this_dir).parent
    cutlass_dir = repo_root / "generative-recommenders" / "generative_recommenders" / "ops" / "cpp" / "cutlass"
    return [
        Path(cfg.this_dir) / "kernel",
        cutlass_dir / "include",
    ]


def build_cuda_extensions(cfg: BuildConfig):
    """Build CUDA extension modules based on the given config."""
    if cfg.skip_cuda_build:
        return []

    print("\n\ntorch.__version__  = {}\n\n".format(torch.__version__))
    validate_cuda_version()

    if cfg.force_cxx11_abi:
        torch._C._GLIBCXX_USE_CXX11_ABI = True

    cc_flag = ["-gencode", "arch=compute_90a,code=sm_90a"]
    feature_args = get_feature_args(cfg)
    sources = get_kernel_sources(cfg)
    nvcc_flags = get_nvcc_flags()
    include_dirs = get_include_dirs(cfg)

    return [
        CUDAExtension(
            name=f"{cfg.package_name}._C",
            sources=sources,
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17", "-DPy_LIMITED_API=0x03090000"]
                + feature_args,
                "nvcc": nvcc_threads_args() + nvcc_flags + cc_flag + feature_args,
            },
            include_dirs=include_dirs,
            py_limited_api=True,
        )
    ]


cfg = BuildConfig()
ext_modules = build_cuda_extensions(cfg)

setup(
    name=cfg.package_name,
    version="0.1.0",
    packages=find_packages(
        exclude=(
            "build",
            "csrc",
            "include",
            "tests",
            "dist",
            "docs",
            "benchmarks",
        )
    ),
    description="HSTU FlashAttention from-scratch kernel (hstu_ai_optimized)",
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},
    python_requires=">=3.8",
    install_requires=[
        "torch",
        "packaging",
        "ninja==1.11.1.1",
    ],
)
