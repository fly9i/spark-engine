# Builds the part of exllamav3's CUDA extension that spark-engine links: EXL3 GEMM / MGEMM (quant/), reconstruct,
# Hadamard transforms and CUDA graph helpers. The CPU MoE (x86 SIMD), multi-GPU CPU all-reduce and Python bindings are
# not built, so the library has no Python or x86 dependencies. Run inside an exllamav3 checkout (v1.4.9), same compiler
# flags as upstream's setup.py.
import os
from setuptools import setup
from torch.utils import cpp_extension

ext = os.path.join("exllamav3", "exllamav3_ext")
sources = [os.path.join(r, f) for r, _, fs in os.walk(os.path.join(ext, "quant")) for f in fs if f.endswith(".cu")]
sources += [os.path.join(ext, f) for f in ("graph.cu", "hadamard.cpp", "cuda_drv.cpp")]
setup(
    name="exllamav3_ext_min",
    ext_modules=[cpp_extension.CUDAExtension(
        "exllamav3_ext", sorted(sources),
        extra_compile_args={"cxx": ["-Ofast"],
                            "nvcc": ["-lineinfo", "-O3", "--use_fast_math", "-Xcudafe", "--diag_suppress=177",
                                     "-Xcudafe", "--diag_suppress=20012"]})],
    cmdclass={"build_ext": cpp_extension.BuildExtension},
)
