from setuptools import setup, Extension
import numpy as np

setup(
    ext_modules=[Extension(
        name='dzida_phy.native_optimized.corr_ext',
        sources=['src/dzida_phy/native_optimized/corr_ext.cpp'],
        include_dirs=[np.get_include()],
        extra_compile_args=[
            '-O3',
            '-march=native',       # tune for current CPU; implies -mavx2 -mfma
            '-mavx2', '-mfma',     # explicit for documentation
            '-ffast-math',         # allow FP reassociation, finite-only, no NaN/Inf
            '-funroll-loops',      # unroll loops beyond what -O3 does (edge regions)
            '-fno-semantic-interposition',  # allow inlining across .so boundaries
        ],
    )]
)
# compile: uv run python setup.py build_ext --build-lib src
