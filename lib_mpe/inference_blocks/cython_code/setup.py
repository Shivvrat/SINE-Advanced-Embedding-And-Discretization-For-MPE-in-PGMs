import numpy as np
import torch
from Cython.Build import cythonize
from Cython.Compiler import Options
from setuptools import Extension, setup

# Enable Cython compiler directives for optimization
Options.docstrings = False
Options.embed_pos_in_docstring = False
Options.generate_cleanup_code = False

# Define compiler and linker flags
extra_compile_args = ["-O3", "-march=native", "-ffast-math", "-fopenmp"]
extra_link_args = ["-fopenmp"]

extensions = [
    Extension(
        "*",
        ["*.pyx"],
        include_dirs=[np.get_include(), torch._C.__file__],
        language="c++",
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
    )
]

setup(
    ext_modules=cythonize(
        extensions,
        compiler_directives={
            "language_level": "3",
            "boundscheck": False,
            "wraparound": False,
            "initializedcheck": False,
            "nonecheck": False,
            "cdivision": True,
        },
        annotate=True,
    ),
)
