from distutils.extension import Extension
from Cython.Distutils import build_ext

from distutils.core import setup
from Cython.Build import cythonize
import numpy as np

setup(
    name="pydbm",
    ext_modules=cythonize("pydbm/approximation/*.pyx", include_path = [np.get_include()]),
    cmdclass={"build_ext": build_ext}
)
