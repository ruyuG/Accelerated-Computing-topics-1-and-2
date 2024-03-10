from setuptools import setup,Extension
from Cython.Build import cythonize
import numpy 

extension = Extension(
    "LebwohlLasher_cython",
    ["LebwohlLasher_cython.pyx"],
    include_dirs=[numpy.get_include()],  # NumPy header file paths
)

setup(
    name="LebwohlLasher_cython",
    ext_modules=cythonize([extension]),
)

# cause I got a error： fatal error: numpy/arrayobject.h: No such file or directory
# It means that Cython compiler couldn't find the header file numpy/arrayobject.h
# So ues numpy.get_include() to get the directory where the NumPy header files are located