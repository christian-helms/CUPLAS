from pathlib import Path
from setuptools import setup, find_packages
from torch.utils.cpp_extension import (
    CUDAExtension,
    BuildExtension,
)

library_name = "plas"


def get_extensions():
    this_dir = Path(__file__).parent
    extensions_dir = this_dir / library_name
    sources = list((Path(library_name) / "src").glob("*"))

    extra_compile_args = {
        "cxx": [
            "-O3",
            "-fdiagnostics-color=always",
        ],
        "nvcc": [
            "-O3",
            "-I" + str(extensions_dir / "include"),
            "-I/usr/local/cuda/include",
        ],
    }

    extra_link_args = [
        "-lcudart",
    ]

    ext_modules = [
        CUDAExtension(
            f"{library_name}.cuplas",
            sources,
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args,
        )
    ]

    return ext_modules


setup(
    name=library_name,
    version="0.1",
    ext_modules=get_extensions(),
    packages=find_packages(),
    cmdclass={"build_ext": BuildExtension},
)
