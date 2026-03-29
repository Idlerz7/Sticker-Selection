"""Preload setuptools vendored ``distutils.version`` before ``torch.utils.tensorboard``.

setuptools>=60 exposes ``distutils`` without a populated ``.version`` until the submodule
is imported; PyTorch's tensorboard helper uses ``distutils.version.LooseVersion`` without
that import (``AttributeError: module 'distutils' has no attribute 'version'``).
"""

from setuptools import distutils as _distutils_shim  # noqa: F401

import distutils.version  # noqa: F401
