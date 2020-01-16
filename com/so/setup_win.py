# Run as:
#    python setup_win.py build_ext --inplace

import numpy as np
from distutils.extension import Extension
from Cython.Distutils import build_ext
from distutils.core import setup

try:
    numpy_include = np.get_include()
except AttributeError:
    numpy_include = np.get_numpy_include()

# 导入示例： from com.so.bbox.cython_bbox import bbox_overlaps
ext_modules = [
    Extension(
        "bbox.cython_bbox",  # 创建出的 so文件
        ["bbox/bbox.pyx"],  # pyx的路径
        extra_compile_args={'gcc': ["-Wno-cpp", "-Wno-unused-function"]},
        include_dirs=[numpy_include]
    ),
    Extension(
        "nms.cpu_nms",
        ["nms/cpu_nms.pyx"],
        extra_compile_args={'gcc': ["-Wno-cpp", "-Wno-unused-function"]},
        include_dirs=[numpy_include]
    )
]

setup(
    cmdclass={'build_ext': build_ext},
    ext_modules=ext_modules,
    # 注意这一句一定要有，不然只编译成C代码，无法编译成pyd文件
    include_dirs=[np.get_include()]
)
