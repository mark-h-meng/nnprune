from setuptools import setup, find_packages
import platform

VERSION = '1.0.0.post3'
DESCRIPTION = 'PAODING-DL: A Data-free Robustness-preserving Neural Network Pruning Toolkit'

with open("README.md", "r", encoding="utf-8") as fh:
    LONG_DESCRIPTION = fh.read()

if platform.system() == 'Darwin':
    tf_dependency = 'tensorflow-macos>=2.3.0'
else:
    tf_dependency = 'tensorflow>=2.3.0'

# Setting up
setup(
    # the name must match the folder name
    name="paoding-dl",
    version=VERSION,
    author="Mark H. Meng",
    author_email="<menghs@i2r.a-star.edu.sg>",
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    url="https://github.com/mark-h-meng/nnprune",
    project_urls={  # To be filled later
        "Bug Tracker": "https://github.com/mark-h-meng/nnprune/issues",
    },

    packages=find_packages(),
    test_suite="tests",
    install_requires=[
        tf_dependency,
        'scikit-learn',
        'pandas',
        'opencv-python>=4.5'
        'numpy',
        'tabulate',
        'pydot'
    ],

    keywords=['python', 'neural network pruning'],
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent'
    ]
)
