from setuptools import setup, find_packages

VERSION = '0.0.20' 
DESCRIPTION = 'Stochastic data-free robustness preserving neural network pruning'

with open("README.md", "r", encoding="utf-8") as fh:
    LONG_DESCRIPTION = fh.read()

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
        project_urls={ # To be filled later
            "Bug Tracker": "https://github.com/mark-h-meng/nnprune/issues",
        },
        
        packages=find_packages(),
        test_suite="tests",
        install_requires=[
            'tensorflow==2.3.0',
            'scikit-learn',
            'pandas',
            'progressbar2',
            'opencv-python>=4.5'
            'numpy'
        ], 
        
        keywords=['python', 'neural network pruning'],
        classifiers= [
            'Development Status :: 2 - Pre-Alpha',
            'Intended Audience :: Education',
            'Intended Audience :: Science/Research',
            'Programming Language :: Python :: 3',
            'Operating System :: OS Independent'
        ]
)