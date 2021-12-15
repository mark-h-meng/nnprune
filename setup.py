from setuptools import setup, find_packages

VERSION = '0.9a1' 
DESCRIPTION = 'Data-free robustness preserving neural network pruning'

with open("README.md", "r", encoding="utf-8") as fh:
    LONG_DESCRIPTION = fh.read()

# Setting up
setup(
       # the name must match the folder name 'nnprune'
        name="paoding-dl", 
        version=VERSION,
        author="Mark H. Meng",
        author_email="<menghs@i2r.a-star.edu.sg>",
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        long_description_content_type="text/markdown",
        url="https://github.com/pypa/sampleproject", # To be filled later
        project_urls={ # To be filled later
            "Bug Tracker": "https://github.com/pypa/sampleproject/issues",
        },
        
        packages=find_packages(),
        install_requires=[
            'tensorflow==2.3.0',
            'scikit-learn',
            'pandas',
            'progressbar2',
            'opencv-python>=4.5'
        ], 
        
        keywords=['python', 'neural network pruning'],
        classifiers= [
            "Development Status :: 3 - Alpha",
            "Intended Audience :: Education",
            "Programming Language :: Python :: 2",
            "Programming Language :: Python :: 3",
            "Operating System :: MacOS :: MacOS X",
            "Operating System :: Microsoft :: Windows",
        ]
)