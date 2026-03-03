from setuptools import setup, find_packages

setup(
    name="logipar",
    version="1.0.0",
    description="Logic-Infused Framework for Clinical Risk Assessment",
    author="Logi-PAR Team",
    python_requires=">=3.8",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "timm>=0.9.0",
        "numpy>=1.21.0",
        "scipy>=1.7.0",
        "matplotlib>=3.5.0",
        "scikit-learn>=1.0.0",
        "tqdm>=4.64.0",
        "pillow>=9.0.0",
        "pyyaml>=6.0",
        "tensorboard>=2.11.0",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)
