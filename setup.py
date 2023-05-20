from setuptools import setup, find_packages

setup(
    name="wave-mixer",
    packages=find_packages(exclude=[]),
    version="0.1.0",
    license="MIT",
    description="WaveMixer for PyTorch",
    author="Tomonobu Inayama",
    author_email="tomonobu-inayama@codemajin.net",
    url="https://github.com/codemajin/wave-mixer",
    keywords=[
        "artificial intelligence",
        "deep learning",
        "speech recognition"
    ],
    install_requires=[
        "einops>=0.6",
        "torch>=1.12",
        "torchinfo>=1.8",
    ],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.9",
    ],
)
