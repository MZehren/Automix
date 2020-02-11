#!/usr/bin/env python
import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()
setuptools.setup(
    name='AutoMix',
    version='0.1',
    author="Mickael Zehren",
    author_email="mickael.zehren@gmail.com",
    description="Automatic DJ-mixing of tracks",
    long_description=long_description,
    install_requires=[
        "numpy", "scipy", "cython", "matplotlib", "pandas", "pyaudio", "madmom", "librosa", "essentia", "youtube-dl", "scdl",
        "mir_eval", "msaf", "graphviz"
    ],
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
)
