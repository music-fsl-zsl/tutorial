from setuptools import setup, find_packages
from pathlib import Path

# open requirements.txt and read the requirements
with open(Path(__file__).parent / "requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name='music_fsl',
    version='0.1.0',
    description='Few-shot learning for music instrument recognition using PyTorch',
    author='Hugo Flores García',
    author_email='hugofloresgarcia@u.northwestern.edu',
    url='https://blog.godatadriven.com/setup-py',
    packages=find_packages(include=['music_fsl']),
    install_requires=requirements,
)