from setuptools import find_packages
from distutils.core import setup

setup(name='rl_g1_punch',
      version='1.0.0',
      author='kvkcon',
      license="BSD-3-Clause",
      packages=find_packages(),
      author_email='',
      description='punch walk for Unitree Robots',
      install_requires=['isaacgym', 'rsl-rl', 'matplotlib', 'numpy==1.20', 'tensorboard', 'mujoco==3.2.3', 'pyyaml'])
