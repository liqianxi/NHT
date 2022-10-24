import re
from setuptools import setup, find_packages
import sys

if sys.version_info.major != 3:
    print('This Python is only compatible with Python 3, but you are running '
          'Python {}. The installation will likely fail.'.format(sys.version_info.major))



setup(name='nht',
      packages=[package for package in find_packages()
                if package.startswith('nht')],
      install_requires=[
          'pytorch-lightning',
      ],
      extras_require=None,
      description='Neural Householder Transforms',
      author='Kerrick Johnstonbaugh & Michael Przystupa',
      url='https://github.com/KerrickJohnstonbaugh/SCL_wam',
      author_email='kerrick@ualberta.ca',
      version='0.0.1')


