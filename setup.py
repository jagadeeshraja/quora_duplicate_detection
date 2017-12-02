try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

setup(name='simple_deep_network',
      version='0.1',
      description=('Predicting Duplicate Questions'),
      packages=['src'],
      scripts=[]
      )
