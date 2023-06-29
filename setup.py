from setuptools import setup

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='ComplexSystemSimulation',
    version='1.0.0',
    packages=[''],
    url='',
    license='',
    description='Artic ice melt ponds Cellular Automata (CA) model',
    install_requires = requirements
)
