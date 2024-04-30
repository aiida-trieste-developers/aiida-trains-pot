from setuptools import setup

setup(
    name='LammpsWorkChain',
    packages=['LammpsWorkChain'],
    entry_points={
        'aiida.workflows': ["lammpsworkchain = LammpsWorkChain.LammpsWorkChain:LammpsWorkChain"]
    }
)