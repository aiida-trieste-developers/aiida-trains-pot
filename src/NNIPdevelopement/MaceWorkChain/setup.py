from setuptools import setup

setup(
    name='MaceWorkChain',
    packages=['MaceWorkChain'],
    entry_points={
        'aiida.workflows': ["maceworkchain = MaceWorkChain.MaceWorkChain:MaceWorkChain"]
    }
)
