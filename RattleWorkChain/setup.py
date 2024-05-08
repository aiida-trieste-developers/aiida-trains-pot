from setuptools import setup

setup(
    name='RattleWorkChain',
    packages=['RattleWorkChain'],
    entry_points={
        'aiida.workflows': ["rattleworkchain = RattleWorkChain.RattleWorkChain:RattleWorkChain"]
    }
)
