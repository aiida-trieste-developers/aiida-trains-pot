from setuptools import setup

setup(
    name='QECalculation',
    packages=['QECalculation'],
    entry_points={
        'aiida.workflows': ["qecalculation = QECalculation.QECalculationWorkChain:QECalculationWorkChain"]
    }
)
