from setuptools import setup

setup(
    name='DatasetGenerator',
    packages=['DatasetGenerator'],
    entry_points={
        'aiida.workflows': ["datasetgenerator = DatasetGenerator.DatasetGeneratorWorkChain:DatasetGeneratorWorkChain"]
    }
)
