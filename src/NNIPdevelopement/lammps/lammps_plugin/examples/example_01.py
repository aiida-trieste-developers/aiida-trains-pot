#!/usr/bin/env python
"""Run a test calculation on localhost.

Usage: ./example_01.py
"""
from os import path

import click

from aiida import cmdline, engine
from aiida.plugins import CalculationFactory, DataFactory

from aiida_diff import helpers

INPUT_DIR = path.join(path.dirname(path.realpath(__file__)), "input_files")


def test_run(diff_code):
    """Run a calculation on the localhost computer.

    Uses test helpers to create AiiDA Code on the fly.
    """
    if not diff_code:
        # get code
        computer = helpers.get_computer()
        diff_code = helpers.get_code(entry_point="diff", computer=computer)

    # Prepare input parameters
    DiffParameters = DataFactory("diff")
    parameters = DiffParameters({"ignore-case": True})

    SinglefileData = DataFactory("core.singlefile")
    file1 = SinglefileData(file=path.join(INPUT_DIR, "file1.txt"))
    file2 = SinglefileData(file=path.join(INPUT_DIR, "file2.txt"))

    # set up calculation
    inputs = {
        "code": diff_code,
        "parameters": parameters,
        "file1": file1,
        "file2": file2,
        "metadata": {
            "description": "Test job submission with the aiida_diff plugin",
        },
    }

    # Note: in order to submit your calculation to the aiida daemon, do:
    # from aiida.engine import submit
    # future = submit(CalculationFactory('diff'), **inputs)
    result = engine.run(CalculationFactory("diff"), **inputs)

    computed_diff = result["diff"].get_content()
    print(f"Computed diff between files: \n{computed_diff}")


@click.command()
@cmdline.utils.decorators.with_dbenv()
@cmdline.params.options.CODE()
def cli(code):
    """Run example.

    Example usage: $ ./example_01.py --code diff@localhost

    Alternative (creates diff@localhost-test code): $ ./example_01.py

    Help: $ ./example_01.py --help
    """
    test_run(code)


if __name__ == "__main__":
    cli()  # pylint: disable=no-value-for-parameter
