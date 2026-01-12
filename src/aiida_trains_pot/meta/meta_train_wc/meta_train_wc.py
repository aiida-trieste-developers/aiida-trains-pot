"""MetaTrain WorkChain to launch METAtrain training."""

import os

from aiida import load_profile
from aiida.common.extendeddicts import AttributeDict
from aiida.engine import (
    BaseRestartWorkChain,
    ProcessHandlerReport,
    process_handler,
    while_,
)
from aiida.orm import FolderData
from aiida.plugins import CalculationFactory

load_profile()

MetaCalculation = CalculationFactory("trains_pot.metatrain")


class MetaTrainWorkChain(BaseRestartWorkChain):
    """WorkChain to launch METAtrain training."""

    _process_class = MetaCalculation

    @classmethod
    def define(cls, spec):
        """Specify inputs and outputs."""
        super().define(spec)

        spec.expose_inputs(MetaCalculation, namespace="train", namespace_options={"validator": None})
        spec.expose_outputs(MetaCalculation)
        spec.input_namespace(
            "checkpoints",
            valid_type=FolderData,
            required=False,
            help="Checkpoints file",
        )
        spec.outline(
            cls.setup,
            while_(cls.should_run_process)(
                cls.run_process,
                cls.inspect_process,
            ),
            cls.results,
        )

    def report_error_handled(self, calculation, action):
        """Report an action taken for a calculation that has failed.

        This should be called in a registered error handler if its condition
        is met and an action was taken.
        :param calculation: the failed calculation node
        :param action: a string message with the action taken
        """
        self.report(
            f"{calculation.process_label}<{calculation.pk}> failed "
            f"with exit status {calculation.exit_status}: {calculation.exit_message}"
        )
        self.report(f"Action taken: {action}")

    def set_restart(self, calculation):
        """Set the parameters to run the restart calculation.

        Depending on the type of restart several variables of the input parameters
        will be changed to try to ensure that the calculation can resume from
        the last stored structure

        :param calculation: node from the previous calculation
        """
        files_retrieved = calculation.outputs.retrieved.list_object_names()
        for file in files_retrieved:
            output_filename = file
            if "checkpoint" in output_filename:
                folder_data = FolderData()
                folder_contents = calculation.outputs.retrieved.list_object_names(output_filename)
                for file_in_folder in folder_contents:
                    file_path = os.path.join(output_filename, file_in_folder)
                    with calculation.outputs.retrieved.open(file_path, "rb") as handle:
                        folder_data.put_object_from_filelike(handle, file_in_folder)
                self.ctx.inputs.checkpoints_restart = folder_data

        if "checkpoints" in calculation.outputs:
            self.ctx.inputs.checkpoints_restart = calculation.outputs.checkpoints

    def setup(self):
        """Call the ``setup`` of the ``BaseRestartWorkChain`` and create the inputs dictionary in ``self.ctx.inputs``.

        This ``self.ctx.inputs`` dictionary will be used by the ``BaseRestartWorkChain`` to submit the calculations
        in the internal loop.

        The ``parameters`` and ``settings`` input ``Dict`` nodes are converted into a regular dictionary and the
        default namelists for the ``parameters`` are set to empty dictionaries if not specified.
        """
        super().setup()
        self.ctx.inputs = AttributeDict(self.exposed_inputs(MetaCalculation, namespace="train"))

    @process_handler(
        priority=610,
        exit_codes=[
            MetaCalculation.exit_codes.ERROR_OUT_OF_WALLTIME,  # pylint: disable=no-member
        ],
    )
    def handle_out_of_walltime(self, calculation):
        """Handle calculations where the walltime was reached.

        The handler will try to find a configuration to restart from with the
        following priority

        Use a stored restart file in the repository from the previous calculation.
        """
        self.report("Walltime reached attempting restart")

        if "retrieved" in calculation.outputs:
            self.set_restart(
                calculation=calculation,
            )
            self.report_error_handled(
                calculation,
                "restarting from the stored checkpoints",
            )

        return ProcessHandlerReport(True)
