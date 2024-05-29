"""
Calculations provided by aiida_diff.

Register calculations via the "aiida.calculations" entry point in setup.json.
"""
from aiida.common import datastructures
from aiida.engine import CalcJob
from aiida.orm import SinglefileData, StructureData
from aiida.plugins import DataFactory
import io
from contextlib import redirect_stdout
from ase.io import write


TrajectoryData = DataFactory('core.array.trajectory')


class LammpsBaseCalculation(CalcJob):
    """
    AiiDA calculation plugin wrapping the diff executable.

    Simple AiiDA plugin wrapper for 'diffing' two files.
    """

    @classmethod
    def define(cls, spec):
        """Define inputs and outputs of the calculation."""
        super().define(spec)

        # set default values for AiiDA options
        spec.inputs["metadata"]["options"]["resources"].default = {"num_machines": 1, "num_mpiprocs_per_machine": 1,}
        spec.inputs["metadata"]["options"]["parser_name"].default = "NNIPdevelopement.lammpsmd"

        # new ports
        spec.input("input", valid_type=SinglefileData, help="Input file for LAMMPS.",)
        spec.input("potential", valid_type=SinglefileData, help="Potenetial file for LAMMPS.",)
        spec.input("structure", valid_type=StructureData, help="Input structure file for LAMMPS.",)
        # spec.output("_scheduler-stderr", valid_type=SinglefileData, help="Standard error of the scheduler.",)
        # spec.output("_scheduler-stdout", valid_type=SinglefileData, help="Standard output of the scheduler.",)
        spec.output("coord_lammpstrj", valid_type=SinglefileData, help="Trajectory coord lammpstrj file",)
        spec.output("msd", valid_type=SinglefileData, help="Mean square displacement file",)
        spec.output("adf", valid_type=SinglefileData, help="Angular distribution function file",)
        spec.output_namespace("rdf", valid_type=SinglefileData, dynamic=True, help="Output rdf files",)
        spec.output("lammps_out", valid_type=SinglefileData, help="Lammps output file",)
        spec.output("final_structure", valid_type=SinglefileData, help="Final structure file",)
        spec.output("lmp_restart", valid_type=SinglefileData, help="Lmp restart file",)
        # spec.output("trajectory", valid_type=TrajectoryData, help="Trajectory coord atom file",)
        spec.output("coord_atom", valid_type=SinglefileData, help="Trajectory coord atom file",)
        spec.output("log_lammps", valid_type=SinglefileData, help="Log lammps file",)
        

        spec.exit_code(300, "ERROR_MISSING_OUTPUT_FILES", message="Calculation did not produce all expected output files.",)

    @classmethod
    def get_builder_from_protocol(
        cls, **kwargs):

        builder = cls.get_builder(**kwargs)

        return builder
    
    def prepare_for_submission(self, folder):
        """
        Create input files.

        :param folder: an `aiida.common.folders.Folder` where the plugin should temporarily place all files
            needed by the calculation.
        :return: `aiida.common.datastructures.CalcInfo` instance
        """

        codeinfo = datastructures.CodeInfo()
        codeinfo.cmdline_params = '-k on g 1 -sf kk -in input.in'.split()
            # file1_name=self.inputs.file1.filename, file2_name=self.inputs.file2.filename
        codeinfo.code_uuid = self.inputs.code.uuid
        codeinfo.stdout_name = "lammps.out"
        

        # structure_txt = io.BytesIO(str(write('-', self.inputs.structure.get_ase(), format='lammps-data')).encode())
        with io.StringIO() as buf, redirect_stdout(buf):
            write('-', self.inputs.structure.get_ase(), format='lammps-data')
            structure_txt = buf.getvalue()

        # structure_file = SinglefileData(file=structure_txt, filename="structure.dat")
        # structure_file.add_incoming(self.inputs.structure, link_type="structure")
        with folder.open('structure.dat', "w") as handle:
                    handle.write(structure_txt)
        # Prepare a `CalcInfo` to be returned to the engine
        calcinfo = datastructures.CalcInfo()
        calcinfo.codes_info = [codeinfo]
        calcinfo.local_copy_list = [
            (
                self.inputs.input.uuid,
                self.inputs.input.filename,
                "input.in",
            ),
            (
                self.inputs.potential.uuid,
                self.inputs.potential.filename,
                "potential.dat",
            ),
        ]
        calcinfo.retrieve_list = ['rdf*', 'coord*', 'msd*', 'lammps.out', 'log.lammps', '_scheduler-std*', 'lmp_restart*', 'lmp.data', 'adf*']

        return calcinfo
