# -*- coding: utf-8 -*-
"""Equation of State WorkChain."""
from aiida.engine import WorkChain, append_, calcfunction, workfunction
from aiida import load_profile
from aiida.orm import Code, Dict, Int, List, FolderData, SinglefileData
from aiida.plugins import CalculationFactory
from ase.io import write
import os
import io
from contextlib import redirect_stdout

load_profile()

MaceCalculation = CalculationFactory('trains_pot.macetrain')




class MaceTrainWorkChain(WorkChain):
    """WorkChain to launch MACE training."""

    @classmethod
    def define(cls, spec):
        """Specify inputs and outputs."""
        super().define(spec)

        spec.expose_inputs(MaceCalculation, namespace="mace", namespace_options={'validator': None})
        spec.expose_outputs(MaceCalculation, namespace="mace_calc")
        spec.input_namespace("checkpoints", valid_type=FolderData, required=False, help="Checkpoints file",)
        spec.outline(
            cls.run_mace,
            cls.finalize
        )

    @classmethod
    def get_builder_from_protocol(
        cls, **kwargs):
        
        builder = cls.get_builder()

        return builder
    


    def run_mace(self):
        """Run MACE calculations."""

        inputs = self.exposed_inputs(MaceCalculation, namespace="mace")

        future = self.submit(MaceCalculation, **inputs)

        self.report(f'Launched MACE calculation <{future.pk}>')
        self.to_context(mace_calculation=future)

    def finalize(self):
        """Finalize."""
        self.out_many(self.exposed_outputs(self.ctx.mace_calculation, MaceCalculation, namespace="mace_calc"))
