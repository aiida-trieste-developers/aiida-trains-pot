# -*- coding: utf-8 -*-
"""Equation of State WorkChain."""
from aiida.engine import WorkChain, append_, calcfunction, workfunction
from aiida import load_profile
from aiida.orm import Code, Float, Str, StructureData, Int, Float, SinglefileData, List
from aiida_quantumespresso.workflows.protocols.utils import ProtocolMixin
from aiida.plugins import CalculationFactory
import random
import itertools
import numpy as np
from ase.io import write
from ase.calculators.singlepoint import SinglePointCalculator
from ase import Atoms
import os
import io
from contextlib import redirect_stdout

load_profile()

MaceCalculation = CalculationFactory('mace_base')

@calcfunction
def WriteDataset(**params):
    """Calculation function to write a dataset to a file

    :param structures: A list of AiiDA `StructureData` nodes
    """
    from ase.io import write
    from aiida.orm import SinglefileData
    import os

    dataset_list = []
    for key, value in params.items():

        dataset_list.append({'aiida_model': value['aiida_model'],
                             'aiida_swa_model': value['aiida_swa_model'],
                             'mace': value['mace'],
                             })

    return {'aiida_model':dataset_list['aiida_model'], 'aiida_swa_model':dataset_list['aiida_swa_model'], 'mace':dataset_list['mace']}

@calcfunction
def split_dataset(dataset):
    """Divide dataset into training, validation and test sets."""
    # data = self.inputs.dataset_list.get_list()
    data = dataset.get_list()

    # Define a function to extract the grouping key
    def get_grouping_key(d):
        return tuple((k, v) for k, v in d.items() if k not in ["energy", "cell", "stress", "forces", "symbols", "positions"])

    # Sort the data based on the grouping key
    sorted_data = sorted(data, key=get_grouping_key)

    # Group the sorted data by the grouping key
    grouped_data = itertools.groupby(sorted_data, key=get_grouping_key)

    # Iterate over the groups and print the group key and the list of dictionaries in each group
    training_set = []
    validation_set = []
    test_set = []
    all_set = []

    for key, group in grouped_data:
    # Calculate the number of elements for each set
        group_list = list(group)        
        _ = random.shuffle(group_list)        
        all_set += group_list[:]
    
    
    total_elements = len(all_set)
    training_size = int(0.8 * total_elements)
    validation_size = int(0.1 * total_elements)
    test_size = total_elements - training_size - validation_size
        # Split the data into sets
    training_set += all_set[:training_size]
    validation_set += all_set[training_size:training_size+validation_size]
    test_set += all_set[training_size+validation_size:]

    return {"train_set":List(training_set), "validation_set":List(validation_set), "test_set":List(test_set)}


class MaceWorkChain(ProtocolMixin, WorkChain):
    """WorkChain to launch MACE training."""

    @classmethod
    def define(cls, spec):
        """Specify inputs and outputs."""
        super().define(spec)

        spec.expose_inputs(MaceCalculation, namespace="mace", exclude=('training_set','validation_set','test_set'), namespace_options={'validator': None})
        spec.expose_outputs(MaceCalculation, namespace="mace_out")


        spec.input("code", valid_type=Code)
        spec.input("dataset_list", valid_type=List)
        spec.input("parent_folder", valid_type=Str)
        spec.output("aiida_model",valid_type=SinglefileData)
        spec.output("aiida_swa_model",valid_type=SinglefileData)
        spec.output("mace",valid_type=SinglefileData)
        spec.output("validation_set", valid_type=List)
        # spec.input("dt", valid_type=Float)
        # spec.input("num_steps", valid_type=Int)
        # spec.input("pressure", valid_type=Float)
        # spec.input("structure", valid_type=StructureData)
        # spec.input("potential", valid_type=SinglefileData)
        # spec.input("parent_folder", valid_type=Str)

        spec.outline(
            cls.run_mace,
            cls.results,
            cls.finalize            
        )

        # spec.output("coord_lmmpstrj", valid_type=SinglefileData, help="trajectory coord lammpstrj file")
        # spec.output("msd", valid_type=SinglefileData, help="msd file")
        # spec.output_namespace("rdf", valid_type=SinglefileData, dynamic=True, help="output rdf files")
        # spec.output("lammps_out", valid_type=SinglefileData, help="lammps output file")
        # spec.output("final_structure", valid_type=SinglefileData, help="final structure file")
        # spec.output("lmp_restart", valid_type=SinglefileData, help="lmp restart file")
        # spec.output("coord_atom", valid_type=SinglefileData, help="trajectory coord atom file")
        # spec.output("log_lammps", valid_type=SinglefileData, help="log lammps file")

    @classmethod
    def get_builder_from_protocol( 
        cls, **kwargs):
        
        builder = cls.get_builder()

        return builder
    


    def run_mace(self):
        """Run Lammps calculations."""

        split_datasets = split_dataset(self.inputs.dataset_list)
        train_set = split_datasets["train_set"]
        validation_set = split_datasets["validation_set"]
        test_set = split_datasets["test_set"]
        
        self.report(f"Training set size: {len(train_set.get_list())}")
        self.report(f"Validation set size: {len(validation_set.get_list())}")
        self.report(f"Test set size: {len(test_set.get_list())}")
        
        self.out("validation_set", validation_set)  

        future = self.submit(MaceCalculation,
                             code=self.inputs.code,
                             training_set=train_set,
                             validation_set=validation_set,
                             test_set=test_set,
                             **self.exposed_inputs(MaceCalculation, namespace="mace"))

        self.report(f'Launched MACE calculation <{future.pk}>')
        self.to_context(mace_calculations=append_(future))
  
    """TODO"""	
    def results(self):
        """Process results."""
        inputs = {}

        #count = 0
        #for value in self.ctx.mace_calculations:
        #    count += 1            
        #    self.report(f'Self.ctx.mace_calculations <{value.pk}>')
        #    inputs[f'conf{count}'] = {"aiida_model": value.base.links.get_outgoing().get_node_by_label("aiida_model"), "aiida_swa_model": value.base.links.get_outgoing().get_node_by_label("aiida_swa_model"), "mace": value.base.links.get_outgoing().get_node_by_label("mace")}
        
        #out= WriteDataset(**inputs)
        #aiida_model = out['aiida_model']
        #aiida_swa_model = out['aiida_swa_model']
        #mace = out['mace']

        #self.out("aiida_model", aiida_model)  
        #self.out("aiida_swa_model", aiida_swa_model)  
        #self.out("mace", mace)  
        self.report(f'Self.ctx.mace_calculations 2 <{value.pk}>')
        self.out("aiida_model", self.ctx.mace_calculations.base.links.get_outgoing().get_node_by_label("aiida_model"))  
        self.out("aiida_swa_model", self.ctx.mace_calculations.base.links.get_outgoing().get_node_by_label("aiida_swa_model"))  
        self.out("mace", self.ctx.mace_calculations.base.links.get_outgoing().get_node_by_label("mace"))  

    


    def finalize(self):
        """Finalize."""


        self.out_many(self.exposed_outputs(self.ctx.mace_calculations[0], MaceCalculation, namespace="mace_out"))
        
        
        
        # for ii, val in enumerate(self.ctx.lammps_calculations):
        #     self.out(f'rdf', val.outputs.rdf)
        #     self.out(f'coord_lmmpstrj', val.outputs.coord_lmmpstrj)
        #     self.out(f'msd', val.outputs.msd)
        #     self.out(f'lammps_out', val.outputs.lammps_out)
        #     self.out(f'final_structure', val.outputs.final_structure)
        #     self.out(f'lmp_restart', val.outputs.lmp_restart)
        #     self.out(f'coord_atom', val.outputs.coord_atom)
        #     self.out(f'log_lammps', val.outputs.log_lammps)



    def save_files(self):
        """Create folder and save files."""
        folder = f'{self.inputs.parent_folder.value}/Data/MAVE_pot{self.inputs.dataset_list.pk}'
        os.makedirs(folder, exist_ok=True)
        retrived = self.ctx.mace_calculations[0].get_retrieved_node()
        
        for r in retrived.list_objects():
            self.report(r.name)
            with retrived.open(r.name, 'rb') as handle:
                with open(f'{folder}/{r.name}', "w") as f:
                    try:
                        f.write(handle.read().decode('utf-8'))
                    except:
                        pass
                        
        retrived = self.ctx.mace_calculations[0].outputs.retrieved
        
        for r in retrieved.list_object_names():
            self.report(r.name)
            with retrived.open(r.name, 'rb') as handle:
                with open(f'{folder}/{r.name}', "w") as f:
                    try:
                        f.write(handle.read().decode('utf-8'))
                    except:
                        pass

                        
                

