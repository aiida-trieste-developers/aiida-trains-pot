# -*- coding: utf-8 -*-
"""Equation of State WorkChain."""
from aiida.engine import WorkChain, append_, calcfunction, workfunction
from aiida import load_profile
from aiida.orm import Code, Dict, Int, List, FolderData, SinglefileData
from aiida.plugins import CalculationFactory
import random
import itertools
from ase.io import write
import os
import time
import io
from contextlib import redirect_stdout

load_profile()

MaceCalculation = CalculationFactory('NNIPdevelopment.macetrain')

@calcfunction
def SplitDataset(dataset):
    """Divide dataset into training, validation and test sets."""
    # data = self.inputs.dataset_list.get_list()
    data = dataset.get_list()

    exclude_list = ["energy", "cell", "stress", "forces", "symbols", "positions"]
    # Define a function to extract the grouping key
    def check_esclude_list(string):
        for el in exclude_list:
             if el in string:
                 return False
        return True
    
    def get_grouping_key(d):
        return tuple((k, v) for k, v in d.items() if check_esclude_list(k))

    # Sort the data based on the grouping key
    sorted_data = sorted(data, key=get_grouping_key)

    # Group the sorted data by the grouping key
    grouped_data = itertools.groupby(sorted_data, key=get_grouping_key)

    # Iterate over the groups and print the group key and the list of dictionaries in each group
    training_set = []
    validation_set = []
    test_set = []

    for _, group in grouped_data:
    # Calculate the number of elements for each set
        group_list = list(group)
        # print(group_list[0]['gen_method'])
        # if 'n_vacancies' in group_list[0].keys() and 'sigma_strain' in group_list[0].keys() and 'rattle_radius' in group_list[0].keys() and 'gen_method' in group_list[0].keys() and 'positions' in group_list[0].keys():
            
        #     if group_list[0]['n_vacancies'] == 0 and group_list[0]['sigma_strain'] == 1.0 and group_list[0]['rattle_radius'] == 0.0 and group_list[0]['gen_method'] != "EQUILIBRIUM" and len(group_list[0]['positions']) > 1:
        #         # print(group_list[0]['gen_method'], group_list[0]['energy'], len(group_list[0]['positions']))
        #         continue

        if group_list[0]['gen_method'] == "INPUT_STRUCTURE" or group_list[0]['gen_method'] == "ISOLATED_ATOM" or len(group_list[0]['positions']) == 1 or group_list[0]['gen_method'] == "EQUILIBRIUM":
                training_set += group_list
                continue
        elif 'set' in group_list[0].keys():
            if group_list[0]['set'] == 'TRAINING':
                training_set += group_list
                continue
            elif group_list[0]['set'] == 'VALIDATION':
                validation_set += group_list
                continue
            elif group_list[0]['set'] == 'TEST':
                test_set += group_list
                continue
        total_elements = len(group_list)
        training_size = int(0.8 * total_elements)
        test_size = int(0.1 * total_elements)
        validation_size = total_elements - training_size - test_size
        
        random.seed(int(time.time()))
        _ = random.shuffle(group_list)


        # Split the data into sets
        training_set += group_list[:training_size]
        validation_set += group_list[training_size:training_size+validation_size]
        test_set += group_list[training_size+validation_size:]

    for ii in range(len(training_set)):
        training_set[ii]['set'] = 'TRAINING'
    for ii in range(len(validation_set)):
        validation_set[ii]['set'] = 'VALIDATION'
    for ii in range(len(test_set)):
        test_set[ii]['set'] = 'TEST'


    return {"train_set":List(training_set), "validation_set":List(validation_set), "test_set":List(test_set), "global_splitted":List(training_set+validation_set+test_set)}


class MaceTrainWorkChain(WorkChain):
    """WorkChain to launch MACE training."""

    @classmethod
    def define(cls, spec):
        """Specify inputs and outputs."""
        super().define(spec)

        spec.expose_inputs(MaceCalculation, namespace="mace", exclude=('training_set','validation_set','test_set', 'checkpoints'), namespace_options={'validator': None})
        spec.output_namespace("mace", dynamic=True)

        # spec.input("code", valid_type=Code)
        spec.input("dataset_list", valid_type=List)
        #spec.input("mace_config", valid_type=Dict, help="Config parameters for MACE")
        spec.input("num_potentials", valid_type=Int, default=lambda:Int(1), required=False)
        spec.input_namespace("checkpoints", valid_type=FolderData, required=False, help="Checkpoints file",)

        spec.output("global_list_splitted", valid_type=List, help="List of configurations splitted into sets")
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

        split_datasets = SplitDataset(self.inputs.dataset_list)
        train_set = split_datasets["train_set"]
        validation_set = split_datasets["validation_set"]
        test_set = split_datasets["test_set"]

        self.global_splitted=split_datasets["global_splitted"]
        
        self.report(f"Training set size: {len(train_set.get_list())}")
        self.report(f"Validation set size: {len(validation_set.get_list())}")
        self.report(f"Test set size: {len(test_set.get_list())}")
        
        # Make sure the path to preprocess.py is absolute
        #preprocess_script_path = os.path.abspath('../../aiida_scripts/src/NNIPdevelopment/mace/mace_train_wc/preprocess_config.py')
        preprocess_script_path = os.path.abspath('/home/nataliia/Documents/aiida_scripts/src/NNIPdevelopment/mace/mace_train_wc/preprocess_config.py')
        preprocess_script_file = SinglefileData(file=preprocess_script_path) 
    
        if 'checkpoints' in self.inputs:
            chkpts = list(dict(self.inputs.checkpoints).values())
 
        for ii in range(self.inputs.num_potentials.value):
            inputs = self.exposed_inputs(MaceCalculation, namespace="mace")
            inputs["training_set"] = train_set
            inputs["validation_set"] = validation_set
            inputs["test_set"] = test_set
            inputs["preprocess_script"] = preprocess_script_file

            if 'checkpoints' in self.inputs and ii < len(chkpts):
                inputs["checkpoints"] = chkpts[ii]
            future = self.submit(MaceCalculation, **inputs)


            self.report(f'Launched MACE calculation <{future.pk}>')
            self.to_context(mace_calculations=append_(future))

    def finalize(self):
        """Finalize."""
        potentials = {}
        for ii, calc in enumerate(self.ctx.mace_calculations):
            potentials[f'mace_{ii}']={}
            for out in calc.outputs:
                potentials[f'mace_{ii}'][out] = calc.outputs[out]

            self.out('mace', potentials)
        self.out("global_list_splitted", self.global_splitted)

