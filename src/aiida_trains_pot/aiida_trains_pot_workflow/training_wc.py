from aiida.engine import WorkChain, ToContext, append_, calcfunction
from aiida.orm import StructureData, Dict, List, Int, Bool, FolderData, Str
from aiida.plugins import WorkflowFactory, DataFactory
from aiida.common import AttributeDict
import random
import itertools
import random
import time


MetaWorkChain   = WorkflowFactory('trains_pot.metatrain')
MaceWorkChain   = WorkflowFactory('trains_pot.macetrain')
PESData         = DataFactory('pesdata')

@calcfunction
def SplitDataset(dataset):
    """Divide dataset into training, validation and test sets."""
    # data = self.inputs.dataset_list.get_list()
    data = dataset.get_list()

    exclude_list = [
        "energy",
        "cell",
        "stress",
        "forces",
        "symbols",
        "positions",
        "id_lammps",
        "input_structure_uuid",
        "sigma_strain",
    ]

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
        if "gen_method" in group_list[0].keys():
            if (
                group_list[0]["gen_method"] == "INPUT_STRUCTURE"
                or group_list[0]["gen_method"] == "ISOLATED_ATOM"
                or len(group_list[0]["positions"]) == 1
                or group_list[0]["gen_method"] == "EQUILIBRIUM"
            ):
                training_set += group_list
                continue
        if "set" in group_list[0].keys():
            if group_list[0]["set"] == "TRAINING":
                training_set += group_list
                continue
            elif group_list[0]["set"] == "VALIDATION":
                validation_set += group_list
                continue
            elif group_list[0]["set"] == "TEST":
                test_set += group_list
                continue
        total_elements = len(group_list)
        training_size = round(0.8 * total_elements)

        random.seed(int(time.time()))
        _ = random.shuffle(group_list)

        # Split the data into sets
        training_set += group_list[:training_size]
        validation_set += group_list[training_size:][::2]
        test_set += group_list[training_size:][1::2]

    for ii in range(len(training_set)):
        training_set[ii]["set"] = "TRAINING"
        if "gen_method" not in training_set[ii].keys():
            training_set[ii]["gen_method"] = "UNKNOWN"
    for ii in range(len(validation_set)):
        validation_set[ii]["set"] = "VALIDATION"
        if "gen_method" not in validation_set[ii].keys():
            validation_set[ii]["gen_method"] = "UNKNOWN"
    for ii in range(len(test_set)):
        test_set[ii]["set"] = "TEST"
        if "gen_method" not in test_set[ii].keys():
            test_set[ii]["gen_method"] = "UNKNOWN"

    pes_training_set = PESData()
    pes_training_set.set_list(training_set)

    pes_validation_set = PESData()
    pes_validation_set.set_list(validation_set)

    pes_test_set = PESData()
    pes_test_set.set_list(test_set)

    pes_global_splitted = PESData()
    pes_global_splitted.set_list(validation_set + test_set + training_set)

    return {
        "train_set": pes_training_set,
        "validation_set": pes_validation_set,
        "test_set": pes_test_set,
        "global_splitted": pes_global_splitted,
    }


class TrainingWorkChain(WorkChain):
    """A workchain to loop over structures and submit MACEWorkChain."""

    @classmethod
    def define(cls, spec):
        """Input and output specification."""
        super().define(spec)
        spec.input("num_potentials", valid_type=Int, default=lambda:Int(1), required=False)   
        spec.input('engine', valid_type=Str, help='Training engine', required=False)
        spec.input("dataset", valid_type=PESData, help="Training dataset",)        
        spec.input_namespace("checkpoints", valid_type=FolderData, required=False, help="Checkpoints file",)
        spec.expose_inputs(MaceWorkChain, namespace="mace",  exclude=('train.training_set', 'train.validation_set', 'train.test_set'), namespace_options={'validator': None})     
        spec.expose_inputs(MetaWorkChain, namespace="meta",  exclude=('train.training_set', 'train.validation_set', 'train.test_set'), namespace_options={'validator': None})  
        spec.output_namespace("training", dynamic=True, help="Training outputs")
        spec.output(
            "global_splitted",
            valid_type=PESData,
        )
        spec.outline(cls.run_training, cls.finalize)

    def run_training(self):
        """Run MACEWorkChain for each structure."""
        split_datasets = SplitDataset(self.inputs.dataset)
        train_set = split_datasets["train_set"]
        validation_set = split_datasets["validation_set"]
        test_set = split_datasets["test_set"]

        self.out("global_splitted", split_datasets["global_splitted"])

        self.report(f"Training set size: {len(train_set.get_list())}")
        self.report(f"Validation set size: {len(validation_set.get_list())}")
        self.report(f"Test set size: {len(test_set.get_list())}")

        if self.inputs.engine.value == "MACE":

            inputs = self.exposed_inputs(MaceWorkChain, namespace="mace")

            inputs.train["training_set"] =  train_set
            inputs.train["validation_set"] =  validation_set
            inputs.train["test_set"] =  test_set

            
            if 'checkpoints' in self.inputs:
                inputs['checkpoints'] = self.inputs.checkpoints
                inputs.train['restart'] = Bool(True)

            if 'checkpoints' in inputs:
                chkpts = list(dict(inputs.checkpoints).values())
            
            for ii in range(self.inputs.num_potentials.value):            
                if 'checkpoints' in self.inputs and ii < len(chkpts):
                    inputs.train["checkpoints"] = chkpts[ii]
                
                inputs.train["index_pot"] = Int(ii)
                future = self.submit(MaceWorkChain, **inputs)
                self.to_context(mace_wc = append_(future))        
            pass

        if self.inputs.engine.value == "META":

            inputs = self.exposed_inputs(MetaWorkChain, namespace="meta")

            inputs.train["training_set"] =  train_set
            inputs.train["validation_set"] =  validation_set
            inputs.train["test_set"] =  test_set

            
            if 'checkpoints' in self.inputs:
                inputs['checkpoints'] = self.inputs.checkpoints
                inputs.train['restart'] = Bool(True)

            if 'checkpoints' in inputs:
                chkpts = list(dict(inputs.checkpoints).values())
            
            for ii in range(self.inputs.num_potentials.value):            
                if 'checkpoints' in self.inputs and ii < len(chkpts):
                    inputs.train["checkpoints"] = chkpts[ii]
                
                inputs.train["index_pot"] = Int(ii)
                future = self.submit(MetaWorkChain, **inputs)
                self.to_context(meta_wc = append_(future))        
            pass

    def finalize(self):
        """Collect and output results from all MACEWorkChain instances."""
        results = {}

        if self.inputs.engine.value == "MACE":
            for ii, calc in enumerate(self.ctx.mace_wc):
                results[f'mace_{ii}']={}
                for el in calc.outputs:       
                    results[f'mace_{ii}'][el] = calc.outputs[el]
                    
                self.out('training', results)

        if self.inputs.engine.value == "META":
            for ii, calc in enumerate(self.ctx.meta_wc):
                results[f'meta_{ii}']={}
                for el in calc.outputs:       
                    results[f'meta_{ii}'][el] = calc.outputs[el]
                    
                self.out('training', results)
        
