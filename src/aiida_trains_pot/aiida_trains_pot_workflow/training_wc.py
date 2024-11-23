from aiida.engine import WorkChain, ToContext, append_, calcfunction
from aiida.orm import StructureData, Dict, List, Int, Bool, FolderData
from aiida.plugins import WorkflowFactory, DataFactory
from aiida.common import AttributeDict
import random
import itertools
import time

MaceWorkChain   = WorkflowFactory('trains_pot.macetrain')
PESData         = DataFactory('pesdata')

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

    pes_training_set = PESData()    
    pes_training_set.set_list(training_set)    

    pes_validation_set = PESData()    
    pes_validation_set.set_list(validation_set)  

    pes_test_set = PESData()    
    pes_test_set.set_list(test_set)  

    pes_global_splitted = PESData()    
    pes_global_splitted.set_list(training_set+validation_set+test_set)  
    
    return {"train_set":pes_training_set, "validation_set":pes_validation_set, "test_set":pes_test_set, "global_splitted":pes_global_splitted}

 
class TrainingWorkChain(WorkChain):
    """A workchain to loop over structures and submit MACEWorkChain."""

    @classmethod
    def define(cls, spec):
        super().define(spec)
        spec.input("num_potentials", valid_type=Int, default=lambda:Int(1), required=False)   
        spec.input("dataset", valid_type=PESData, help="Training dataset list",)        
        spec.input_namespace("checkpoints", valid_type=FolderData, required=False, help="Checkpoints file",)
        spec.expose_inputs(MaceWorkChain, namespace="mace",  exclude=('train.training_set', 'train.validation_set', 'train.test_set'), namespace_options={'validator': None})     
        spec.output_namespace("training", dynamic=True, help="Training outputs")
        spec.output("global_splitted", valid_type=PESData,)        
        spec.outline(            
            cls.run_training,
            cls.finalize            
        )        
        

            

    def run_training(self):
        """Run MACEWorkChain for each structure."""

        split_datasets = SplitDataset(self.inputs.dataset)
        train_set = split_datasets["train_set"]
        validation_set = split_datasets["validation_set"]
        test_set = split_datasets["test_set"]

        
        self.out('global_splitted', split_datasets["global_splitted"])

        
        self.report(f"Training set size: {len(train_set.get_list())}")
        self.report(f"Validation set size: {len(validation_set.get_list())}")
        self.report(f"Test set size: {len(test_set.get_list())}") 

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

    def finalize(self):        
        results = {}
        for ii, calc in enumerate(self.ctx.mace_wc):
            results[f'mace_{ii}']={}
            for el in calc.outputs:       
                results[f'mace_{ii}'][el] = calc.outputs[el]
                
            self.out('training', results)
        