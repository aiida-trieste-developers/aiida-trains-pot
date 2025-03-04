from pathlib import Path
from aiida.orm import PortableCode
from aiida import load_profile
import aiida_trains_pot
from aiida.orm import Code
import argparse
load_profile()


def install_committee_evaluation(label = 'committee_evaluation_portable', prepend = None, append = None):

    committee_evaluation_path = Path.joinpath(Path(aiida_trains_pot.__path__[0]), 'portable_codes/committee_evaluation/')

    if prepend is None:
        prepend = input("Prepend command (es. source mace_env/bin/activate): ")
    if append is None:
        append = input("Append command: ")

    code = PortableCode(
        label = label,
        filepath_files = committee_evaluation_path,
        prepend_text = f'''{prepend}
function launch() {{
    ./launch $@
}}
export launch''',
        append_text = append,
        filepath_executable = 'launch',
    )

    code.store()

    print(f"Stored code '{code.label}' with pk = {code.pk}")

def check_code_exists(label):
    codes = Code.collection.find()
    for code in codes:
        if code.label == label:
            return code.pk
    else:
        return None
            
    
def main():
    parser = argparse.ArgumentParser(description='Install or list portable codes.')
    parser.add_argument('-l', '--list', action='store_true', help='List existing codes')
    args = parser.parse_args()

    if args.list:
        codes = Code.collection.find()
        print("Existing portable codes:")
        print("    PK     Label")
        print("------------------------")
        for code in codes:
            if 'portable' in code.node_type:
                print(f" {code.pk:7n} - {code.label}")
        return
    print()
    print("Creating a new portable code for committee evaluation")
    print("-----------------------------------------------------")
    print()
    pk_code = 10
    while pk_code is not None:
        label = input("Enter code label (es. committee_evaluation_portable): ")
        pk_code = check_code_exists(label)
        if pk_code is not None:
            print(f"Code '{label}' already exists with pk = {pk_code}")

    install_committee_evaluation(label)

if __name__ == '__main__':

    main()