from pathlib import Path
from aiida.orm import PortableCode
from aiida import load_profile
import aiida_trains_pot
from aiida.orm import Code
load_profile()


def install_committee_evaluation():

    committee_evaluation_path = Path.joinpath(Path(aiida_trains_pot.__path__[0]), 'portable_codes/committee_evaluation/')

    prepend = input("Prepend command (es. source mace_env/bin/activate): ")
    append = input("Append command: ")

    code = PortableCode(
        label = 'committee_evaluation_portable',
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

    print(f"Stored code '{code.label}' with pk={code.pk}")

def main():
    codes = Code.collection.find()
    for code in codes:
        if code.label == 'committee_evaluation_portable':
            raise ValueError(f"'committee_evaluation_portable' code already exists with pk = {code.pk}")
        
    install_committee_evaluation()

if __name__ == '__main__':
    main()