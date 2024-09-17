from pathlib import Path
from aiida.orm import PortableCode
from aiida import load_profile
import aiida_trains_pot


def install_cometee_evaluation():

    cometee_evaluation_path = Path.joinpath(Path(aiida_trains_pot.__path__[0]), 'portable_codes/evaluation/')

    load_profile()

    code = PortableCode(
        label = 'cometee_evaluation_portable',
        filepath_files = cometee_evaluation_path,
        filepath_executable = 'launch.sh',
    )

    code.store()

    print(f"Stored code '{code.label}' with pk={code.pk}")

def main():
    install_cometee_evaluation()


if __name__ == '__main__':
    main()