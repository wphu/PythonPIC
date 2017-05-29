import os

from ..classes import load_simulation


def config_filename(run_name, category_name=None):
    return f"data_analysis/{category_name+'/' if category_name else ''}{run_name}/{run_name}.hdf5"


def try_run(run_name, category_name, config_function, *args, **kwargs):
    full_path = config_filename(run_name, category_name)
    print(f"Path is {full_path}")
    file_exists = os.path.isfile(full_path)
    if file_exists:
        print("Found file. Attempting to load...")
        try:
            s = load_simulation(full_path)
        except KeyError as err:
            print(err)
            print("Running full sim.")
            s = config_function(full_path, *args, **kwargs)
        print("Managed to load file.")
    else:
        print("Running simulation")
        s = config_function(full_path, *args, **kwargs)
    return s
