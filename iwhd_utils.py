from typing import Tuple


def get_stressor_param_from_directory_name(dir_name: str) -> Tuple[str, str]:
    """Gets stressor paramater number. Copied from score_all_trees.py

    Stressors appear as directories with an alphanumeric name - the first set of
    characters correspond to the stressor and the numbers correspond to the
    parameter. For example "char10" would indicate that this directory stores
    results from benchmarks with 10 characters. This function separates the
    stressor name and parameter.

    Args:
        filename: Stressor directory name.

    Returns:
        Stressor name and parameter.
    """
    param = ""
    stressor_name = ""
    for character in dir_name:
        if character.isdigit():
            param += character
        else:
            stressor_name += character

    return stressor_name, param