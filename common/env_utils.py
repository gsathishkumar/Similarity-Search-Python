import os

def get_bool_env(var_name, default=False):
    """
    Reads an environment variable and converts it to a boolean.
    Returns the default value if the variable is unset.
    """
    value = os.getenv(var_name)
    if value is None:
        return default
    # Convert to lowercase and check against common true values
    return value.lower() in ('true', '1', 'yes', 'on')

def get_str_env(var_name, default=''):
    """
    Reads an environment variable and returns the value as string.
    Returns the default value if the variable is unset.
    """
    return str(os.getenv(var_name, default))

def get_int_env(var_name, default=0):
    """
    Reads an environment variable and returns the value as int.
    Returns the default value if the variable is unset.
    """
    return int(os.getenv(var_name, default))