from configobj import ConfigObj
from validate import Validator

def read_config(config_file, config_spec):
    """
    read config files and return a dict
    :param config_file: config file
    :param config_spec: config spec. file
    :return: dict of configs
    """
    configspec = ConfigObj(config_spec, raise_errors=True)
    config = ConfigObj(
        config_file,
        configspec=configspec,
        raise_errors=True,
        file_error=True
    )
    config.validate(Validator())
    return config

def write_config(config_file, config_spec, section, option, val):
    configspec = ConfigObj(config_spec, raise_errors=True)
    config = ConfigObj(
        config_file,
        config_spec=configspec,
        raise_errors=True,
        file_error=True
    )
    try:
        config[section][option] = val
        config.validate(Validator())
        config.write()
        return True
    except:
        return False
