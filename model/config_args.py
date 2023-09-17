import yaml

class Arguments:
    """
    An argument class that supports loading a YAML configuration file.
    """
    def update(self, **kwargs) -> None:
        """
        Update arguments. 
        
        If `config_path` is provided, the arguments will be updated from 
        the config file first, which means that the value settings in the config file will 
        be overwritten by other arguments provided here.
        """
        args_dict = {}
        if "config_path" in kwargs.keys():
            with open(kwargs["config_path"], "r") as f:
                args_dict = yaml.safe_load(f)
                assert args_dict is not None, "there is nothing in the configure file"
            kwargs.pop("config_path")

        args_dict.update(kwargs)
        for k, v in args_dict.items():
            setattr(self, k, v)
        
    def add_arguments(self, **kwargs) -> None:
        """
        Add arguments.
        """
        for k, v in kwargs.items():
            setattr(self, k, v)

    def print_arguments(self) -> dict:
        """
        Return the current arguments configuration.
        """
        args = [a for a in dir(self) if not a.startswith('__') and not callable(getattr(self, a))]
        args_dict = {a: getattr(self, a) for a in args}

        return args_dict


class TrainingArguments(Arguments):
    learning_rate: float = 0.001
    train_batch_size: int = 128
    eval_batch_size: int = 512
    num_workers: int = 0
    eval_ratio: float = 0.2
    estp_delta: float = 0.0
    estp_patience: int = 7


