import yaml

class TrainingArguments:
    learning_rate: float = 0.001
    train_batch_size: int = 128
    eval_batch_size: int = 512
    num_workers: int = 2
    eval_ratio: float = 0.2
    estp_delta: float = 0.0
    estp_patience: int = 7

    def update(self, **kwargs):
        #配置文件的值会被覆盖
        args_dict = {}
        if "config_path" in kwargs.keys():
            with open(kwargs["config_path"], "r") as f:
                args_dict = yaml.safe_load(f)
                assert args_dict is not None, "there is nothing in the configure file"
            kwargs.pop("config_path")

        args_dict.update(kwargs)
        for k, v in args_dict.items():
            setattr(self, k, v)
        
    def add_argument(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def print_args(self):
        args = [a for a in dir(self) if not a.startswith('__') and not callable(getattr(self, a))]
        args_dict = {a: getattr(a) for a in args}

        return args_dict