from .utils import Accumulator
from .nn_model3 import Trainer
class HookBase:
    trainer: Trainer = None
    def __init__(self, name) -> None:
        self.name = name

    def before_train(self) -> None:
        """Called before the first epoch."""
        pass

    def after_train(self) -> None:
        """Called after the last epoch."""
        pass

    def before_epoch(self) -> None:
        """Called before each epoch."""
        pass

    def after_epoch(self) -> None:
        """Called after each epoch."""
        pass

    def before_iter(self) -> None:
        """Called before each iteration."""
        pass

    def after_iter(self) -> None:
        """Called after each iteration."""
        pass

class MetricHook(HookBase):
    def __init__(self, name, calc_fn) -> None:
        self.name = name
        self.calc_fn = calc_fn
    def after_epoch(self, *args, **kwargs) -> None:
        result = self.calc_fn(*args, **kwargs)
        return result

    
class LossHook(HookBase):
    
    def before_epoch(self) -> None:
        self.LossLog = Accumulator(2)

    def after_iter(self) -> None:
        self.LossLog.add(
            float(self.trainer.iter_loss * self.trainer.iter_size),
            self.trainer.iter_size
        )
    
    def after_epoch(self):
        epoch_loss = self.LossLog[0] / self.LossLog[1]

        return epoch_loss
