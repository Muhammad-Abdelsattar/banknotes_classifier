import lightning.pytorch.callbacks as callbacks
from dvclive.lightning import DVCLiveLogger

def build_callbacks(config: dict):
    callbacks_list = []
    for callback,args in config.items():
        callbacks_list.append(getattr(callbacks, callback)(**args))
    return callbacks_list


def build_logger(config: dict):
    return DVCLiveLogger(**config)