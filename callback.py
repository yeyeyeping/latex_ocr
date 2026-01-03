from transformers.trainer_callback import TrainerCallback
import json
class LoggingCallback(TrainerCallback):
    def __init__(self, logger):
        super().__init__()
        self.logger = logger
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        _ = logs.pop("total_flos", None)
        if state.is_local_process_zero:
            self.logger.info(json.dumps({
                **logs,
                "step": state.global_step
            }))