import os
from datetime import datetime

from mltu.configs import BaseModelConfigs


class ModelConfigs(BaseModelConfigs):
    def __init__(self):
        super().__init__()
        self.model_path = os.path.join("Models/08_handwriting_recognition_torch", datetime.strftime(datetime.now(), "%Y%m%d%H%M"))
        self.vocab = ""
        self.height = 64
        self.width = 256
        self.max_text_length = 0
        self.batch_size = 32
        self.learning_rate = 0.01 ### 0.01
        self.train_epochs = 12  ###400