import torch
from transformers import SegformerForSemanticSegmentation


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class ModelManager:

    def __init__(self):

        self.models = {}

    def load_model(self, model_name, path, num_classes):

        print(f"Loading {model_name} model...")

        model = SegformerForSemanticSegmentation.from_pretrained(
            path,
            num_labels=num_classes
        ).to(DEVICE)

        model.eval()

        self.models[model_name] = model

    def get_model(self, model_name):

        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not loaded")

        return self.models[model_name]


model_manager = ModelManager()