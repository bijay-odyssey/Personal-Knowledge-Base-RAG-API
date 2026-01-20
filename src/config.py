import yaml

class Config:
    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

    def get(self, key: str, default=None):
        return self.config.get(key, default)

    def get_embedding_model(self) -> str:
        return self.config["embedding"]["model"]
