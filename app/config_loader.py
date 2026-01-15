import yaml


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        config = yaml.safe_load(f)

    if config is None:
        raise ValueError("Config file loaded as None. Check YAML formatting.")

    return config