import hydra
from omegaconf import OmegaConf, DictConfig

@hydra.main(version_base=None ,config_path="conf", config_name="config")
def train(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    




if __name__ == "__main__":
    train()