import argparse

class BaseParser():
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    def parse(self):
        self.parser.add_argument("--mode", default="train", choices=["train", "test"])
        self.parser.add_argument("--config", default="./config.yaml", help="path to config")
        self.parser.add_argument("--checkpoint", default=True,help="path to checkpoint to restore")
        return self.parser.parse_args()
