import argparse
import json

class COCODatasetSplitter:
    test_split = 0
    val_split = 0
    train_split = 0
    file_path = 0

    def __init__(self, file_path, test_split, val_split):
        self.test_split = test_split
        self.val_split = val_split
        self.train_split = 1.0 - test_split - val_split
        self.file_path = file_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("coco_file", help="The path to the COCO JSON file.")
    parser.add_argument("--test_split", type=float, help="Test split percentage (float)", default=0.2)
    parser.add_argument("--val_split", type=float, help="Validation split percentage (float)", default=0.1)

