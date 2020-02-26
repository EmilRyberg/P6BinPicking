import argparse
import json
import os
import glob
import math
import random
import datetime
import shutil


class COCODatasetSplitter:
    test_split = 0
    val_split = 0
    train_split = 0
    dataset_path = ""
    split_folder = "dataset_splitted"

    def __init__(self, dataset_path, test_split, val_split):
        self.test_split = test_split
        self.val_split = val_split
        self.train_split = 1.0 - test_split - val_split
        self.dataset_path = dataset_path
        if not os.path.isdir(dataset_path):
            raise Exception("Dataset path is not a directory.")

    def split_dataset(self):
        if not os.path.exists(self.split_folder):
            os.mkdir(self.split_folder)
        coco_json_file_names = glob.glob(f"{self.dataset_path}/*.json")
        if len(coco_json_file_names) == 0:
            raise Exception(f"No .json files found in directory {self.dataset_path}")
        elif len(coco_json_file_names)  > 1:
            print(f"More than 1 JSON file in directory, grabbing first: {coco_json_file_names[0]}")
        coco_json_file_name = coco_json_file_names[0]
        image_names = glob.glob(f"{self.dataset_path}/*.png")
        image_names.extend(glob.glob(f"{self.dataset_path}/*.jpg"))

        coco = None
        with open(coco_json_file_name, 'r') as coco_file:
            coco = json.load(coco_file)
        image_mappings = coco['images']
        train_indexes, test_indexes, val_indexes = self.__determine_id_splits(len(image_mappings))
        print(f"train index count: {len(train_indexes)}, test: {len(test_indexes)}, val: {len(val_indexes)}")

        self.__create_single_split(coco, train_indexes, "Training set", "train")
        self.__create_single_split(coco, test_indexes, "Test set", "test")
        self.__create_single_split(coco, val_indexes, "Validation set", "val")

    def __create_single_split(self, coco, indexes, set_name, folder_and_json_name):
        image_mappings = coco['images']

        split_coco = {
            "info": {
                "year": 2020,
                "version": 1.0,
                "description": set_name,
                "url": "www.YEET.nope",
                "date_created": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "Z"
            },
            "images": [],
            "annotations": [],
            "categories": coco["categories"],
            "licenses": coco["licenses"]
        }

        for index in indexes:
            split_coco["images"].append(image_mappings[index])
        split_image_ids = [image["id"] for image in split_coco["images"]]

        split_coco["annotations"] = [annotation for annotation in coco["annotations"] if annotation["image_id"] in split_image_ids]

        print(f"Saving JSON file for set '{set_name}'")
        split_dir = f"{self.split_folder}/{folder_and_json_name}"
        if not os.path.exists(split_dir):
            os.mkdir(split_dir)
        with open(f"{split_dir}/{folder_and_json_name}.json", "w+") as split_file:
            json.dump(split_coco, split_file)

        print(f"Copying image files for set '{set_name}'")
        for image in split_coco["images"]:
            shutil.copy2(f"{self.dataset_path}/{image['file_name']}", split_dir)

    def __determine_id_splits(self, total_length):
        taken_ids = []

        def get_indexes(amount):
            indexes = []
            for j in range(amount):
                found_unique_index = False
                while not found_unique_index:
                    index = random.randint(0, total_length - 1)
                    if index not in taken_ids:
                        taken_ids.append(index)
                        indexes.append(index)
                        found_unique_index = True
            return indexes

        # val
        val_indexes = get_indexes(math.ceil(total_length * self.val_split))
        test_indexes = get_indexes(math.ceil(total_length * self.test_split))

        train_indexes = []
        for i in range(total_length):
            if i not in taken_ids:
                train_indexes.append(i)

        return train_indexes, test_indexes, val_indexes


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_dir", help="The path to the COCO JSON file and images.")
    parser.add_argument("--test_split", type=float, help="Test split percentage (float)", default=0.2)
    parser.add_argument("--val_split", type=float, help="Validation split percentage (float)", default=0.1)

    args = parser.parse_args()
    splitter = COCODatasetSplitter(args.dataset_dir, args.test_split, args.val_split)
    splitter.split_dataset()

