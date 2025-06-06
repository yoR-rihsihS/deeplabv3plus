import cv2
import json
import numpy as np
from pathlib import Path

NAME_TO_LABELID = json.load(open("./name_to_labelId.json", 'r'))
NAME_TO_TRAINID = json.load(open("./name_to_trainId.json", 'r'))
NAME_TO_COLOR = json.load(open("./name_to_color.json", 'r'))

# train id = -1 is to be ignored
# train id = 255 is background
# to make the train id sequential we will change train id = 255 to 19
for name, train_id in NAME_TO_TRAINID.items():
    if train_id == 255:
        NAME_TO_TRAINID[name] = 19

GEN_PATH = Path("../data/generated_gt/")
ANNOT_PATH = Path("../data/gtFine/")
IMAGE_PATH = Path("../data/leftImg8bit/")

TOTAL_ACCURACY = 0
TOTAL_SAMPLES = 0

def make_segmentation_mask(annot_file_path, color_output_path, labelid_output_path, trainid_output_path):
    with open(annot_file_path, 'r') as f:
        json_data = json.load(f)

    height = json_data["imgHeight"]
    width = json_data["imgWidth"]

    color_mask = np.zeros((height, width, 3), dtype=np.uint8)
    labelid_mask = np.zeros((height, width), dtype=np.uint8)
    trainid_mask = np.zeros((height, width), dtype=np.uint8)

    objects = json_data["objects"]
    
    for obj in objects:
        label = obj["label"]
        polygon = np.array([obj["polygon"]], dtype=np.int32)
        if label == 'license plate':
            # don't draw license plate
            continue
        if label not in NAME_TO_LABELID and label.endswith('group'):
            label = label[:-len('group')]
            
        if label in NAME_TO_LABELID:
            cv2.fillPoly(color_mask, polygon, NAME_TO_COLOR[label])
            cv2.fillPoly(labelid_mask, polygon, NAME_TO_LABELID[label])
            cv2.fillPoly(trainid_mask, polygon, NAME_TO_TRAINID[label])
        else:
            raise ValueError(f"Label '{label}' not found in label_to_id mapping.")

    cv2.imwrite(str(color_output_path), cv2.cvtColor(color_mask, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(labelid_output_path), labelid_mask)
    cv2.imwrite(str(trainid_output_path), trainid_mask)

def compare_masks(original_mask_path, generated_mask_path):
    global TOTAL_ACCURACY, TOTAL_SAMPLES

    original_mask = cv2.imread(str(original_mask_path), cv2.IMREAD_GRAYSCALE)
    generated_mask = cv2.imread(str(generated_mask_path), cv2.IMREAD_GRAYSCALE)

    if original_mask is None or generated_mask is None:
        raise FileNotFoundError(f"One of the mask files was not found: {original_mask_path}, {generated_mask_path}")

    if original_mask.shape != generated_mask.shape:
        raise ValueError(f"Shape mismatch: {original_mask.shape} vs {generated_mask.shape}")

    total_pixels = original_mask.size
    diff_pixels = np.sum(original_mask != generated_mask)
    matching_pixels = total_pixels - diff_pixels

    TOTAL_ACCURACY += (matching_pixels / total_pixels) * 100
    TOTAL_SAMPLES += 1

# runs for about 5 minutes
for split in ['train', 'val']:
    split_path = ANNOT_PATH / split
    for city_folder in split_path.iterdir():
        if city_folder.is_dir():
            for file in city_folder.iterdir():
                if file.suffix == ".json":
                    # relative_path is in the form {split}/{city}/{file.json}
                    relative_path = file.relative_to(ANNOT_PATH)
                    
                    # file.name is the json file name without the path {name_gtFine_polygons.json}
                    color_mask_name = file.name.replace("gtFine_polygons", "color").replace(".json", ".png")
                    labelid_mask_name = file.name.replace("gtFine_polygons", "labelId").replace(".json", ".png")
                    trainid_mask_name = file.name.replace("gtFine_polygons", "trainId").replace(".json", ".png")
                    
                    color_output_path = GEN_PATH / relative_path.parent / color_mask_name
                    labelid_output_path = GEN_PATH / relative_path.parent / labelid_mask_name
                    trainid_output_path = GEN_PATH / relative_path.parent / trainid_mask_name

                    # check if the image file exists
                    image_name = file.name.replace("gtFine_polygons", "leftImg8bit").replace(".json", ".png")
                    image_path = IMAGE_PATH / relative_path.parent / image_name
                    if not Path.exists(image_path):
                        raise ValueError(f"Image file {image_path} does not exist.")

                    # create parent directory for output file if it doesn't exist
                    color_output_path.parent.mkdir(parents=True, exist_ok=True)

                    make_segmentation_mask(ANNOT_PATH / relative_path, color_output_path, labelid_output_path, trainid_output_path)

                    # compare the generated label mask with the original label mask
                    original_label_name = file.name.replace("polygons", "labelIds").replace(".json", ".png")
                    original_label_path = ANNOT_PATH / relative_path.parent / original_label_name
                    compare_masks(original_label_path, labelid_output_path)


# We are only checking the accuracy of the label id masks
# and not the train id masks or the color masks
# The accuracy is calculated as the percentage of matching pixels
print(f"Average accuracy: {TOTAL_ACCURACY / TOTAL_SAMPLES:.4f}%")