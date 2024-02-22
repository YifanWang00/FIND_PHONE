import argparse
import os
import torch
from util_funcs import load_image, normalize_coord

model_path = "./yolov5_40epochs.pt"

def find_phone(img_path):
    print(f"Finding the phone in the image: {img_path}")
    model = torch.hub.load("ultralytics/yolov5", 'custom', path=model_path)
    img = load_image(img_path)

    result = model(img)
    result_df = result.pandas().xyxy[0]
    if (result_df.empty):
        return (0.5, 0.5)
    
    max_confidence_index = result_df['confidence'].idxmax()
    max_confidence_row = result_df.loc[max_confidence_index]
    xmin = max_confidence_row['xmin']
    ymin = max_confidence_row['ymin']
    xmax = max_confidence_row['xmax']
    ymax = max_confidence_row['ymax']

    center = ((xmin+xmax)/2, (ymin+ymax)/2)

    return normalize_coord(img, center)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Find the phone location in the given image.")
    parser.add_argument("img_path", type=str,  help="Path to the target image")
    args = parser.parse_args()

    if not os.path.isfile(args.img_path):
        raise ValueError(f"The path {args.img_path} is not a valid directory.")
    
    position = find_phone(args.img_path)
    print(position[0], position[1])