import argparse
import os
import subprocess
import yaml


img_size = 640
epochs = 40
weights = "yolov5s.pt"
root_path = os.getcwd()

def create_train_config(train_config):
    filename = os.path.join(root_path, 'yolov5/data/find_phone_training_config.yaml')
    with open(filename, 'w') as file:
        yaml.dump(train_config, file, default_flow_style=False, sort_keys=False)

    print(f'YAML file "{filename}" has been created.')
    return filename

def train_model(yaml_path):
    yolo_path = os.path.join(root_path, "yolov5")
    os.chdir(yolo_path)
    cmd = [
        "python", "train.py",
        "--img", f"{img_size}",
        "--epochs", f"{epochs}",
        "--data", yaml_path,
        "--weights", f"{weights}"
    ]

    result = subprocess.run(cmd, text=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a phone finder model.")
    parser.add_argument("data_dir", type=str,  help="Path to the folder with labeled images and labels.txt")
    args = parser.parse_args()

    if not os.path.isdir(args.data_dir):
        raise ValueError(f"The path {args.data_dir} is not a valid directory.")

    config = {
        'path': os.path.join("..", args.data_dir),  # dataset root dir
        'train': 'train/images',  # train images (relative to 'path') 128 images
        'val': 'valid/images',  # val images (relative to 'path') 128 images
        'test': 'test/images',  # test images (optional)
        'names': {
            0: 'phone'
        }
    }
    
    yaml_path = create_train_config(config)
    train_model(yaml_path)