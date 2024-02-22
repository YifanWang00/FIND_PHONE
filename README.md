# Find Phone
In this project, I incorporated the classical version of YOLO - YOLOv5 for the phone detection task and finetuned the model on the provided dataset. For evaluation, the model achieved 97% of accuracy (based on given matrix) on the dataset, where 80% of the images are used as train set.

## How To Run
1. Create environment
    ```
    conda env create -f environment.yml
    ```

2. Clone YOLOv5

    ```
    git clone https://github.com/ultralytics/yolov5
    ```

3. Download annotated datasets on [google drive](https://drive.google.com/file/d/13PYzavJZmUGVLmyGMgUC_07ykBAlB2Dq/view?usp=share_link) and unzip in the working directory

4. Finetune model with `train_phone_finder.py`
    ```
    python train_phone_finder.py datasets
    ```

5. Get finetuned .pt file from ./yolov5/runs and do inference with `find_phone.py`, or directed use `yolov5_40epochs.pt` provided [here](https://drive.google.com/file/d/18AbAlc5HJY-vjcKzejlMbYWygUNjyTmB/view?usp=share_link)
    ```
    python find_phone.py $IMAGE_PATH
    ```

6. Evaluation on the given dataset
    ```
    python evaluation.py
    ```

## Future work
1. Try to apply the latest YOLO version (YOLOv8) to improve the performance. 
2. Since it is a relatively small dataset, I mannually annotated the images to ensure the accuracy of labeling. But we can use machine learning methods like [this method](https://www.edwardrosten.com/work/rosten_2006_machine.pdf) to conduct edge detection based on given labels, which could deal with larger datasets and are highly automated.
3. To improve the data collection of the customer, we can increase diversity of the data, for example provide more angles of photos to help the model learn to recognize phones in different orientations. Or capture images under a range of lighting conditions (e.g., bright, dim, artificial, natural) to improve robustness against lighting variations.