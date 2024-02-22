import cv2

def load_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def normalize_coord(image, point):
    w, h = image.shape[:2]
    x = round(point[0]/h, 4)
    y = round(point[1]/w, 4)
    return (x, y) 