import cv2 as cv

import os

#dataset ratio: 70% train (280), 20% val(80), 10% test (40)
class DataSetCreator():
    def __init__(self,background_path,image_path):
        self.backgrounds = [f for f in os.listdir(background_path) if os.isfile(os.join(background_path,f))]
        self.images = [f for f in os.listdir(image_path) if os.isfile(os.join(image_path,f))]
        self.classes
    def create_augmented_image():
        pass
    def create_annotation():
        pass
    def create_yaml():
        pass
def main():
    
    pass
if __name__ == "__main__":
    main()
