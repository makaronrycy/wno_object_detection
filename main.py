import cv2 as cv
import random
import os


#dataset ratio: 70% train (280), 20% val(80), 10% test (40)
class DataSetCreator():
    def __init__(self,background_path,image_path):
        self.backgrounds = [cv.imread(f) for f in os.listdir(background_path) if os.isfile(os.path.join(background_path,f))]
        self.images ={
            "screwdrivers": {
                "class" : 0,
                "images" :[cv.imread(f) for f in os.listdir(os.path.join(image_path,"screwdrivers"))]
            },
            "pliers": {
                "class" : 1,
                "images" :[cv.imread(f) for f in os.listdir(os.path.join(image_path,"pliers"))],
            },
            "swords": {
                "class" : 2,
                "images": [cv.imread(f) for f in os.listdir(os.path.join(image_path,"swords"))],

            }
        }
        

    def create_augmented_image(self,image,class_id):
        for i in range(0,5):
            background: cv.Mat = random.choice(self.backgrounds)
            

    def create_annotation(self,class_ids,position):
        pass
    def create_yaml(self):
        pass
def main():
    DataSetCreator("backgrounds","img")
    pass
if __name__ == "__main__":
    main()
