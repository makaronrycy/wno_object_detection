import cv2 as cv
import random
import os
import numpy as np

background_path = "backgrounds"
image_path = "img"
#dataset ratio: 70% train (280), 20% val(80), 10% test (40)
class DataSetCreator():
    def __init__(self,background_path,image_path,max_images=400,max_images_per_background=5):
        self.backgrounds = [cv.imread(os.path.join(background_path,f)) for f in os.listdir(background_path) if os.path.isfile(os.path.join(background_path,f))]
        self.max_images = max_images
        self.max_images_per_background = max_images_per_background
        self.images ={
            0: {

                "images" :[cv.imread(os.path.join(image_path,"screwdrivers",f)) for f in os.listdir(os.path.join(image_path,"screwdrivers"))]
            },
            1: {
                "images" :[cv.imread(os.path.join(image_path,"pliers",f)) for f in os.listdir(os.path.join(image_path,"pliers"))],
            },
            2: {
                "images": [cv.imread(os.path.join(image_path,"swords",f)) for f in os.listdir(os.path.join(image_path,"swords"))],

            }
        }
        
    def create_dataset(self):
        train_images = int(self.max_images * 0.7)
        val_images = int(self.max_images * 0.2)
        test_images = int(self.max_images * 0.1)
        print(f"Train images: {train_images}, Val images: {val_images}, Test images: {test_images}")
        # Create directories
        if not os.path.exists("dataset"):
            os.makedirs("dataset")
        if not os.path.exists("dataset/images"):
            os.makedirs("dataset/images")
        if not os.path.exists("dataset/labels"):
            os.makedirs("dataset/labels")

        for i in range(train_images):
            self.create_image_composite(i,"train",True)
        for i in range(train_images,train_images+val_images):
            self.create_image_composite(i,"val",True)
        for i in range(train_images+val_images,self.max_images):
            self.create_image_composite(i,"test",True)
    def create_image_composite(self,index,dataset_type,annotate=False,example_image=False):
        background = random.choice(self.backgrounds)
        background = cv.resize(background,(640,480))
        
        # Apply background augmentations (global effects)
        if random.random() < 0.3:
            # Adjust brightness
            brightness = random.uniform(0.7, 1.3)
            background = cv.convertScaleAbs(background, alpha=brightness, beta=0)
        
        if random.random() < 0.3:
            # Adjust contrast
            contrast = random.uniform(0.7, 1.3)
            mean = np.mean(background)
            background = cv.convertScaleAbs(background, alpha=contrast, beta=mean*(1-contrast))
        
        if random.random() < 0.2:
            # Add gaussian noise
            noise = np.random.normal(0, random.uniform(5, 15), background.shape).astype(np.uint8)
            background = cv.add(background, noise)
        
        if random.random() < 0.2:
            # Add gaussian blur
            blur_size = random.choice([3, 5, 7])
            background = cv.GaussianBlur(background, (blur_size, blur_size), 0)
            
        annotation = {}
        for j in range(random.randint(1,self.max_images_per_background)):
            class_id = random.randint(0,2)
            image = random.choice(self.images[class_id]["images"]) 
            
            # Convert image to BGR if it has alpha channel
            has_alpha = False
            if image.shape[2] == 4:
                has_alpha = True
                # Keep alpha channel for now
                alpha_channel = image[:,:,3]
                image_bgr = cv.cvtColor(image, cv.COLOR_BGRA2BGR)
            else:
                image_bgr = image.copy()
        
            # Apply foreground object augmentations
            if random.random() < 0.3:
                # Color jitter - hue shift
                hsv = cv.cvtColor(image_bgr, cv.COLOR_BGR2HSV)
                # Convert to int16 first to handle negative values
                h = hsv[:,:,0].astype(np.int16)  
                h = (h + random.randint(-10, 10)) % 180
                hsv[:,:,0] = h.astype(np.uint8)  # Convert back to uint8
                image_bgr = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
                
            if random.random() < 0.3:
                # Adjust saturation
                hsv = cv.cvtColor(image_bgr, cv.COLOR_BGR2HSV)
                hsv[:,:,1] = np.clip(hsv[:,:,1] * random.uniform(0.7, 1.3), 0, 255).astype(np.uint8)
                image_bgr = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
                
            # Create mask for black and white backgrounds
            # Detect pixels that are either black or white (with some tolerance)
            mask = np.ones(image_bgr.shape[:2], dtype=np.uint8) * 255
            
            # Identify black pixels (dark pixels)
            black_pixels = np.all(image_bgr < 30, axis=2)
            # Identify white pixels (bright pixels)
            white_pixels = np.all(image_bgr > 225, axis=2)
            
            # Set the mask value to 0 (transparent) for black or white pixels
            mask[black_pixels | white_pixels] = 0
            
            # If image already had alpha, combine with our new mask
            if has_alpha:
                mask = cv.min(mask, alpha_channel)
                
            # Rotation augmentation with more varied angles
            angle = random.randint(0,360)
            M = cv.getRotationMatrix2D((image_bgr.shape[1]//2,image_bgr.shape[0]//2),angle,1)
            image_bgr = cv.warpAffine(image_bgr, M, (image_bgr.shape[1],image_bgr.shape[0]))
            mask = cv.warpAffine(mask, M, (mask.shape[1],mask.shape[0]))
            
            # Flip image
            if random.random() < 0.5:
                image_bgr = cv.flip(image_bgr, 1)  # horizontal flip
                mask = cv.flip(mask, 1)
            
            # Occasional vertical flip
            if random.random() < 0.2:
                image_bgr = cv.flip(image_bgr, 0)  # vertical flip
                mask = cv.flip(mask, 0)
                
            # Enhanced size variation
            scale_factor = random.uniform(0.7, 1.5)
            base_size = random.randint(50, 100)
            new_w, new_h = int(base_size * scale_factor), int(base_size * scale_factor)
            
            # Ensure the image isn't too big or too small
            new_w = max(30, min(new_w, 150))
            new_h = max(30, min(new_h, 150))
            
            image_bgr = cv.resize(image_bgr, (new_w, new_h))
            mask = cv.resize(mask, (new_w, new_h))
            
            # Apply perspective transform occasionally
            if random.random() < 0.3:
                # Define random perspective transform
                h, w = image_bgr.shape[:2]
                src_pts = np.float32([[0, 0], [w-1, 0], [0, h-1], [w-1, h-1]])
                
                # Define random distortion (not too extreme)
                max_offset = min(w, h) * 0.15
                dst_pts = np.float32([
                    [0+random.uniform(0, max_offset), 0+random.uniform(0, max_offset)],
                    [w-1-random.uniform(0, max_offset), 0+random.uniform(0, max_offset)],
                    [0+random.uniform(0, max_offset), h-1-random.uniform(0, max_offset)],
                    [w-1-random.uniform(0, max_offset), h-1-random.uniform(0, max_offset)]
                ])
                
                # Apply perspective transform
                M = cv.getPerspectiveTransform(src_pts, dst_pts)
                image_bgr = cv.warpPerspective(image_bgr, M, (w, h))
                mask = cv.warpPerspective(mask, M, (w, h))
            
            x_offset = random.randint(0, background.shape[1]-image_bgr.shape[1])
            y_offset = random.randint(0, background.shape[0]-image_bgr.shape[0])
            
            # Only blend pixels where mask is non-zero
            for c in range(0, 3):
                background_roi = background[y_offset:y_offset+image_bgr.shape[0], 
                                      x_offset:x_offset+image_bgr.shape[1], c]
                foreground = image_bgr[:,:,c]
                alpha = mask.astype(float) / 255.0
                background_roi[:] = (1.0 - alpha[:,:]) * background_roi + alpha[:,:] * foreground
                
            if class_id not in annotation:
                annotation[class_id] = []
            annotation[class_id].append((x_offset,y_offset,image_bgr.shape[1],image_bgr.shape[0]))
        if annotate:
            self.create_label_file(index,dataset_type,annotation,background)
        if example_image:
            if not os.path.exists(f"examples"):
                os.makedirs(f"examples")
            cv.imwrite(f"examples/{index}.jpg",background)
            return f"examples/{index}.jpg"
        else:
            cv.imwrite(f"dataset/images/{dataset_type}/{index}.jpg",background)
            return f"dataset/images/{dataset_type}/{index}.jpg"
    def create_label_file(self, index, dataset_type,annotations,background=None):
        if not os.path.exists(f"dataset/images/{dataset_type}"):
            os.makedirs(f"dataset/images/{dataset_type}")
        if not os.path.exists(f"dataset/labels/{dataset_type}"):
            os.makedirs(f"dataset/labels/{dataset_type}")
        with open(f"dataset/labels/{dataset_type}/{index}.txt", "w") as f:
            for class_id, boxes in annotations.items():
                for box in boxes:
                    x, y, w, h = box
                    # Normalize coordinates
                    x = (x + w / 2) / background.shape[1]
                    y = (y + h / 2) / background.shape[0]
                    w = w / background.shape[1]
                    h = h / background.shape[0]
                    f.write(f"{class_id} {x} {y} {w} {h}\n")
def main():
    dataset = DataSetCreator("backgrounds","img")
    dataset.create_dataset()
    print("Dataset created")
if __name__ == "__main__":
    main()
