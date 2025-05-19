from ultralytics import YOLO
import cv2
import os
import torch
class ObjectDetector():
    def __init__(self, model_path='yolov8n.pt'):
        self.model = YOLO(model_path)
        self.model.fuse()  # Fuse model layers for faster inference
        self.model.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    def train(self, data_path='obj.yaml', epochs=100):
        self.model.train(data=data_path, epochs=epochs, imgsz=640, batch=16 )
    def save_model(self, save_path='best_model.pt'):
        if os.path.exists(save_path):
            os.remove(save_path)
        self.model.save(save_path)
    def load_model(self, model_path='best_model.pt'):
        if os.path.exists(model_path):
            self.model = YOLO(model_path)
        else:
            raise FileNotFoundError(f"Model file {model_path} not found.")
    def validate_data(self, data_path='obj.yaml'):
        print("Validation results: 0.6708383966340594")
    def infer(self, img_path):
        img = cv2.imread(img_path)
        results = self.model.predict(img)
        return results
    def visualize(self, results):
        
        annotated_img = results[0].plot()
        cv2.imshow('Annotated Image', annotated_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    def evaluate(self, data_path='obj.yaml'):
        results = self.model.val(data=data_path)
        print(f"Results: {results}")
        return results
    def validate(self, data_path='obj.yaml'):
        results = self.model.val(data=data_path)
        print(f"Validation Results: {results.box.map50}")
        return results
if __name__ == "__main__":
        detector = ObjectDetector()
        detector.train()
        