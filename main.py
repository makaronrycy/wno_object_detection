from dataset import DataSetCreator
from object_detector import ObjectDetector
import os
if __name__ == "__main__":
    # Create dataset
    dataset = DataSetCreator("backgrounds", "img")
    if not os.path.exists('dataset'):
        dataset.create_dataset()
        print("Dataset created")
    else:
        print("Dataset already exists, skipping creation.")

    # Initialize object detector
    detector = ObjectDetector()
    if os.path.exists('best_model.pt'):
        detector.load_model('best_model.pt')
        print("Model loaded from best_model.pt")
    else:
        detector.train()
        
        detector.save_model()
        print("Model trained and saved as best_model.pt")
    for i in range(5):
        # Create a sample image composite
        example_image = dataset.create_image_composite(i, "example", annotate=False, example_image=True)
        results = detector.infer(example_image)
        detector.visualize(results)

    # Evaluate the model
    detector.validate()
    #detector.evaluate()