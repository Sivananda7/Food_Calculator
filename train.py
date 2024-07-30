from ultralytics import YOLO

def main():

    model = YOLO("yolov8n.yaml")

    data_yaml_file = "C:/Users/kaavi/OneDrive/Desktop/food/data.yaml"

    project = "C:/Users/kaavi/OneDrive/Desktop/food"
    experiment = "food-model"

    batch_size = 4

    results = model.train(data=data_yaml_file,
                          epochs = 50,
                          project=project,
                          name = experiment,
                          batch = batch_size,
                          device = 0,
                          patience = 5,
                          imgsz=640,
                          verbose = True,
                          val=True)
    
if __name__ == "__main__":
    main()