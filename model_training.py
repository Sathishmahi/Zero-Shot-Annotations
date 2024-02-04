from ultralytics import YOLO
import torch
import yaml

def train_yolo_model(epochs,train_path=None,val_path=None,classes_list=None,task="detection",
                     model_varient="yolov8n.pt",yaml_file_name = "config.yaml",classification_data_dir=None):
  
  

  if task=="detection" or task=="track":
    
    model = YOLO(model=model_varient,task=task)

    device = 0 if torch.cuda.is_available() else "cpu"
   

    yaml_dict = {"train":train_path,"val":val_path,
                "names":{i:c for i,c in enumerate(classes_list)}}
    print(f"{yaml_dict=}")
    with open("config.yaml","w") as yf:
      yaml.safe_dump(yaml_dict,yf)

    result = model.train(data=yaml_file_name,epochs=epochs,device=device)

  else:
    yaml_file_name = classification_data_dir
    device = 0 if torch.cuda.is_available() else "cpu"
    model = YOLO(model_varient,task=task)


    result = model.train(data=yaml_file_name,epochs = int(epochs), device=device)

  return result