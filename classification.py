from logging import root
import cv2
import glob
import random
import shutil
import os
from tqdm import tqdm
import numpy as np
from PIL import Image

from ZSIC import ZeroShotImageClassification
from zero_shot_annotations.model_training import train_yolo_model

zsic = ZeroShotImageClassification()

model_varient_dict = {"nano":"n","small":"s","large":"l","x-large":"xl"}

def get_class(img_obj,candidate_labels):
  preds = zsic(image=img_obj,
            candidate_labels=candidate_labels)
  
  return preds["labels"][np.argmax(preds["scores"])],preds


def classification_prepare_data(video_path, classes_list, train_dir_name = "train",
                              val_dir_name = "val",imgs_thersold = 15,data_dir="data",root_dir = "datasets" ):
  
  temp_overall_img_dir_name = "temp_images"
  img_dir_name = "images"
  [os.makedirs(os.path.join(temp_overall_img_dir_name,i),exist_ok=True) for i in classes_list ]

  temp_img_name = "temp_class.jpg"
  # class_name = os.path.splitext(os.path.split(video_path)[-1])[0]

  # [os.makedirs(os.path.join(d,class_name)) for d in [train_dir_name,val_dir_name]]

  # train_path,val_path =( os.path.join(train_dir_name,class_name),
  #                       os.path.join(val_dir_name,class_name))
  cap = cv2.VideoCapture(video_path)

  if cap.isOpened():
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if os.path.exists(root_dir): #### changed line
     for counter in range(total_frames):
      
      suc,frame = cap.read()
      cv2.imwrite(temp_img_name,frame)
      pil_img_obj = Image.open(temp_img_name)
      class_name,preds = get_class(pil_img_obj,candidate_labels=classes_list)
      img_path = os.path.join(f"{temp_overall_img_dir_name}/{class_name}",f"{counter}.jpg")        
      cv2.imwrite(img_path,frame)

    for class_dir in tqdm(os.listdir(temp_overall_img_dir_name),desc = "preparing the classification data process"):
      class_dir_path = os.path.join(temp_overall_img_dir_name,class_dir)
      all_imgs_path = glob.glob(f"{class_dir_path}/*.jpg")

      if len(all_imgs_path) > imgs_thersold:

        total_imgs = len(all_imgs_path)
          
        TRAIN_DATA_THER = 0.9
        VAL_DATA_THER = 1 - TRAIN_DATA_THER

        TRAIN_DATA_SIZE = int(total_imgs*TRAIN_DATA_THER)
        VAL_DATA_SIZE = total_frames - TRAIN_DATA_SIZE


        for counter,src_img_path in enumerate(all_imgs_path):
          class_name = class_dir
          

          train_path,val_path =( os.path.join(root_dir,data_dir,img_dir_name,train_dir_name,class_name),
                        os.path.join(root_dir,data_dir,img_dir_name,val_dir_name,class_name))

          [os.makedirs(os.path.join(d),exist_ok=True) for d in [train_path,val_path]]
          img_file_name = os.path.split(src_img_path)[-1]
          
          img_path = os.path.join(train_path,img_file_name) if counter<=TRAIN_DATA_SIZE else os.path.join(val_path,img_file_name)
          
          shutil.move(src_img_path,img_path)

    return os.path.join(data_dir,img_dir_name)

  else:
    print("WRONG PATH OR VIDEO FILE NOT OPENED")


def classification_combine_all(epochs,video_path,classes_list_raw,task,model_varient,imgs_thersold,data_dir="data"):

  class_list = [c.replace(" ","")for c in classes_list_raw.split(",")]

  dir_path = classification_prepare_data(video_path, class_list, train_dir_name = "train",
                              val_dir_name = "val",imgs_thersold = imgs_thersold,data_dir=data_dir)
  varient = model_varient_dict[model_varient]
  model_name = f"yolov8{varient}-cls.pt"

  result = train_yolo_model(epochs=epochs,classification_data_dir=dir_path,task=task,model_varient=model_name)

  return result
