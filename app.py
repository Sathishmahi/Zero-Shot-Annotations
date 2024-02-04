import streamlit as st
import os
import json
import zipfile
from torch import det
from ultralytics import YOLO
from zero_shot_annotations.infernce_model import inference_model
from zero_shot_annotations.detection import detection_combine_all
from zero_shot_annotations.classification import classification_combine_all

classfication_task = "classification"
detection_task = "detection"
show_img_path = "predicted.jpg"
out_video_path="out_video.mp4"
n_classes = 2

run_json = "status.json"
zip_path = "data.zip"
status = 0
if not os.path.isfile(run_json):
  status = 0
  with open(run_json,"w") as f:
    json.dump({"status":status,"model_path":0,"task":0},f)





def zip_folder(folder_path, zip_path):
    with zipfile.ZipFile(zip_path, 'w') as zip_file:
        for foldername, subfolders, filenames in os.walk(folder_path):
            for filename in filenames:
                file_path = os.path.join(foldername, filename)
                arcname = os.path.relpath(file_path, folder_path)
                zip_file.write(file_path, arcname=arcname)

# Example usage
folder_to_zip = 'zero_shot_annotations'
output_zip_file = 'zero_shot_annotations.zip'

zip_folder(folder_to_zip, output_zip_file)
    

if not status:
  task = st.radio("Select the Task",[classfication_task,detection_task,"track"])

  video_file_path = st.text_input("enter the input video path : ")
  classes = st.text_input("enter the class names to detect [comma sep values] : ")
  model_varient = st.selectbox("pick the model varient",["nano","small","medium","large","x-large"])
  epochs = st.select_slider("select the no of epochs to train : ",list(range(1,20)))
  batch_size = st.select_slider("select the batch size to train : ",list([2**i for i in range(2,6)]))

  is_ready_to_infer = False

  if task == classfication_task:

    
    train_btn = st.button("train")
    if train_btn and video_file_path:
    
      result = classification_combine_all(epochs,video_file_path,classes,
      task,model_varient,imgs_thersold=20)

      is_ready_to_infer = True

      model_path = os.path.join(f"{str(result.save_dir)}","weights","best.pt")
      with open(run_json,"w") as f:
        json.dump({"status":1,"model_path":model_path,"task":classfication_task,"metrics":result.results_dict},f)

  else:
    
    box_ther = st.text_input("enter the box ther")
    text_ther = st.text_input("enter the text ther")
    train_btn = st.button("train")

    if train_btn and video_file_path:
      print(epochs,video_file_path,classes,float(box_ther),
      float(text_ther),task,model_varient)
      result = detection_combine_all(epochs,video_file_path,classes,float(box_ther),
      float(text_ther),task,model_varient)

      model_path = os.path.join(f"{str(result.save_dir)}","weights","best.pt")
      with open(run_json,"w") as f:
        json.dump({"status":1,"model_path":model_path,"task":task,"metrics":result.results_dict},f)

      is_ready_to_infer = True


if os.path.isfile(run_json):
    is_infer_model = st.checkbox("inference model")
    with open(run_json,"r") as f:
      con = json.load(f)
    if con["status"]:
      model,task = YOLO(con["model_path"]),con["task"]
      
      

      if is_infer_model and is_infer_model:

        source_mode = st.radio("pick the source",["image","video"])
        if source_mode=="image":

          img_path = st.text_input("enter the inference img path : ")

          if img_path:
            inference_model(model,img_path,source_mode,task,n_classes = n_classes,
            show_img_path = show_img_path)

            st.write("inference result : ")
            st.image(show_img_path)
        else:
          video_path = st.text_input("enter the inference video path : ")
          if video_path:
            print(f"{n_classes=}")
            inference_model(model,video_path,source_mode,task,n_classes = n_classes,
            out_video_path=out_video_path)
            st.write("inference result : ")
            st.video(out_video_path)

      with open(con["model_path"],"rb") as f:
        model_data = f.read()

      zip_folder("datasets",zip_path)
      
      st.download_button("download model",model_data,file_name="model.pt")


      st.download_button("download data",zip_path,file_name="data.zip")



