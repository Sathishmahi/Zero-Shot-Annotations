
from torchvision.ops import box_convert
import numpy as np
from IPython.display import Image, clear_output
from groundingdino.util.inference import load_model, load_image, predict
import torch
import cv2
import os
import wget
from tqdm import tqdm
from PIL import Image
import subprocess
import shutil
from random import randint 
from zero_shot_annotations.model_training import train_yolo_model
from zero_shot_annotations.classification import get_class


color_dict = {}
model_varient_dict = {"nano":"n","small":"s","large":"l","x-large":"xl"}
temp_img_name = "temp.jpg"
weight_url = "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth"

CONFIG_PATH = "groundingdino/config/GroundingDINO_SwinT_OGC.py"




def convert_to_yolo_format(bbox, img_w, img_h,pharse,original_pharses):
    x1, y1, x2,y2 = bbox
    w,h = (x2 - x1) , (y2-y1)
    # Convert to normalized coordinates
    center_x = (x1 + x2) / (2 * img_w)
    center_y = (y1 + y2) / (2 * img_h)
    normalized_width = w / img_w
    normalized_height = h / img_h
    return [str(i)for i in [original_pharses.index(pharse),center_x, center_y, normalized_width, normalized_height]]



### write coor into txt file

def write_a_bboxes_into_txt(boxes, logits, phrases,original_pharses,txt_file_path,w,h):

  try:
    yolo_bboxes_content_list = []
    for bb,lo,ph in zip(boxes, logits, phrases ):
      con=" ".join(convert_to_yolo_format(bb,w,h,ph,original_pharses))
      con+="\n"
      yolo_bboxes_content_list.append(con)

    with open(txt_file_path,"w") as f:
      f.writelines(yolo_bboxes_content_list)
    return True

  except Exception as e:
    raise e
    return False


def color_provider(id):
    """
    Generate a random color for object tracking based on the provided ID.
    """
    color = [randint(0, 256) for _ in range(3)]
    if id in color_dict:
        return color_dict[id]
    else:
        if color in color_dict.values():
            while True:
                color = [randint(0, 256) for _ in range(3)]
                if color not in color_dict.values():
                    break
        color_dict[id] = color
    return color_dict[id]
    

def convert_video(input_video,output_video):
    if not os.path.exists(output_video):
        command = f"ffmpeg -i {input_video} -vcodec libx264 {output_video}"
        try:
            subprocess.run(command, shell=True, check=True)
            print("Video conversion completed successfully.")
        except subprocess.CalledProcessError as e:
            print(f"Video conversion failed with error: {e}")
    else:print(f"OUTPUT FOUND : {output_video}")

def annotate(image_source: np.ndarray, boxes: torch.Tensor) -> np.ndarray:
    h, w, _ = image_source.shape
    boxes = boxes * torch.Tensor([w, h, w, h])
    xyxy = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy().astype(np.int32)
    return xyxy

def download_wight(weight_dir_name):
  os.makedirs(weight_dir_name,exist_ok=True)
  WEIGHTS_NAME = "groundingdino_swint_ogc.pth"
  weight_path = os.path.join("weights", WEIGHTS_NAME)
  if not os.path.exists(weight_path):
    wget.download(weight_url,out = weight_path)
  else:print("WEIGHT FILE ALREADY DOWNLOADED")  
  return weight_path


model = load_model(CONFIG_PATH, download_wight("weights"))


def prepare_detection_data(video_path,TEXT_PROMPT,BOX_TRESHOLD,TEXT_TRESHOLD,
                task="detection",root_dir="datasets",data_dir="data"):
  
  TRAIN_DATA_DIR_NAME, VAL_DATA_DIR_NAME =  "train", "val"
  IMG_DIR_NAME, TXT_DIR_NAME = "images","labels"
  (train_images_path,train_txt_path),(val_img_path,val_txt_path) = [ (os.path.join(root_dir,data_dir,p,IMG_DIR_NAME),
                                                                  os.path.join(root_dir,data_dir,p,TXT_DIR_NAME)) for p in [TRAIN_DATA_DIR_NAME,VAL_DATA_DIR_NAME]]
  [os.makedirs(i,exist_ok=True)for i in (train_images_path,train_txt_path,val_img_path,val_txt_path)]
  cap =  cv2.VideoCapture(video_path)
  video_frame_width,video_frame_height = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                                        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
  original_pharses = list(map(lambda c:c.replace(" ",""),TEXT_PROMPT.split(",")))

  total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
  train_data_ther = 0.9
  val_data_ther = 0.1
  train_data_size = int(total_frames*train_data_ther)
  val_data_size = int(total_frames*val_data_ther)

  if os.path.exists(root_dir):
   for c in tqdm(range(total_frames),desc = "preparing the detection data process"):
    
    suc,frame = cap.read()
    cv2.imwrite(temp_img_name,frame)
    image_source, image = load_image(temp_img_name)
    # img = cv2.cvtColor(cv2.imread("/content/Om2.jpg"), cv2.COLOR_BGR2RGB)/255.
    class_name,_=get_class(Image.open(temp_img_name),original_pharses)
    boxes, logits, phrases = predict(
        model=model,
        # image=torch.tensor(img,dtype=torch.float32),
        image=image,
        caption=class_name,
        box_threshold=BOX_TRESHOLD,
        text_threshold=TEXT_TRESHOLD
    )

    boxes = annotate(image_source,boxes)

    if boxes.size:
      img_path,txt_path = (os.path.join(train_images_path,f"{c}.jpg"),
                          os.path.join(train_txt_path,f"{c}.txt")) if c<=train_data_size else (os.path.join(val_img_path,f"{c}.jpg"),
                          os.path.join(val_txt_path,f"{c}.txt"))
      write_a_bboxes_into_txt(boxes, logits, phrases,original_pharses, txt_path, video_frame_width, video_frame_height )
      cv2.imwrite(img_path,frame)

  return [ (os.path.join(data_dir,p,IMG_DIR_NAME)) for p in [TRAIN_DATA_DIR_NAME,VAL_DATA_DIR_NAME]]

  
def detection_combine_all(epochs,video_path,TEXT_PROMPT,BOX_TRESHOLD,TEXT_TRESHOLD,
                task,model_varient,root_dir="datasets",data_dir="data"):
  train_images_path,val_img_path = prepare_detection_data(video_path,TEXT_PROMPT,BOX_TRESHOLD,TEXT_TRESHOLD,
                task=task,root_dir=root_dir,data_dir=data_dir)

  print(f" ================    PREPATE DATA COMPLETED    =================")
  class_list = [c.replace(" ","")for c in TEXT_PROMPT.split(",")]
  varient = model_varient_dict[model_varient]
  model_name = f"yolov8{varient}.pt"
  result = train_yolo_model(epochs=epochs,train_path=train_images_path,
  val_path = val_img_path,task=task,model_varient=model_name,classes_list=class_list)

  return result
  # cap.release()
