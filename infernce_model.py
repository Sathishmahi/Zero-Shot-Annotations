import cvzone
import cv2
import torch
from itertools import zip_longest
import numpy as np
from tqdm import tqdm

def single_frame_writer(source,result,task,n_classes=None):

  res = result[0]
  if task == "classification":
    np_prob = res.probs.data.cpu().numpy()
    print(f"{np_prob=}")
    print(f"{n_classes=}")
    argmax_probs = np.argsort(np_prob)[-n_classes:][::-1]
    txt_h,txt_w = 25,100


    for argmax in argmax_probs:
      class_prob = np_prob[argmax]
      class_name = res.names[argmax]
      s = f"# {class_name}  {(class_prob):.1f}"
      print(s)
      # cvzone.putTextRect(frame,s,txt_coors)
      # cv2.rectangle(frame,(x1,y1),(x2,y2),(255,0,0),2)
      cv2.putText(source,s,[txt_w,txt_h],cv2.FONT_HERSHEY_COMPLEX,1,(255,0,0),2)
      txt_h+=25
    return source

  else:

    ids = res.boxes.id.cpu().numpy().astype(int) if task=="track" else []
    cls_names = res.names
    coords = res.boxes.xyxy.cpu().numpy().astype(int)
    clses = res.boxes.cls.cpu().numpy().astype(int)
    cnfs = res.boxes.conf.cpu().numpy()

    for coord,cls,cnf,obj_id in zip_longest(coords,clses,cnfs,ids):
      x1,y1,x2,y2 = coord
      s = f"# {cls_names[cls]}  {(cnf):1f}" if not obj_id else f"# {obj_id}  {cls_names[cls]}  {(cnf):.1f}"
      txt_coor = x1,y1
      cv2.rectangle(source,(x1,y1),(x2,y2),(0,0,255),2)
      cv2.putText(source,s,txt_coor,cv2.FONT_HERSHEY_COMPLEX,1,(255,0,0),2)
      # cvzone.putTextRect(frame,s,txt_coor)

    return source
      
    
def inference_model(model,source,source_type,task,n_classes=3,show_img_path = "show.jpg",
                    out_video_path = None):
  
  if torch.cuda.is_available():model.to("cuda")
  if task=="classification":

    if source_type=="image":
      
      frame = cv2.imread(source)
      print(f"=========== {frame=}")
      result = model.predict(frame,verbose = False)
      frame = single_frame_writer(frame,result,task,n_classes)
      cv2.imwrite(show_img_path,frame)

    else:

      cap = cv2.VideoCapture(source)
      h,w,fps,fc = [
            int(cap.get(i))
            for i in [cv2.CAP_PROP_FRAME_HEIGHT,cv2.CAP_PROP_FRAME_WIDTH,cv2.CAP_PROP_FPS,cv2.CAP_PROP_FRAME_COUNT]
        ]
      writer = cv2.VideoWriter(out_video_path,cv2.VideoWriter_fourcc(*"VP09")
                               ,fps,(w,h))
      if cap.isOpened():
        for c in tqdm(range(fc),desc="inference training progress"):
          frame = cap.read()[-1]
          result = model.predict(frame,verbose = False)
          frame = single_frame_writer(frame,result,task,n_classes)
          writer.write(frame)
        writer.release()
        cap.release()

  else:

    if source_type=="image":
      
      frame = cv2.imread(source)
      print()
      result = model.predict(frame,verbose = False)
      frame = single_frame_writer(frame,result,task)
      cv2.imwrite(show_img_path,frame)

    else:
      cap = cv2.VideoCapture(source)
      
      h,w,fps,fc = [
            int(cap.get(i))
            for i in [cv2.CAP_PROP_FRAME_HEIGHT,cv2.CAP_PROP_FRAME_WIDTH,cv2.CAP_PROP_FPS,cv2.CAP_PROP_FRAME_COUNT]
        ]
      writer = cv2.VideoWriter(out_video_path,cv2.VideoWriter_fourcc(*"VP09")
                                  ,fps,(w,h))
      if cap.isOpened():
        for c in tqdm(range(fc),desc="inference training progress"):
          frame = cap.read()[-1]
          result = model.track(frame,persist = True, verbose = False)
          frame = single_frame_writer(frame,result,task)
          
          writer.write(frame)

        writer.release()
        cap.release()