import json
import glob
import numpy as np
import cv2
import copy
import os
import random
import time
import sys
import seaborn as sn
import matplotlib.pyplot as plt
import face_recognition
from tqdm.autonotebook import tqdm
import torch
import torchvision
from torch import nn
from torchvision import models
from torch.autograd import Variable
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
import pickle
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

im_size = 112
mean=[0.485, 0.456, 0.406]
std=[0.229, 0.224, 0.225]
sm = nn.Softmax()
inv_normalize =  transforms.Normalize(mean=-1*np.divide(mean,std),std=np.divide([1,1,1],std))
def im_convert(tensor):
    image = tensor.to("cpu").clone().detach()
    image = image.squeeze()
    image = inv_normalize(image)
    image = image.numpy()
    image = image.transpose(1,2,0)
    image = image.clip(0, 1)
    cv2.imwrite('./2.png',image*255)
    return image

def predict(model,img,path = './'):
  fmap,logits = model(img.to('cuda'))
  params = list(model.parameters())
  weight_softmax = model.linear1.weight.detach().cpu().numpy()
  logits = sm(logits)
  _,prediction = torch.max(logits,1)
  confidence = logits[:,int(prediction.item())].item()*100
  print('confidence of prediction:',logits[:,int(prediction.item())].item()*100)
  idx = np.argmax(logits.detach().cpu().numpy())
  bz, nc, h, w = fmap.shape
  out = np.dot(fmap[-1].detach().cpu().numpy().reshape((nc, h*w)).T,weight_softmax[idx,:].T)
  predict = out.reshape(h,w)
  predict = predict - np.min(predict)
  predict_img = predict / np.max(predict)
  predict_img = np.uint8(255*predict_img)
  out = cv2.resize(predict_img, (im_size,im_size))
  heatmap = cv2.applyColorMap(out, cv2.COLORMAP_JET)
  img = im_convert(img[:,-1,:,:,:])
  result = heatmap * 0.5 + img*0.8*255
  cv2.imwrite('./1.png',result)
  result1 = heatmap * 0.5/255 + img*0.8
  r,g,b = cv2.split(result1)
  result1 = cv2.merge((r,g,b))
  plt.imshow(result1)
  plt.show()
  return [int(prediction.item()),confidence]

class ValidationDataset(Dataset):
    def __init__(self, video_names, save_dir, sequence_length=100, transform=None):
        self.video_names = video_names
        self.save_dir = save_dir  # Base directory where frames will be saved
        self.transform = transform
        self.count = sequence_length

        # Create base save directory if it doesn't exist
        os.makedirs(self.save_dir, exist_ok=True)

    def __len__(self):
        return len(self.video_names)

    def __getitem__(self, idx):
        video_path = self.video_names[idx]

        # Save all frames directly in `save_dir` (not in a subfolder)
        frame_paths = self.frame_extract(video_path, self.save_dir)
        
        # Check if any frame contains a face
        face_detected = False
        frames = []
        for frame_path in frame_paths[:self.count]:  # Limit to sequence_length frames
            frame = cv2.imread(frame_path)  # Read saved frame
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert to RGB

            # Detect faces in the frame
            face_locations = face_recognition.face_locations(frame_rgb)
            if face_locations:  # If at least one face is detected
                top, right, bottom, left = face_locations[0]  # Use first detected face
                face_image = frame_rgb[top:bottom, left:right]
                face_detected = True
            else:
                # If no face, use the full frame or skip (here, we'll skip)
                continue

            if self.transform:
                frame_rgb = self.transform(frame_rgb)

            frames.append(frame_rgb)

        # If no face is detected in any frame, raise an error
        if not face_detected:
            raise RuntimeError("No face found in the video frames.")

        # Pad frames if necessary
        if len(frames) < self.count:
            while len(frames) < self.count:
                frames.append(frames[-1])  # Duplicate last frame to maintain consistency

        frames = torch.stack(frames)  # Shape: (sequence_length, C, H, W)
        return frames.unsqueeze(0)  # Shape: (1, sequence_length, C, H, W)

    def frame_extract(self, path, save_dir):
        """Extract frames from a video and save them directly in save_dir."""
        vidObj = cv2.VideoCapture(path)
        success, image = vidObj.read()
        frame_paths = []
        frame_count = 0

        while success:
            frame_filename = os.path.join(save_dir, f"frame_{frame_count:04d}.jpg")
            cv2.imwrite(frame_filename, image)  # Save frame
            frame_paths.append(frame_filename)
            frame_count += 1
            success, image = vidObj.read()

        return frame_paths
