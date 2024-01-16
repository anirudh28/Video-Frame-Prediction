import os
import numpy as np
import torch
import torch.utils.data as data
from torch.utils.data import random_split, Dataset, DataLoader
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

class MovingObjectDataSet(data.Dataset):
    def __init__(self, root, n_frames_input=11, n_frames_output=11):
        super(MovingObjectDataSet, self).__init__()

        root = os.path.join(root, 'val')
        labelled_dirs = [video for video in os.listdir(root) if os.path.splitext(video)[1] != '.DS_Store']
        self.videos = [os.path.join(root, video) + '/' for video in labelled_dirs]
        
        self.length = len(self.videos)
        self.n_frames_input = n_frames_input
        self.n_frames_output = n_frames_output

    def __getitem__(self, index):

        video_folder = os.listdir(self.videos[index])
        imgs = []
        for image in video_folder:
            if image.endswith('.png'):
                imgs.append(np.array(Image.open(self.videos[index] + '/' + image)))

        past_clips = imgs[0:self.n_frames_input]

        past_clips = [torch.from_numpy(clip) for clip in past_clips]
        past_clips = torch.stack(past_clips).permute(0, 3, 1, 2)
        
        return past_clips.contiguous().float()

    def __len__(self):
        return self.length

def load_data(batch_size, val_batch_size,data_root, num_workers):

    whole_data = MovingObjectDataSet(root=data_root, n_frames_input=11, n_frames_output=11)
    val_loader = DataLoader(whole_data, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    mean, std = 0, 1
    return val_loader, mean, std
