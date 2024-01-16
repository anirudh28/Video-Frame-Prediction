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
        
        root = os.path.join(root, 'unlabeled')
        unlabeled = [video for video in os.listdir(root) if os.path.splitext(video)[1] != '.DS_Store']
        self.videos = [os.path.join(root, video) + '/' for video in unlabeled]

        self.length = len(self.videos)
        self.n_frames_input = n_frames_input
        self.n_frames_output = n_frames_output

    def __getitem__(self, index):
        length = self.n_frames_input + self.n_frames_output
        video_folder = os.listdir(self.videos[index])
        
        imgs = [np.array(Image.open(os.path.join(self.videos[index], image))) for image in video_folder]

        past_clips = imgs[:self.n_frames_input]
        future_clips = imgs[-self.n_frames_output:]

        past_clips = [torch.from_numpy(clip) for clip in past_clips]
        future_clips = [torch.from_numpy(clip) for clip in future_clips]

        past_clips = torch.stack(past_clips).permute(0, 3, 1, 2)
        future_clips = torch.stack(future_clips).permute(0, 3, 1, 2)
        
        return past_clips.contiguous().float(), future_clips.contiguous().float()

    def __len__(self):
        return self.length

def load_data(batch_size, val_batch_size, data_root, num_workers):
    data = MovingObjectDataSet(root=data_root, n_frames_input=11, n_frames_output=11)
    
    train_size = int(0.95 * len(data))
    val_size = int(0.05 * len(data))

    train_data, val_data = random_split(data, [train_size, val_size])
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    mean, std = 0, 1
    return train_loader, val_loader, mean, std
