import os
import torch
import numpy as np
import torchmetrics
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm
from segformer_predict import *

class SegmentationDataSet(Dataset):

    def __init__(self, args,transform=None):

        self.stored_images_path=args.res_dir+'/simvp/predictions/last_frames.npy'

        print("last frames stored path::",self.stored_images_path)

        self.last_frames = np.load(self.stored_images_path) #(2000,1,3,160,240)
        
        print("last frames shape:", self.last_frames.shape)

    def __len__(self):
        return len(self.last_frames)

    def __getitem__(self, index):
        return self.last_frames[index]  # we want to return (3,160,240) this dimension


def Segformer_Module(args):
    model2_path = args.model2_path

    val_dataset = SegmentationDataSet(args, None)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    DEVICE = torch.device('cuda:{}'.format(0))

    model = torch.load(model2_path).to(DEVICE)

    loaded_segformer_model = torch.load(model2_path).state_dict()
    model.load_state_dict(loaded_segformer_model)

    model.eval()

    masks_pred_list = []

    with torch.no_grad():
        for x in tqdm(val_dataloader):

            x = x.type(torch.cuda.FloatTensor).to(DEVICE)

            softmax = nn.Softmax(dim=1)

            preds = torch.argmax(softmax(model(x)), axis=1)

            masks_pred_list.append(preds)


    torch_y_pred_masks=torch.cat(masks_pred_list,dim=0)
    numpy_y_pred_masks=torch_y_pred_masks.to('cpu').numpy()

    print("After segmentation shape", numpy_y_pred_masks.shape)

    np.save(args.res_dir+'/pred_masks.npy',numpy_y_pred_masks)
    print("segmentation done successfully")
