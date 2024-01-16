import os
import numpy as np
import torch
import PIL
import einops
import torchvision
from torch import nn
from torch.utils.data import Dataset, DataLoader
from einops import rearrange
from PIL import Image
import typing
from torchvision import transforms
from torchvision.ops import StochasticDepth
from typing import List, Iterable
from tqdm import tqdm
import torchmetrics
from matplotlib import pyplot as plt
import matplotlib.image as mpimg

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

class LayerNorm2d(nn.LayerNorm):
    def forward(self, x):
        x = rearrange(x, "b c h w -> b h w c")
        x = super().forward(x)
        x = rearrange(x, "b h w c -> b c h w")
        return x

class OverlapPatchMerging(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, patch_size: int, overlap_size: int):
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=patch_size,
                stride=overlap_size,
                padding=patch_size // 2,
                bias=False
            ),
            LayerNorm2d(out_channels)
        )

class EfficientMultiHeadAttention(nn.Module):
    def __init__(self, channels: int, reduction_ratio: int = 1, num_heads: int = 8):
        super().__init__()
        self.reducer = nn.Sequential(
            nn.Conv2d(
                channels, channels, kernel_size=reduction_ratio, stride=reduction_ratio
            ),
            LayerNorm2d(channels),
        )
        self.att = nn.MultiheadAttention(
            channels, num_heads=num_heads, batch_first=True
        )

    def forward(self, x):
        _, _, h, w = x.shape
        reduced_x = self.reducer(x)
        # attention needs tensor of shape (batch, sequence_length, channels)
        reduced_x = rearrange(reduced_x, "b c h w -> b (h w) c")
        x = rearrange(x, "b c h w -> b (h w) c")
        out = self.att(x, reduced_x, reduced_x)[0]
        # reshape it back to (batch, channels, height, width)
        out = rearrange(out, "b (h w) c -> b c h w", h=h, w=w)
        return out

class MixMLP(nn.Sequential):
    def __init__(self, channels: int, expansion: int = 4):
        super().__init__(
            # dense layer
            nn.Conv2d(channels, channels, kernel_size=1),
            # depth wise conv
            nn.Conv2d(
                channels,
                channels * expansion,
                kernel_size=3,
                groups=channels,
                padding=1,
            ),
            nn.GELU(),
            # dense layer
            nn.Conv2d(channels * expansion, channels, kernel_size=1),
        )

class ResidualAdd(nn.Module):
    
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        out = self.fn(x, **kwargs)
        x = x + out
        return x

class SegFormerEncoderBlock(nn.Sequential):
    def __init__(
        self,
        channels: int,
        reduction_ratio: int = 1,
        num_heads: int = 8,
        mlp_expansion: int = 4,
        drop_path_prob: float = .0
    ):
        super().__init__(
            ResidualAdd(
                nn.Sequential(
                    LayerNorm2d(channels),
                    EfficientMultiHeadAttention(channels, reduction_ratio, num_heads),
                )
            ),
            ResidualAdd(
                nn.Sequential(
                    LayerNorm2d(channels),
                    MixMLP(channels, expansion=mlp_expansion),
                    StochasticDepth(p=drop_path_prob, mode="batch")
                )
            ),
        )

class SegFormerEncoderStage(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        patch_size: int,
        overlap_size: int,
        drop_probs: List[int],
        depth: int = 2,
        reduction_ratio: int = 1,
        num_heads: int = 8,
        mlp_expansion: int = 4,
    ):
        super().__init__()
        self.overlap_patch_merge = OverlapPatchMerging(
            in_channels, out_channels, patch_size, overlap_size,
        )
        self.blocks = nn.Sequential(
            *[
                SegFormerEncoderBlock(
                    out_channels, reduction_ratio, num_heads, mlp_expansion, drop_probs[i]
                )
                for i in range(depth)
            ]
        )
        self.norm = LayerNorm2d(out_channels)

def chunks(data: Iterable, sizes: List[int]):
    
    curr = 0
    for size in sizes:
        chunk = data[curr: curr + size]
        curr += size
        yield chunk

class SegFormerEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        widths: List[int],
        depths: List[int],
        all_num_heads: List[int],
        patch_sizes: List[int],
        overlap_sizes: List[int],
        reduction_ratios: List[int],
        mlp_expansions: List[int],
        drop_prob: float = .0
    ):
        super().__init__()
        # create drop paths probabilities (one for each stage's block)
        drop_probs =  [x.item() for x in torch.linspace(0, drop_prob, sum(depths))]
        self.stages = nn.ModuleList(
            [
                SegFormerEncoderStage(*args)
                for args in zip(
                    [in_channels, *widths],
                    widths,
                    patch_sizes,
                    overlap_sizes,
                    chunks(drop_probs, sizes=depths),
                    depths,
                    reduction_ratios,
                    all_num_heads,
                    mlp_expansions
                )
            ]
        )

    def forward(self, x):
        features = []
        for stage in self.stages:
            x = stage(x)
            features.append(x)
        return features


class SegFormerDecoderBlock(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, scale_factor: int = 2):
        super().__init__(
            nn.UpsamplingBilinear2d(scale_factor=scale_factor),
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
        )

class SegFormerDecoder(nn.Module):
    def __init__(self, out_channels: int, widths: List[int], scale_factors: List[int]):
        super().__init__()
        self.stages = nn.ModuleList(
            [
                SegFormerDecoderBlock(in_channels, out_channels, scale_factor)
                for in_channels, scale_factor in zip(widths, scale_factors)
            ]
        )

    def forward(self, features):
        new_features = []
        for feature, stage in zip(features,self.stages):
            x = stage(feature)
            if x.shape[-1] == 64:
              x = x[:, :, :, :60]
            new_features.append(x)
        return new_features

class SegFormerSegmentationHead(nn.Module):
    def __init__(self, channels: int, num_classes: int, num_features: int = 4):
        super().__init__()
        self.fuse = nn.Sequential(
            nn.Conv2d(channels * num_features, channels, kernel_size=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(channels)
        )

        self.predict = nn.Conv2d(channels, num_classes, kernel_size=1)


    def forward(self, features):
        x = torch.cat(features, dim=1)
        x = self.fuse(x)
        x = self.predict(x)
        return x

class SegFormer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        widths: List[int],
        depths: List[int],
        all_num_heads: List[int],
        patch_sizes: List[int],
        overlap_sizes: List[int],
        reduction_ratios: List[int],
        mlp_expansions: List[int],
        decoder_channels: int,
        scale_factors: List[int],
        num_classes: int,
        drop_prob: float = 0.0,
    ):

        super().__init__()
        self.encoder = SegFormerEncoder(
            in_channels,
            widths,
            depths,
            all_num_heads,
            patch_sizes,
            overlap_sizes,
            reduction_ratios,
            mlp_expansions,
            drop_prob,
        )
        self.decoder = SegFormerDecoder(decoder_channels, widths[::-1], scale_factors)
        self.head = SegFormerSegmentationHead(decoder_channels, num_classes, num_features=len(widths))
        self.upsample_layer = nn.UpsamplingBilinear2d(scale_factor=(4, 4))

    def forward(self, x):
        features = self.encoder(x)
        features = self.decoder(features[::-1])
        segmentation = self.head(features)
        segmentation = self.upsample_layer(segmentation)
        return segmentation

segformer = SegFormer(
    in_channels=3,
    widths=[64, 128, 256, 512],
    depths=[3, 4, 6, 3],
    all_num_heads=[1, 2, 4, 8],
    patch_sizes=[7, 3, 3, 3],
    overlap_sizes=[4, 2, 2, 2],
    reduction_ratios=[8, 4, 2, 1],
    mlp_expansions=[4, 4, 4, 4],
    decoder_channels=256,
    scale_factors=[8, 4, 2, 1],
    num_classes=49,
)

class SegmentationDataSet(Dataset):

    def __init__(self, video_dir, transform=None):
        self.transforms = transform
        self.images, self.masks = [], []
        for i in video_dir:
            imgs = os.listdir(i)
            self.images.extend([i + '/' + img for img in imgs if not img.startswith('mask')])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img = np.array(Image.open(self.images[index]))
        x = self.images[index].split('/')
        image_name = x[-1]
        mask_index = int(image_name.split("_")[1].split(".")[0])
        x = x[:-1]
        mask_path = '/'.join(x)
        mask = np.load(mask_path + '/mask.npy')
        mask = mask[mask_index, :, :]

        if self.transforms is not None:
            aug = self.transforms(image=img, mask=mask)
            img = aug['image']
            mask = aug['mask']

        return img, mask

train_set_path = '/scratch/sb9509/data/dataset/train/video_' #Change this to your train set path
val_set_path = '/scratch/sb9509/data/dataset/val/video_' #Change this to your validation path

train_data_dir = [train_set_path + str(i) for i in range(0, 1000)]
train_dataset = SegmentationDataSet(train_data_dir, None)
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True)

val_data_dir = [val_set_path + str(i) for i in range(1000, 2000)]
val_dataset = SegmentationDataSet(val_data_dir, None)
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=4, shuffle=True)

model = segformer.to(device)
best_model = None

LEARNING_RATE = 1e-4
num_epochs = 20
max_patience = 3
epochs_no_improve = 0
early_stop = False
SMOOTH = 1e-6

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
scaler = torch.cuda.amp.GradScaler()

def check_accuracy(loader, model):
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    model.eval()
    jaccard = torchmetrics.JaccardIndex(task="multiclass", num_classes=49).to(device)

    y_preds_list = []
    y_trues_list = []

    with torch.no_grad():
        for x, y in tqdm(loader):

            x = x.permute(0, 3, 1, 2).type(torch.cuda.FloatTensor).to(device)
            y = y.to(device)
            softmax = nn.Softmax(dim=1)
            preds = torch.argmax(softmax(model(x)), axis=1)

            y_preds_list.append(preds)
            y_trues_list.append(y)

            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)

            dice_score += (2 * (preds * y).sum()) / ((preds + y).sum() + 1e-8)

    y_preds_concat = torch.cat(y_preds_list, dim=0)
    y_trues_concat = torch.cat(y_trues_list, dim=0)
    print("IoU over val: ", mean_thresholded_iou)

    print(len(y_preds_list))
    print(y_preds_concat.shape)

    jac_idx = jaccard(y_trues_concat, y_preds_concat)

    print(f"Jaccard Index {jac_idx}")

    print(f"Got {num_correct}/{num_pixels} with acc {num_correct / num_pixels * 100:.2f}")
    print(f"Dice score: {dice_score / len(loader)}")

def batch_iou_pytorch(SMOOTH, outputs: torch.Tensor, labels: torch.Tensor):

    intersection = (outputs & labels).float().sum((1, 2))  # Will be zero if Truth=0 or Prediction=0
    union = (outputs | labels).float().sum((1, 2))  # Will be zzero if both are 0

    iou = (intersection + SMOOTH) / (union + SMOOTH)  # We smooth our devision to avoid 0/0

    thresholded = torch.clamp(20 * (iou - 0.5), 0, 10).ceil() / 10  # This is equal to comparing with thresolds

    return thresholded  # Or thresholded.mean() if you are interested in average across the batch

training_loss = []
validation_loss = []

# Train loop
for epoch in range(num_epochs):
    loop = tqdm(train_dataloader)
    train_losses = []
    validation_loss = []
    for idx, (data, targets) in enumerate(loop):
        data = data.permute(0, 3, 1, 2).to(torch.float16).to(device)
        targets = targets.to(device)
        targets = targets.type(torch.long)
        # forward
        with torch.cuda.amp.autocast():
            predictions = model(data)
            loss = loss_fn(predictions, targets)

        train_losses.append(loss.item())
        # backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # update tqdm loop
        loop.set_postfix(loss=loss.item())

    avg_train_loss = sum(train_losses) / len(train_losses)
    training_loss.append(avg_train_loss)

    val_losses = []
    last_val_loss = 1000000
    model.eval()
    mean_thresholded_iou = []
    ious = []
    last_iou = 0
    softmax = nn.Softmax(dim=1)

    with torch.no_grad():
        for i, (x, y) in enumerate(tqdm(val_dataloader)):
            x = x.permute(0, 3, 1, 2).type(torch.cuda.FloatTensor).to(device)
            y = y.to(device)
            y = y.type(torch.long)
            # forward
            with torch.cuda.amp.autocast():
                preds = model(x)
                vloss = loss_fn(preds, y)

            val_losses.append(vloss.item())

            preds_arg = torch.argmax(softmax(preds), axis=1)
            thresholded_iou = batch_iou_pytorch(SMOOTH, preds_arg, y)
            ious.append(thresholded_iou)

        sum_thresholded = 0
        mean_thresholded_iou = sum(ious[i].data for i in range(len(ious))) / len(ious)
        avg_val_loss = sum(val_losses) / len(val_losses)
        validation_loss.append(avg_val_loss)
        print(f"Epoch: {epoch}, avg IoU: {mean_thresholded_iou}, avg val loss: {avg_val_loss}")

    if avg_val_loss < last_val_loss:
        best_model = model
        torch.save(best_model, '/scratch/sb9509/video-frame-prediction/SimVP/outputs/simvp/segformer.pt')
        last_val_loss = avg_val_loss
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1

    if epochs_no_improve > max_patience and epoch > 10:
        early_stop = True
        print("Early Stopping")
