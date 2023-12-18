import torch
import numpy as np
import nibabel as nib
from functools import lru_cache
from monai.transforms import (Compose, ScaleIntensityRanged, NormalizeIntensityd, ClassesToIndicesd, 
                              RandCropByLabelClassesd, BorderPadd, RandRotated, RandAffined,
                              CenterSpatialCropd, RandGaussianNoised, RandGaussianSmoothd, 
                              RandAdjustContrastd, RandScaleIntensityd, Rand3DElasticd,
                              RandAxisFlipd, RandZoomd, Resized, CastToTyped, SqueezeDimd,
                              ToDeviced, LoadImaged, EnsureChannelFirstd, EnsureTyped, ScaleIntensityRangePercentilesd)
from monai.data import CacheDataset, Dataset
from monai.data.utils import partition_dataset

#from monai.data.meta_obj import set_track_meta
#set_track_meta(False)

class NiftiDataset(Dataset):
    def __init__(self, image_files, label_files):
        
        self.data = []
        self.data_string = []
        
        
        fg_pixels_list = []
        
        for image_fn, label_fn in zip(image_files, label_files):
            self.data_string.append({"image":image_fn, "label":label_fn})

            # expects the data to not have a channel dimension
            image = nib.load(image_fn).get_fdata()
            label = nib.load(label_fn).get_fdata()
            
            d = {
                "image" : torch.from_numpy(image).float().unsqueeze(0), 
                "label" : torch.from_numpy(label).long().unsqueeze(0)
            }
            self.data.append(d)
            fg_pixels_list.append(image[label != 0])
            
        foreground_pixels = np.concatenate(fg_pixels_list, axis=None)
        self.mean = np.mean(foreground_pixels)
        self.std = np.std(foreground_pixels)
        self.min = np.percentile(foreground_pixels, 0.5)
        self.max = np.percentile(foreground_pixels, 99.5)
        self.range = self.max - self.min

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]
    
def makeTransforms(train_stats, patch_size, batch_size, num_classes, class_sample_ratios, device):
    """
    transform expects path strings.
    loaded image and label should both have shapes of (num_channels, H, W, D)
    """
    initial_crop_size = [2*s for s in patch_size]
    
    transforms = Compose([
        # load images from path strings
        LoadImaged(keys=["image", "label"], image_only=True),
        
        # Remove metadata
        EnsureTyped(keys=["image", "label"], track_meta=False),
        
        # add a channel dimension
        EnsureChannelFirstd(keys=["image", "label"], channel_dim="no_channel"),
        
        # clip by 0.5 and 99.5 percentiles
        ScaleIntensityRanged(keys="image", a_min=train_stats["min"], a_max=train_stats["max"], 
                             b_min=0, b_max=1, clip=True),

        # pad borders of the image
        BorderPadd(keys=["image", "label"], spatial_border=64, mode="constant", value=0),

        # get class label indices for oversampling
        ClassesToIndicesd(keys="label", indices_postfix="_cls_indices", num_classes=num_classes, image_key="image", 
                          image_threshold=0),

        # initial larger crop to 256
        RandCropByLabelClassesd(keys=["image", "label"], label_key="label", spatial_size=initial_crop_size, 
                                ratios=class_sample_ratios, num_classes=num_classes, num_samples=batch_size, 
                                indices_key="label_cls_indices"),
        
        # send to device after random transform. Dataset too large to be cached on GPU
        EnsureTyped(keys=["image", "label"], device=device, track_meta=False),

        # rotation
        RandRotated(keys=["image", "label"], range_x=3.14, range_y=3.14, range_z=3.14, prob=0.5, 
                    mode=["bilinear", "nearest"], padding_mode="zeros", dtype=None),

        # scaling and shear
        RandAffined(keys=["image", "label"], prob=0.5, shear_range=[0.1,0.1,0.1], scale_range=[0.3,0.3,0.3],
                    mode=["bilinear", "nearest"], padding_mode="zeros"),

        # elastic
        Rand3DElasticd(keys=["image", "label"], prob=0.2, sigma_range=[3,7], magnitude_range=[50,150],
                    mode=["bilinear", "nearest"], padding_mode="zeros"),
        
        # center crop to final resultion of 128
        CenterSpatialCropd(keys=["image", "label"], roi_size=patch_size),

        # normalize by mean and std
        NormalizeIntensityd(keys="image", subtrahend=(train_stats["mean"]-train_stats["min"])/train_stats["range"], 
                            divisor=train_stats["std"]/train_stats["range"]),

        # gaussian noise
        RandGaussianNoised(keys="image", prob=0.1, mean=0.0, std=0.1),

        # gaussian blurring
        RandGaussianSmoothd(keys="image", prob=0.1, sigma_x=(0.5,1), sigma_y=(0.5,1), sigma_z=(0.5,1)),

        # brightness 
        RandScaleIntensityd(keys="image", prob=0.15, factors=0.25),
        
        # contrast
        #IntensityStatsd
        #RandScaleIntensityd
        #ScaleIntensityRange
        # ask chatgpt

        # simulation of low resolution
        RandZoomd(keys="image", prob=0.15, min_zoom=0.5, max_zoom=1.0, mode="nearest", keep_size=False),
        Resized(keys="image", spatial_size=patch_size, mode="trilinear"),

        # gamma 
        RandAdjustContrastd(keys="image", prob=0.15, gamma=(0.7, 1.5)),

        # flip
        RandAxisFlipd(keys=["image", "label"], prob=0.5),
        
        # FOR SWIN UNETR
        ScaleIntensityRanged(keys="image", a_min=-2, a_max=2, b_min=0, b_max=1, clip=True),
        
        # cast to final type
        CastToTyped(keys=["image", "label"], dtype=[torch.float, torch.long]),
        
        # squeeze channel dimension of label
        SqueezeDimd(keys="label", dim=0)

    ])
    
    return transforms

def makeDataset(image_files, label_files, device, patch_size=[128,128,128], batch_size=2, num_classes=5, 
                class_sample_ratios=[1,5,2,1,1], rank=None, world_size=None):
    
    train_stats = get_data_statistics(image_files, label_files)
    dataset_string = train_stats["data_string"]
    
    transforms = makeTransforms(train_stats, patch_size=patch_size, batch_size=batch_size, 
                                num_classes=num_classes, class_sample_ratios=class_sample_ratios, 
                                device=device)
    if rank is not None:
        # must be evenly divisible for DDP!
        data = partition_dataset(data=dataset_string, num_partitions=world_size,
                                 shuffle=True, seed=420, even_divisible=True)[rank]
    else:
        data = dataset_string
    
    dataset = CacheDataset(data, transforms, num_workers=6, copy_cache=False, runtime_cache=True)
    #dataset = Dataset(data, transforms)
    return dataset


def get_data_statistics(image_files, label_files):
    """
    transform to hashable inputs for caching purposes
    """
    return get_data_statistics_hashable_wrapper(tuple(image_files), tuple(label_files))

@lru_cache(maxsize=2)
def get_data_statistics_hashable_wrapper(image_files_tuple, label_files_tuple):
    niftiDataset = NiftiDataset(image_files_tuple, label_files_tuple)
    d = {"mean": niftiDataset.mean,
         "std" : niftiDataset.std,
         "min" : niftiDataset.min,
         "max" : niftiDataset.max,
         "range" : niftiDataset.range,
         "data_string" : niftiDataset.data_string}
    return d
    
    
