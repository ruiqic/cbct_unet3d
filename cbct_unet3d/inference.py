import os
import torch
from monai.transforms import LoadImage, Compose, ScaleIntensityRange, NormalizeIntensity, ScaleIntensityRangePercentiles, ScaleIntensityRange
from monai.inferers import SlidingWindowInferer
from monai.data import NibabelWriter
from cbct_unet3d.model import UNet3D
from cbct_unet3d.dataset import get_data_statistics

def sliding_predict(train_image_files, train_label_files, test_image_files, 
                    model, checkpoint_path, zero_mean=True, patch_size=[128,128,128],
                    overlap=0.5, mode="gaussian", sigma_scale=0.125):
    """
    file list of training images and labels needed to extract image statistics
    such as mean, std of foreground pixel intensities.
    """
    
    device_type = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_type)
    
    train_stats = get_data_statistics(train_image_files, train_label_files)
    
    transforms = [LoadImage(image_only=True, ensure_channel_first=True),
                         ScaleIntensityRange(a_min=train_stats["min"], a_max=train_stats["max"], 
                             b_min=0, b_max=1, clip=True),
                         NormalizeIntensity(subtrahend=(train_stats["mean"]-train_stats["min"])/train_stats["range"], 
                            divisor=train_stats["std"]/train_stats["range"])]
    if not zero_mean:
        transforms.append(ScaleIntensityRange(a_min=-2, a_max=2, b_min=0, b_max=1, clip=True))
    
    transform = Compose(transforms)
    
    network = model.to(device)
    network.load_state_dict(torch.load(checkpoint_path))
    network.eval()
    inferer = SlidingWindowInferer(roi_size=patch_size, sw_batch_size=4, overlap=overlap, 
                                   mode=mode, sigma_scale=sigma_scale, sw_device=device, 
                                   device="cpu", cache_roi_weight_map=True, progress=False)
    
    pred_logits = []
    
    for test_fn in test_image_files:
        data = transform(test_fn).unsqueeze(0).to(device)
        with torch.no_grad():
            pred = inferer(inputs=data, network=network)
            pred_logits.append(pred.squeeze(0))
    
    return pred_logits


def write_files(dst, filenames, pred_logits):
    """
    filenames expect lists of full path.
    Only the last part is used as the filename
    
    """
    os.makedirs(dst, exist_ok=True)
    writer = NibabelWriter()
    for fn, pred in zip(filenames, pred_logits):
        short_fn = fn.split("/")[-1]
        out_fn = os.path.join(dst, short_fn)
        
        out_pred = pred.argmax(dim=0)
        
        # the split background class recombined
        out_pred[out_pred == 5] = 0
        
        writer.set_data_array(out_pred, channel_dim=None)
        writer.write(out_fn, verbose=False)
        