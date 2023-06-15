import os
import torch
from monai.transforms import LoadImage, Compose, ScaleIntensityRange, NormalizeIntensity
from monai.inferers import SlidingWindowInferer
from monai.data import NibabelWriter
from cbct_unet3d.model import UNet3D
from cbct_unet3d.dataset import get_data_statistics

def sliding_predict(train_image_files, train_label_files, test_image_files, 
                    checkpoint_path, unet_channels=[16,32,64,128,256,512]):
    """
    file list of training images and labels needed to extract image statistics
    such as mean, std of foreground pixel intensities.
    """
    
    
    
    device_type = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_type)
    
    train_stats = get_data_statistics(train_image_files, train_label_files)
    
    transform = Compose([LoadImage(image_only=True, ensure_channel_first=True),
                         ScaleIntensityRange(a_min=train_stats["min"], a_max=train_stats["max"], 
                             b_min=0, b_max=1, clip=True),
                         NormalizeIntensity(subtrahend=(train_stats["mean"]-train_stats["min"])/train_stats["range"], 
                            divisor=train_stats["std"]/train_stats["range"])])
    
    network = UNet3D(in_channels=1, num_classes=5, strides=[1,2,2,2,2,2], channels=unet_channels).to(device)
    network.load_state_dict(torch.load(checkpoint_path))
    network.eval()
    inferer = SlidingWindowInferer(roi_size=(128,128,128), sw_batch_size=4, overlap=0.5, 
                                   mode="gaussian", sigma_scale=0.125, sw_device=device, 
                                   device="cpu", cache_roi_weight_map=True, progress=False)
    
    pred_logits = []
    
    for test_fn in test_image_files:
        data = transform(test_fn).unsqueeze(0).to(device)
        with torch.no_grad():
            pred = inferer(inputs=data, network=lambda x: network(x, deep_supervision=False))
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
        writer.set_data_array(out_pred, channel_dim=None)
        writer.write(out_fn, verbose=False)
        