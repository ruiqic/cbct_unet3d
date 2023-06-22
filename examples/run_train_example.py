import os
import time
import logging
from datetime import datetime
import torch
import torch.nn as nn
from monai.data import DataLoader, ThreadDataLoader
from monai.optimizers import Novograd

from cbct_unet3d.dataset import makeDataset
from cbct_unet3d.model import UNet3D
from cbct_unet3d.losses import GeneralizedDiceCELoss

import warnings
warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')

def setup_logger(name, log_file, level=logging.INFO):
    """To setup as many loggers as you want"""

    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    handler = logging.FileHandler(log_file)        
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)

    return logger

if __name__ == "__main__" :
    
    # Replace with your save directory
    save_name = "../saved/example_%s"%(datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
    os.makedirs(save_name, exist_ok=True)
    logger_train = setup_logger('train_logger', '%s/train.log'%save_name)

    # Replace with your directory where the images and labels are
    image_root = "/storage/home/hhive1/rchen438/data/nnunet/nnUNet_raw/Dataset102_MarDentalAll/imagesTr"
    label_root = "/storage/home/hhive1/rchen438/data/nnunet/nnUNet_raw/Dataset102_MarDentalAll/labelsTr"
    image_fns = sorted(filter(lambda x: x.endswith(".nii.gz"), os.listdir(image_root)))
    label_fns = sorted(filter(lambda x: x.endswith(".nii.gz"), os.listdir(label_root)))
    image_files = [os.path.join(image_root, x) for x in image_fns]
    label_files = [os.path.join(label_root, x) for x in label_fns]
    
    device_type = "cuda" if torch.cuda.is_available() else "cpu"
    #device_type = "cpu" # for debugging
    device = torch.device(device_type)
    
    repeats = 2
    dataset = makeDataset(image_files, label_files, device=device, class_sample_ratios=[1,5,2,1,1],
                          patch_size=[128,128,128], batch_size=4, num_classes=5)
    dataloader = ThreadDataLoader(dataset, num_workers=0, repeats=repeats, buffer_size=4, shuffle=True)

    logger_train.info("%d training volumes"%(len(dataset)))

    # SETUP LR AND OTHER PARAMETERS
    lr = 0.001
    weight_decay = 0.01
    num_epochs = 600
    save_freq = 50

    network = UNet3D(in_channels=1, num_classes=5, strides=[1,2,2,2,2,2], channels=[32,64,128,256,320,320], prelu=True)
    network = network.to(device)

    criterion = GeneralizedDiceCELoss(weight=torch.tensor([0.1, 0.5, 0.2, 0.1, 0.1]).to(device))
    optimizer = torch.optim.AdamW(network.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, steps_per_epoch=len(dataset)*repeats, epochs=num_epochs)
    scaler = torch.cuda.amp.GradScaler()

    # Train!
    t = time.time()
    for epoch in range(num_epochs):
        epoch_start = time.time()
        train_loss = 0
        for batch_num, data in enumerate(dataloader):
            network.train()
            optimizer.zero_grad()
            x, y = data["image"], data["label"]
            with torch.autocast(device_type=device_type):
                x, y = x.to(device), y.to(device)
                outputs = network(x)
                loss = criterion(outputs, y) 
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            #loss.backward()
            #optimizer.step()
            train_loss += x.shape[0] * loss.item()

            scheduler.step()
            
        if epoch % save_freq == save_freq-1:
            torch.save(network.state_dict(), "%s/checkpoint_%d.pt"%(save_name, epoch+1))
        

        print_text = 'Epoch: {}, Training Loss: {:.3f}, Time: {:0.1f}'.format(epoch, train_loss / len(dataset)/repeats, 
                                                                            time.time()-epoch_start)
        print(print_text)
        logger_train.info(print_text)

    print("time taken:", (time.time()-t)/3600)
    logger_train.info("time taken: %0.2f"%((time.time()-t)/3600))
