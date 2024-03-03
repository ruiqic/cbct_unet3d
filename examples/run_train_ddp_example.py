"""
To use Pytorch's Data Distributed Parallel, you need to run the file with 
the "torchrun" command.

Example for running on 2 GPUs on a single node:
torchrun --nproc-per-node=2 run_train_ddp_example.py

"""

import os
import time
from datetime import datetime
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel
from torch.cuda.amp import autocast, GradScaler
from monai.data import ThreadDataLoader, DataLoader
from monai.optimizers import Novograd
from cbct_unet3d.dataset import makeDataset
from cbct_unet3d.model import UNet3D
from cbct_unet3d.losses import GeneralizedDiceCELoss
from cbct_unet3d.utils import setup_logger, find_free_port
from multiprocessing import Manager
import warnings

warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')

def train():
    torch.set_num_threads(6)
    world_size = int(os.environ["LOCAL_WORLD_SIZE"])
    rank = int(os.environ["LOCAL_RANK"])

    dist.init_process_group(backend="nccl", init_method="env://")
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    
    # Replace with your save directory
    save_name = "../saved/example_ddp_%s" % (datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
    os.makedirs(save_name, exist_ok=True)
    
    logger_train = setup_logger('train_logger', '%s/train.log' % save_name)

    # Replace with your directory where the images and labels are
    dataset_root = os.environ.get("DEFAULT_CBCT_DATASET_PATH")
    image_root = os.path.join(dataset_root, "imagesTr")
    label_root = os.path.join(dataset_root, "labelsTr")
    
    image_fns = sorted(filter(lambda x: x.endswith(".nii.gz"), os.listdir(image_root)))
    label_fns = sorted(filter(lambda x: x.endswith(".nii.gz"), os.listdir(label_root)))
    image_files = [os.path.join(image_root, x) for x in image_fns]
    label_files = [os.path.join(label_root, x) for x in label_fns]
    
    dataset = makeDataset(image_files, label_files, device=device, class_sample_ratios=[1,5,2,1,1], 
                          patch_size=[128,128,128], batch_size=4, num_classes=5,
                          rank=rank, world_size=world_size)
    
    repeats = 2
    dataloader = ThreadDataLoader(dataset, num_workers=0, repeats=repeats, buffer_size=4, shuffle=True)
    #dataloader = DataLoader(dataset, num_workers=2, prefetch_factor=2)

    logger_train.info("%d training volumes"%(len(dataset)))

    # SETUP LR AND OTHER PARAMETERS
    lr = 0.003
    weight_decay = 0.001
    num_epochs = 900
    save_freq = 50

    network = UNet3D(in_channels=1, num_classes=5, strides=[1,2,2,2,2,2], channels=[16,32,64,128,256,512], prelu=True)
    network = network.to(device)
    network = nn.SyncBatchNorm.convert_sync_batchnorm(network)
    network = DistributedDataParallel(network, device_ids=[device])

    criterion = GeneralizedDiceCELoss(weight=torch.tensor([0.1, 0.6, 0.1, 0.1, 0.1]).to(device))
    
    # can customize optimizer and LR scheduler
    optimizer = Novograd(network.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, steps_per_epoch=len(dataset)*repeats,
                                                    epochs=num_epochs)
    scaler = GradScaler()

    # Train!
    t = time.time()
    for epoch in range(num_epochs):
        epoch_start = time.time()
        train_loss = 0
        for batch_num, data in enumerate(dataloader):
            network.train()
            optimizer.zero_grad()
            x, y = data["image"], data["label"]
            x, y = x.to(device), y.to(device)
            with autocast():
                outputs = network(x)
                loss = criterion(outputs, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += x.shape[0] * loss.item()
            scheduler.step()
        
        if epoch % save_freq == save_freq - 1:
            if rank == 0:
                torch.save(network.module.state_dict(), "%s/checkpoint_%d.pt" % (save_name, epoch + 1))

        

        print_text = 'Epoch: {}, Training Loss: {:.3f}, Time: {:0.1f}'.format(epoch, train_loss / len(dataset)/repeats,
                                                                              time.time() - epoch_start)
        print(rank, print_text)
        logger_train.info(print_text)
    
    print("time taken:", (time.time() - t) / 3600)
    logger_train.info("time taken: %0.2f" % ((time.time() - t) / 3600))
    
    dist.destroy_process_group()

if __name__ == "__main__":
    #free_port = str(find_free_port())

    #num_gpus = torch.cuda.device_count()
    #print("Training on %d gpus"%num_gpus)
    #mp.spawn(train, args=(num_gpus, free_port, ), nprocs=num_gpus, join=True)
    train()
    print("job done")