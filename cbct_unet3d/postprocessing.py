import numpy as np
import torch
from skimage import measure
from skimage import morphology

def distance_transform(img, target_label=3):
    """
    Get distance map from closest target label
    target_label == 3 for bone, 4 for teeth
    """
    img = img.copy()
    img[img!=target_label] = -1
    img[img==target_label] = 0
    return scipy.ndimage.distance_transform_cdt(img)

def filter_far_and_small(img, dist_threshold=10, size_threshold=200):
    # postprocessing
    """
    img : np.ndarray
    """
    
    new_img = img.copy()
    bone_distance = distance_transform(img, target_label=3)
    teeth_distance = distance_transform(img, target_label=4)
    dist = bone_distance + teeth_distance
    
    lesion = img == 1
    lesion_label = measure.label(lesion)
    
    for i in range(lesion_label.max()):
        this_lesion = lesion_label==i+1
        size = this_lesion.sum()
        distance = dist[this_lesion].min()
        if size < size_threshold or distance > dist_threshold:
            new_img[this_lesion] = 0
    
    return new_img

    