import numpy as np
import scipy
from skimage import measure

def filter_small_lesions(img: np.ndarray, threshold=200):
    """
    Given image, sets small lesions as background.
    """
    new_img = img.copy()
    lesion = new_img == 1
    label = measure.label(lesion)
    for i in range(label.max()):
        mask = label == i+1
        if mask.sum() < threshold:
            new_img[mask] = 0
    return new_img

def filter_far_and_small_lesions(img: np.ndarray, dist_threshold=10, size_threshold=200):
    # postprocessing
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

def distance_transform(img: np.ndarray, target_label=3):
    """
    Get distance map from closest target label
    target_label == 3 for bone, 4 for teeth
    """
    img = img.copy()
    img[img!=target_label] = -1
    img[img==target_label] = 0
    return scipy.ndimage.distance_transform_cdt(img)

def sophisticated_filter_lesions(img: np.ndarray):
    """
    Based on decision trees trained on manually extracted features in training data,
    while keeping high recall.
    """
    new_img = img.copy()
    distance_bg = distance_transform(lab, target_label=0)
    distance_bone = distance_transform(lab, target_label=3)
    distance_teeth = distance_transform(lab, target_label=4)
    lesion_label = measure.label(lab == 1)
    rp = measure.regionprops(lesion_label)
    for i in range(lesion_label.max()):
        lesion_dist_bg = distance_bg[lesion_label==i+1]
        lesion_dist_bone = distance_bone[lesion_label==i+1]
        lesion_dist_teeth = distance_teeth[lesion_label==i+1]
        if rp[i].area < 200:
            keep_lesion = False
        elif lesion_dist_bg.std() < 0.365:
            keep_lesion = False
        elif lesion_dist_bg.std()/lesion_dist_bg.mean() < 0.145:
            keep_lesion = False
        elif lesion_dist_bone.std() < 0.420:
            keep_lesion = False
        elif lesion_dist_bone.std()/lesion_dist_bone.mean() < 0.35:
            keep_lesion = False
        elif lesion_dist_teeth.std() < 0.757:
            keep_lesion = False
        elif lesion_dist_teeth.std()/lesion_dist_teeth.mean() < 0.36:
            keep_lesion = False
        else:
            keep_lesion = True
        if not keep_lesion:
            new_img[lesion_label==i+1] = 0
    return new_img