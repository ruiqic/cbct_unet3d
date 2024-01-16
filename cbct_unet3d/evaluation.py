import numpy as np
from skimage import measure
from typing import Iterable
from collections import namedtuple

PrecisonAndRecall = namedtuple("PrecisionAndRecall", ["precision", "recall"])
DetectionScore = namedtuple("DetectionScore", ["detected", "missed", "false_lesion"])

def confusion_table(gt: np.ndarray, pred: np.ndarray, iou_threshold=0.0) -> DetectionScore:
    """
    Given ground truth and prediction volume : np.ndarray, 
    get the number of TP (detected), FN (missed), and FP (false) lesions

    Use postprocessed ground truth to avoid tiny lesions being counted.
    """
    pred_lesion_label = measure.label(pred == 1)
    gt_lesion_label = measure.label(gt == 1)
    
    num_gt_areas = gt_lesion_label.max()
    num_pred_areas = pred_lesion_label.max()
    detected = missed = 0
    pred_validity = np.zeros(num_pred_areas)
    for i in range(num_gt_areas):
        gt_mask = gt_lesion_label == i+1
        found = False
        for j in range(num_pred_areas):
            pred_mask = pred_lesion_label == j+1
            
            intersect = (gt_mask & pred_mask).sum()
            union = (gt_mask | pred_mask).sum()
            if intersect / union > iou_threshold:
                found = True
                pred_validity[j] = pred_validity[j] + 1
        if found:
            detected += 1
        else:
            missed += 1
    false = (pred_validity == 0).sum()
    return DetectionScore(detected=detected, missed=missed, false_lesion=false)

def get_precision_recall(gts: Iterable[np.ndarray], 
                         preds: Iterable[np.ndarray], 
                         iou=0.0) -> PrecisonAndRecall:
    tp = fp = fn = 0
    for gt, pred in zip(gts, preds):
        detection_score = confusion_table(gt, pred, iou_threshold=iou)
        tp += detection_score.detected
        fp += detection_score.false_lesion
        fn += detection_score.missed
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    return PrecisonAndRecall(precision, recall)

