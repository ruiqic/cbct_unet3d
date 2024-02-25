import torch
import numpy as np
from skimage import measure
from typing import Iterable, Union
from collections import namedtuple
from torchmetrics.classification import MulticlassF1Score, MulticlassRecall, MulticlassPrecision

PrecisonAndRecall = namedtuple("PrecisionAndRecall", ["precision", "recall"])
DetectionScore = namedtuple("DetectionScore", ["detected", "missed", "false_lesion"])

def get_detection_score(gt: np.ndarray, pred: np.ndarray, iou_threshold=0.0) -> DetectionScore:
    """
    Given ground truth and prediction volume : np.ndarray, 
    get the number of TP (detected), FN (missed), and FP (false_lesion) in a named tuple

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
    """
    Given an iterable of ground truths and an iterable of predictions 
    in the same order, calcualtes the precision and recall scores with 
    the specified IoU threshold. The ground truth and predicted lesion 
    must meet the IoU threshold to be counted as detected. 
    """
    tp = fp = fn = 0
    for gt, pred in zip(gts, preds):
        detection_score = get_detection_score(gt, pred, iou_threshold=iou)
        tp += detection_score.detected
        fp += detection_score.false_lesion
        fn += detection_score.missed
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    return PrecisonAndRecall(precision, recall)

def calc_dice_scores(gts: Iterable[Union[np.ndarray, torch.Tensor]], 
                    preds: Iterable[Union[np.ndarray, torch.Tensor]]) -> np.ndarray:
    """
    Given an iterable of ground truths and an iterable of predictions 
    in the same order, calcualtes the pixel-level dice scores 
    for the 5 classes for each volume. 
    """
    scores = []
    dice_scorer = MulticlassF1Score(num_classes=5, average=None)
    for gt, pred in zip(gts, preds):
        if isinstance(gt, np.ndarray):
            gt = torch.from_numpy(gt)
        if isinstance(pred, np.ndarray):
            pred = torch.from_numpy(pred)
        score = dice_scorer(pred, gt)
        scores.append(score.numpy())
    return np.asarray(scores)

def calc_precision_scores(gts: Iterable[Union[np.ndarray, torch.Tensor]], 
                    preds: Iterable[Union[np.ndarray, torch.Tensor]]) -> np.ndarray:
    """
    Given an iterable of ground truths and an iterable of predictions 
    in the same order, calcualtes the pixel-level precision scores 
    for the 5 classes for each volume. 
    """
    scores = []
    dice_scorer = MulticlassPrecision(num_classes=5, average=None)
    for gt, pred in zip(gts, preds):
        if isinstance(gt, np.ndarray):
            gt = torch.from_numpy(gt)
        if isinstance(pred, np.ndarray):
            pred = torch.from_numpy(pred)
        score = dice_scorer(pred, gt)
        scores.append(score.numpy())
    return np.asarray(scores)

def calc_recall_scores(gts: Iterable[Union[np.ndarray, torch.Tensor]], 
                    preds: Iterable[Union[np.ndarray, torch.Tensor]]) -> np.ndarray:
    """
    Given an iterable of ground truths and an iterable of predictions 
    in the same order, calcualtes the pixel-level recall scores 
    for the 5 classes for each volume. 
    """
    scores = []
    dice_scorer = MulticlassRecall(num_classes=5, average=None)
    for gt, pred in zip(gts, preds):
        if isinstance(gt, np.ndarray):
            gt = torch.from_numpy(gt)
        if isinstance(pred, np.ndarray):
            pred = torch.from_numpy(pred)
        score = dice_scorer(pred, gt)
        scores.append(score.numpy())
    return np.asarray(scores)

def get_false_positives(gt: np.ndarray, pred: np.ndarray, iou_threshold=0.0):
    pred_lesion_label = measure.label(pred == 1)
    gt_lesion_label = measure.label(gt == 1)
    
    num_gt_areas = gt_lesion_label.max()
    num_pred_areas = pred_lesion_label.max()
    pred_validity = np.zeros(num_pred_areas)
    for i in range(num_gt_areas):
        gt_mask = gt_lesion_label == i+1
        for j in range(num_pred_areas):
            pred_mask = pred_lesion_label == j+1
            
            intersect = (gt_mask & pred_mask).sum()
            union = (gt_mask | pred_mask).sum()
            if intersect / union > iou_threshold:
                pred_validity[j] = pred_validity[j] + 1
    false_positives = np.zeros_like(pred)
    for j in pred_validity:
        if j == 0:
            false_positives[pred_lesion_label == j+1] = 1
    return false_positives
