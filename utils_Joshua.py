import numpy as np
import torch
from functools import partial
import torchvision
from sklearn import metrics

def MultiApply(func, *args, **kwargs):
    pfunc = partial(func, **kwargs) if kwargs else func
    map_results = map(pfunc, *args)
  
    return tuple(map(list, zip(*map_results)))
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# This function compute the IOU between two set of boxes 
def IOU(boxA, boxB):
    iou = torchvision.ops.box_iou(boxA, boxB.to(device))
    return iou

# This function decodes the output of the box head that are given in the [t_x,t_y,t_w,t_h] format
# into box coordinates where it return the upper left and lower right corner of the bbox
# Input:
#       regressed_boxes_t: (total_proposals,4) ([t_x,t_y,t_w,t_h] format)
#       flatten_proposals: (total_proposals,4) ([x1,y1,x2,y2] format)
# Output:
#       box: (total_proposals,4) ([x1,y1,x2,y2] format)
def output_decodingd(regressed_boxes_t,flatten_proposals, device='cpu'):
    
    #COnvert Proposals from x1,y1,x2,y2 to x,y,w,h for decoding
    prop_x = (flatten_proposals[:,0] + flatten_proposals[:,2])/2.0
    prop_y = (flatten_proposals[:,1] + flatten_proposals[:,3])/2.0
    prop_w = (flatten_proposals[:,2] - flatten_proposals[:,0])
    prop_h = (flatten_proposals[:,3] - flatten_proposals[:,1]) 

    prop_xywh = torch.vstack((prop_x,prop_y,prop_w,prop_h)).T
    
    # x = (tx * wa) + xa
    x = (regressed_boxes_t[:,0] * prop_xywh[:,2]) + prop_xywh[:,0]  
    # y = (ty * ha) + ya
    y = (regressed_boxes_t[:,1] * prop_xywh[:,3]) + prop_xywh[:,1]    
    # w = wa * torch.exp(tw) 
    w = prop_xywh[:,2] * torch.exp(regressed_boxes_t[:,2])
    # h = ha * torch.exp(th)
    h = prop_xywh[:,3] * torch.exp(regressed_boxes_t[:,3])
    
    x1 = x - (w/2.0)
    y1 = y - (h/2.0)
    x2 = x + (w/2.0)
    y2 = y + (h/2.0) 

    box = torch.vstack((x1,y1,x2,y2)).T

    return box


def precision_recall_curve(predictions, targets, target_class):

    AP_sum = 0
    tp_fp_array               = []
    total_ground_truth_boxes  = 0
    for i in range(len(predictions)): # i represents batch number
        if len(predictions[i]) != 0:
            predictions_class       = predictions[i][torch.where(predictions[i][:, 0] == target_class)[0]]
            targets_class           = targets[i][torch.where(targets[i][:, 0] == target_class)[0]]
            iou = torchvision.ops.box_iou(predictions_class[:, 2:], targets_class[:, 1:])
            thresh                  = 0.5

            if (iou.numpy()).size > 0:
                ground_truth_boxes_tracker = []
                for row in range(iou.shape[0]):
                    max_thresh = torch.max(iou[row, :], 0)
                    ground_truth_matched = max_thresh[1]

                    if max_thresh[0].item() > thresh:
                            if ground_truth_matched.item() not in ground_truth_boxes_tracker:
                                tp_fp_array.append(torch.tensor([predictions_class[row, 1], 1, 0]))
                                ground_truth_boxes_tracker.append(ground_truth_matched.item())                  
                            else:
                                tp_fp_array.append(torch.tensor([predictions_class[row, 1], 0, 1]))
                    else:
                        tp_fp_array.append(torch.tensor([predictions_class[row, 1], 0, 1]))


        total_ground_truth_boxes += targets_class.shape[0]

    if len(tp_fp_array) == 0:
        return torch.tensor([0, 0]), torch.tensor([0, 0])


    tp_fp_array = torch.stack(tp_fp_array)
    tp_fp_array_sorted = torch.flip(tp_fp_array[tp_fp_array[:, 0].sort()[1]], dims=(0,))
    precision_list    = []
    recall_list       = []

    true_positive     = 0
    for i, each in enumerate(tp_fp_array_sorted):

        if each[1] == 1:
            true_positive += 1

        precision       = true_positive / (i+1)
        recall          = true_positive / total_ground_truth_boxes
        precision_list.append(torch.tensor(precision))
        recall_list.append(torch.tensor(recall))


    recall       = torch.stack(recall_list)
    precision    = torch.stack(precision_list)


    if len(recall) == 1 or len(precision) == 1:
        return torch.hstack((recall,torch.tensor(0))), torch.hstack((precision,torch.tensor(0)))
        
    return recall, precision



def average_precision(predictions, targets, target_class):
    recall, precision = precision_recall_curve(predictions, targets, target_class)
    AP = metrics.auc(recall, precision)
    return AP


def mean_average_precision(predictions, targets):
    classes = [1, 2, 3]
    mean_average_precision = [average_precision(predictions, targets, x) for x in classes]
    return mean_average_precision
    mean_average_precision = sum(mean_average_precision) / 3
    return mean_average_precision