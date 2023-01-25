import numpy as np
import torch
from functools import partial
import torch
from torch.nn import functional as F
from torchvision import transforms
from torch import nn, Tensor
import matplotlib.pyplot as plt
import torchvision
from scipy import stats as st

def MultiApply(func, *args, **kwargs):
    pfunc = partial(func, **kwargs) if kwargs else func
    map_results = map(pfunc, *args)
  
    return tuple(map(list, zip(*map_results)))

# This function computes the IOU between two set of boxes
def IOU(boxA, boxB):
    ##################################
    #TODO compute the IOU between the boxA, boxB boxes
    boxA_conv = torch.zeros_like(boxA)
    boxA_conv[:,0] = boxA[:,0] - (boxA[:,2]/2)
    boxA_conv[:,2] = boxA[:,0] + (boxA[:,2]/2)
    boxA_conv[:,1] = boxA[:,1] - (boxA[:,3]/2)
    boxA_conv[:,3] = boxA[:,1] + (boxA[:,3]/2)

    

    iou_torched = torchvision.ops.box_iou(boxA_conv, boxB)
    ##################################
    return iou_torched



# This function flattens the output of the network and the corresponding anchors 
# in the sense that it concatenates  the outputs and the anchors from all the grid cells
# from all the images into 2D matrices
# Each row of the 2D matrices corresponds to a specific anchor/grid cell
# Input:
#       out_r: (bz,4,grid_size[0],grid_size[1])
#       out_c: (bz,1,grid_size[0],grid_size[1])
#       anchors: (grid_size[0],grid_size[1],4)
# Output:
#       flatten_regr: (bz*grid_size[0]*grid_size[1],4)
#       flatten_clas: (bz*grid_size[0]*grid_size[1])
#       flatten_anchors: (bz*grid_size[0]*grid_size[1],4)
def output_flattening(out_r,out_c,anchors):
    #######################################
    # TODO flatten the output tensors and anchors
    #######################################

    bz = out_r.squeeze(0).shape[0]
    flatten_regr = out_r.squeeze(0).permute(0,2,3,1).reshape(-1,4)
    flatten_clas = out_c.squeeze(1).reshape(-1)
    flatten_anchors = anchors.reshape(-1,4).repeat(bz,1)

    return flatten_regr, flatten_clas, flatten_anchors




# This function decodes the output that is given in the encoded format (defined in the handout)
# into box coordinates where it returns the upper left and lower right corner of the proposed box
# Input:
#       flatten_out: (total_number_of_anchors*bz,4)
#       flatten_anchors: (total_number_of_anchors*bz,4)
# Output:
#       box: (total_number_of_anchors*bz,4)
def output_decoding(flatten_out,flatten_anchors, device='cpu'):
    #######################################
    # TODO decode the 
    conv_box = torch.zeros_like(flatten_anchors)
    box = torch.zeros_like(flatten_anchors)
    conv_box[:,3] = torch.exp(flatten_out[:,3]) * flatten_anchors[:,3]
    conv_box[:,2] = torch.exp(flatten_out[:,2]) * flatten_anchors[:,2]
    conv_box[:,1] = (flatten_out[:,1] * flatten_anchors[:,2]) + flatten_anchors[:,1]
    conv_box[:,0] = (flatten_out[:,0] * flatten_anchors[:,3]) + flatten_anchors[:,0]

    box[:,0] = conv_box[:,0] - (conv_box[:,2]/2)
    box[:,1] = conv_box[:,1] - (conv_box[:,3]/2)
    box[:,2] = conv_box[:,0] + (conv_box[:,2]/2)
    box[:,3] = conv_box[:,1] + (conv_box[:,3]/2)
    #######################################
    return box