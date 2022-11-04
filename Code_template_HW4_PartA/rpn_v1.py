from multiprocessing.connection import wait
import torch
from torch.nn import functional as F
from torchvision import transforms
from torch import nn, Tensor
from utils import *
from dataset_v1 import *
import matplotlib.pyplot as plt
import torchvision
from scipy import stats as st
import pytorch_lightning as pl

from pytorch_lightning.callbacks import ModelCheckpoint
########################

from torchvision.utils import draw_bounding_boxes
import os
import copy
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, random_split, TensorDataset
from torchvision import transforms, utils
from  matplotlib.patches import Rectangle as rec
import numpy as np
import h5py
import cv2
import matplotlib.pyplot as plt
import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from scipy import ndimage
import torch.nn.functional as F

import torch.nn as nn
import pytorch_lightning as pl
import pytorch_lightning.loggers as pl_loggers
import pytorch_lightning.callbacks as pl_callbacks

# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
seed = 17
torch.manual_seed(seed);

from utils import *

from multiprocessing.connection import wait
import torch
from torch.nn import functional as F
from torchvision import transforms
from torch import nn, Tensor
import matplotlib.pyplot as plt
import torchvision
from scipy import stats as st

from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches





def histogram(bboxes_path):
    bboxes      = np.load(bboxes_path, allow_pickle=True)

    ar_list     = []
    scale_list  = []
    for each in bboxes:
        for each_box in each:
            x1 =  each_box[0] * (800/300)
            x2 =  each_box[2] * (800/300) 
            y1 = (each_box[1] * (1066/400)) + 11
            y2 = (each_box[3] * (1066/400)) + 11

            ar = np.abs((x2 - x1)/ (y2 - y1))
            ar_list.append(ar)
            scale = np.sqrt((x2-x1) * (y2-y1))
            scale_list.append(scale)

    mean_vals = [np.mean(scale_list),             np.mean(ar_list)]
    medn_vals = [np.median(scale_list),         np.median(ar_list)]
    mode_vals = [st.mode(scale_list)[0][0], st.mode(ar_list)[0][0]]

    plt.hist(ar_list)
    plt.title('Aspect Ratio Histogram')
    plt.figure()
    plt.hist(scale_list)
    plt.title('Scale Histogram')
    plt.show()
    
    return mean_vals, medn_vals, mode_vals

class RPNHead(pl.LightningModule):

    def __init__(self,  anchors_param=dict(ratio=0.8,scale= 256, grid_size=(50, 68), stride=16)):
        # Initialize the backbone, intermediate layer clasifier and regressor heads of the RPN
        super(RPNHead,self).__init__()

        self.train_losses = []
        self.val_losses   = []

        self.sy     = 800
        self.sx     = 1088
        # TODO Define Backbone
        self.backbone = nn.Sequential(nn.Conv2d(3, 16,    kernel_size=5, stride=1, padding="same"),
                                      nn.BatchNorm2d(16),
                                      nn.ReLU(),
                                      nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
                                      nn.Conv2d(16, 32,   kernel_size=5, stride=1, padding="same"),
                                      nn.BatchNorm2d(32),
                                      nn.ReLU(),
                                      nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
                                      nn.Conv2d(32, 64,   kernel_size=5, stride=1, padding="same"),
                                      nn.BatchNorm2d(64),
                                      nn.ReLU(),
                                      nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
                                      nn.Conv2d(64, 128,  kernel_size=5, stride=1, padding="same"),
                                      nn.BatchNorm2d(128),
                                      nn.ReLU(),
                                      nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
                                      nn.Conv2d(128, 256, kernel_size=5, stride=1, padding="same"),
                                      nn.BatchNorm2d(256),
                                      nn.ReLU())

        # TODO  Define Intermediate Layer
        self.intermediate_layer = nn.Sequential(nn.Conv2d(256, 256, 3,   stride=1, padding="same"), 
                                                nn.BatchNorm2d(256), 
                                                nn.ReLU())

        # TODO  Define Proposal Classifier Head
        self.proposal_classifier_head = nn.Sequential(nn.Conv2d(256,1,1, stride=1, padding="same"),
                                                      nn.Sigmoid())

        # TODO Define Proposal Regressor Head
        self.proposal_regressor_head = nn.Sequential(nn.Conv2d(256,4,1, stride=1, padding="same"))

        #  find anchors
        self.anchors_param=anchors_param
        self.anchors=self.create_anchors(self.anchors_param['ratio'],self.anchors_param['scale'],self.anchors_param['grid_size'],self.anchors_param['stride'])
        self.ground_dict={}





    # Forward  the input through the backbone the intermediate layer and the RPN heads
    # Input:
    #       X: (bz,3,image_size[0],image_size[1])}
    # Ouput:
    #       logits: (bz,1,grid_size[0],grid_size[1])}
    #       bbox_regs: (bz,4, grid_size[0],grid_size[1])}
    def forward(self, X):

        #TODO forward through the Backbone
        x = self.backbone(X)

        #TODO forward through the Intermediate layer
        inter_out = self.intermediate_layer(x)

        #TODO forward through the Classifier Head
        logits = self.proposal_classifier_head(inter_out)

        #TODO forward through the Regressor Head
        bbox_regs  = self.proposal_regressor_head(inter_out)

        assert logits.shape[1:4]==(1,self.anchors_param['grid_size'][0],self.anchors_param['grid_size'][1])
        assert bbox_regs.shape[1:4]==(4,self.anchors_param['grid_size'][0],self.anchors_param['grid_size'][1])

        return logits, bbox_regs




    # Forward input batch through the backbone
    # Input:
    #       X: (bz,3,image_size[0],image_size[1])}
    # Ouput:
    #       X: (bz,256,grid_size[0],grid_size[1])
    def forward_backbone(self,X):
        #####################################
        # TODO forward through the backbone\
        X = self.backbone(X)
        #####################################
        assert X.shape[1:4]==(256,self.anchors_param['grid_size'][0],self.anchors_param['grid_size'][1])

        return X



    # This function creates the anchor boxes
    # Output:
    #       anchors: (grid_size[0],grid_size[1],4)
    def create_anchors(self, aspect_ratio, scale, grid_sizes, stride):
        ######################################
        # TODO create anchors
        anchors = torch.zeros((  grid_sizes[0], grid_sizes[1],4))
        xx, yy  = torch.meshgrid(torch.arange(grid_sizes[1]), torch.arange(grid_sizes[0]))
        anchors[:,:,0] = ((xx.T * stride) + (stride/2))
        anchors[:,:,1] = ((yy.T * stride) + (stride/2))
        
        h = scale/np.sqrt(aspect_ratio)
        w = h * aspect_ratio

        w_grid = torch.ones((grid_sizes[0], grid_sizes[1])) * w
        h_grid = torch.ones((grid_sizes[0], grid_sizes[1])) * h

        anchors[:,:,2] = w_grid
        anchors[:,:,3] = h_grid

        ######################################
        assert anchors.shape == (grid_sizes[0] , grid_sizes[1], 4)

        return anchors


    def get_anchors(self):
        return self.anchors


    # This function creates the ground truth for a batch of images by using
    # create_ground_truth internally
    # Input:
    #      bboxes_list: list:len(bz){(n_obj,4)}
    #      indexes:      list:len(bz)
    #      image_shape:  tuple:len(2)
    # Output:
    #      ground_clas: (bz,1,grid_size[0],grid_size[1])
    #      ground_coord: (bz,4,grid_size[0],grid_size[1])
    def create_batch_truth(self,bboxes_list,indexes,image_shape):
        #####################################
        # TODO create ground truth for a batch of images
        ground_clas_list  = []
        ground_coord_list = []
        for num in range(len(bboxes_list)):
          ground_clas,ground_coord=self.create_ground_truth(torch.from_numpy(bboxes_list[num]), indexes[num], (self.anchors_param['grid_size'][0],self.anchors_param['grid_size'][1]), self.anchors, image_size=image_shape)
          ground_clas_list.append(ground_clas)
          ground_coord_list.append(ground_coord)
        
        ground_clas  = torch.stack(ground_clas_list)
        ground_coord = torch.stack(ground_coord_list)
        #####################################
        assert ground_clas. shape[1:4]==(1,self.anchors_param['grid_size'][0],self.anchors_param['grid_size'][1])
        assert ground_coord.shape[1:4]==(4,self.anchors_param['grid_size'][0],self.anchors_param['grid_size'][1])

        return ground_clas, ground_coord


    # This function creates the ground truth for one image
    # It also caches the ground truth for the image using its index
    # Input:
    #       bboxes:      (n_boxes,4)
    #       index:       scalar (the index of the image in the total dataset used for caching)
    #       grid_size:   tuple:len(2)
    #       anchors:     (grid_size[0],grid_size[1],4)
    # Output:
    #       ground_clas:  (1,grid_size[0],grid_size[1])
    #       ground_coord: (4,grid_size[0],grid_size[1])
    def create_ground_truth(self, bboxes, index, grid_size, anchors, image_size):
        key = str(index)
        if key in self.ground_dict:
            groundt, ground_coord = self.ground_dict[key]
            return groundt, ground_coord

        #####################################################
        # TODO create ground truth for a single image
        ground_clas  =  torch.ones(( 1,grid_size[0],grid_size[1]), dtype=torch.double) * (-1)
        ground_coord =  torch.ones((4,grid_size[0],grid_size[1]), dtype=torch.double) 

        anchors_xywh = anchors.flatten().reshape(-1,4)

        ax      = anchors_xywh[:,0]
        ay      = anchors_xywh[:,1]
        aw      = anchors_xywh[:,2]
        ah      = anchors_xywh[:,3]

        anc_x1  = ax - (aw/2.0)
        anc_y1  = ay - (ah/2.0)
        anc_x2  = ax + (aw/2.0)
        anc_y2  = ay + (ah/2.0)


        #Boxes
        bbox_xy     = bboxes
        box_x1      = bbox_xy[:,0]
        box_y1      = bbox_xy[:,1]
        box_x2      = bbox_xy[:,2]
        box_y2      = bbox_xy[:,3]
        
        bx          = (box_x1 + box_x2)/2.0
        by          = (box_y1 + box_y2)/2.0
        bw          = (box_x2 - box_x1)
        bh          = (box_y2 - box_y1) 
        bbox_xywh   = torch.vstack((bx,by,bw,bh)).T

        #Removing Cross boundary boxes
        invalid_list = torch.tensor([])
        invalid      = torch.where((anc_x1 < 0) | (anc_y1 < 0) |(anc_x2 >= 1088) | (anc_y2 >= 800))[0]
        row          = invalid // 68
        col          = invalid % 68
        invalid_idx  = torch.vstack((row, col)).T
        invalid_list = torch.cat((invalid_list, invalid_idx))
        ground_clas[0, row, col] = -1

        valid_anchor_idx = torch.where((anc_x1 >= 0) & (anc_y1 >= 0) & (anc_x2 < 1088) & (anc_y2 < 800))[0]
        row_anc          = valid_anchor_idx // 68
        col_anc          = valid_anchor_idx % 68
        valid_anchor     = anchors[row_anc, col_anc, :].reshape(-1,4)

        anchors_xywh    = valid_anchor.flatten().reshape(-1,4)

        ax      = anchors_xywh[:,0]
        ay      = anchors_xywh[:,1]
        aw      = anchors_xywh[:,2]
        ah      = anchors_xywh[:,3]

        anc_x1  = ax - (aw/2.0)
        anc_y1  = ay - (ah/2.0)
        anc_x2  = ax + (aw/2.0)
        anc_y2  = ay + (ah/2.0)

        anchor_xy = torch.vstack((anc_x1, anc_y1, anc_x2, anc_y2)).T
        #######################################################
        assigned    = torch.tensor([])
        bbox_dict   = {} 

        for idx in range(len(bboxes)):

            bbox_xy     = bboxes[idx].reshape(1, -1)
            box_x1      = bbox_xy[:,0]
            box_y1      = bbox_xy[:,1]
            box_x2      = bbox_xy[:,2]
            box_y2      = bbox_xy[:,3]
            
            bx          = (box_x1 + box_x2)/2.0
            by          = (box_y1 + box_y2)/2.0
            bw          = (box_x2 - box_x1)
            bh          = (box_y2 - box_y1) 
            bbox_xywh   = torch.vstack((bx,by,bw,bh)).T
            
            #Calculate IOU
            IOU_matrix = torchvision.ops.box_iou(bbox_xy, anchor_xy)

            IOU_max    = torch.max(IOU_matrix).item()
            
            #Threshold somewhere other than >0.7
            true_box = torch.logical_or(IOU_matrix == IOU_max, IOU_matrix >= 0.7)
            true_idx = torch.where(true_box)[1]

            val_row  = row_anc[true_idx] #//68
            val_col  = col_anc[true_idx] #% 68
            valid    = torch.vstack((val_row, val_col)).T

            ground_clas[0,val_row, val_col] = 1

            bbox_dict[idx]  = [valid, IOU_matrix[0,true_idx] , bbox_xywh]


            #Threshold less than 0.3 and invalid
            less_thresh_idx = torch.where((IOU_matrix < 0.3) & (~true_box))[1]
            less_thresh_row = row_anc[less_thresh_idx]#//68
            less_thresh_col = col_anc[less_thresh_idx]#%68
            less_thresh     = torch.vstack((less_thresh_row, less_thresh_col)).T
            dont_del_idx    = torch.where(~torch.isin(less_thresh, assigned).all(axis = 1))[0]
            assigned        = torch.cat((assigned,valid))

            ground_clas[0,less_thresh[dont_del_idx,0], less_thresh[dont_del_idx,1]]  = 0
                     
        indicies    = torch.where(ground_clas == 1) 
        xy_ind      = indicies[1:]
        xy_ind      = torch.vstack(xy_ind[:]).T
        ind_flat    = (indicies[1] * 68) + indicies[2]  

        bbox_xy     = bboxes.reshape(1,-1)

        anchors_xywh = anchors.flatten().reshape(-1,4)

        ax      = anchors_xywh[:,0]
        ay      = anchors_xywh[:,1]
        aw      = anchors_xywh[:,2]
        ah      = anchors_xywh[:,3]

        anc_x1  = ax - (aw/2.0)
        anc_y1  = ay - (ah/2.0)
        anc_x2  = ax + (aw/2.0)
        anc_y2  = ay + (ah/2.0)

        new_f_anchors = torch.vstack((anc_x1, anc_y1, anc_x2, anc_y2)).T


        IOU_matrix  = torchvision.ops.box_iou(bboxes, new_f_anchors[ind_flat])
        IOU_max     = torch.max(IOU_matrix, dim=0)
        
        bbox_xy     = bboxes
        box_x1      = bbox_xy[:,0]
        box_y1      = bbox_xy[:,1]
        box_x2      = bbox_xy[:,2]
        box_y2      = bbox_xy[:,3]
        
        bx          = (box_x1 + box_x2)/2.0
        by          = (box_y1 + box_y2)/2.0
        bw          = (box_x2 - box_x1)
        bh          = (box_y2 - box_y1) 

        bbox_xywh   = torch.vstack((bx,by,bw,bh)).T


        for i, each_ind in enumerate(IOU_max[1]):

            ground_coord[0, indicies[1][i], indicies[2][i]] = (bbox_xywh[each_ind][0] - anchors[indicies[1][i], indicies[2][i]][0])/  anchors[indicies[1][i], indicies[2][i]][2]
            ground_coord[1, indicies[1][i], indicies[2][i]] = (bbox_xywh[each_ind][1] - anchors[indicies[1][i], indicies[2][i]][1])/  anchors[indicies[1][i], indicies[2][i]][3]

            ground_coord[2, indicies[1][i], indicies[2][i]] = torch.log(bbox_xywh[each_ind][2] /  anchors[indicies[1][i], indicies[2][i]][2])
            ground_coord[3, indicies[1][i], indicies[2][i]] = torch.log(bbox_xywh[each_ind][3] /  anchors[indicies[1][i], indicies[2][i]][3])


        #####################################################

        self.ground_dict[key] = (ground_clas, ground_coord)

        assert ground_clas.shape==(1,grid_size[0],grid_size[1])
        assert ground_coord.shape==(4,grid_size[0],grid_size[1])

        return ground_clas, ground_coord


    # Compute the loss of the classifier
    # Input:
    #      p_out:     (positives_on_mini_batch)  (output of the classifier for sampled anchors with positive gt labels)
    #      n_out:     (negatives_on_mini_batch)  (output of the classifier for sampled anchors with negative gt labels)
    def loss_class(self,p_out,n_out):

        #torch.nn.BCELoss()
        # TODO compute classifier's loss
        loss    = torch.nn.BCELoss(reduction="mean")
        t_loss  = loss(p_out, n_out.float())
        return t_loss



    # Compute the loss of the regressor
    # Input:
    #       pos_target_coord: (positive_on_mini_batch,4) (ground truth of the regressor for sampled anchors with positive gt labels)
    #       pos_out_r: (positive_on_mini_batch,4)        (output of the regressor for sampled anchors with positive gt labels)
    def loss_reg(self,pos_target_coord,pos_out_r, non_zero = False):
            #torch.nn.SmoothL1Loss()
            # TODO compute regressor's loss
            loss        = torch.nn.SmoothL1Loss()
            loss_r = sum([loss(pos_target_coord[i], pos_out_r[i]) for i in range(4)]) if non_zero==False else torch.tensor(0)
            return loss_r



    # Compute the total loss
    # Input:
    #       clas_out: (bz,1,grid_size[0],grid_size[1])
    #       regr_out: (bz,4,grid_size[0],grid_size[1])
    #       targ_clas:(bz,1,grid_size[0],grid_size[1])
    #       targ_regr:(bz,4,grid_size[0],grid_size[1])
    #       l: lambda constant to weight between the two losses
    #       effective_batch: the number of anchors in the effective batch (M in the handout)
    def compute_loss(self, clas_out, regr_out, targ_clas, targ_regr, l=1, effective_batch=50):
            #############################
            # TODO compute the total loss
            #############################

            M = effective_batch
            
            positives_indexes   = torch.where(targ_clas == 1)
            negative_indexes    = torch.where(targ_clas == 0)

            p_idx               = int(min(positives_indexes[0].shape[0], effective_batch/2))
            n_idx               = int(effective_batch - p_idx)

            rand_p_idx            = torch.randperm(positives_indexes[0].shape[0])
            rand_n_idx            = torch.randperm(negative_indexes[0].shape[0])

            final_pos_indexes = (positives_indexes[0][rand_p_idx[:p_idx]], positives_indexes[1][rand_p_idx[:p_idx]], positives_indexes[2][rand_p_idx[:p_idx]], positives_indexes[3][rand_p_idx[:p_idx]])            
            final_neg_indexes = (negative_indexes[0][rand_n_idx[:n_idx]], negative_indexes[1][rand_n_idx[:n_idx]], negative_indexes[2][rand_n_idx[:n_idx]], negative_indexes[3][rand_n_idx[:n_idx]])

            final_gt = targ_clas[final_pos_indexes]
            pos_class_pred = clas_out[final_pos_indexes]
            loss1 = self.loss_class(pos_class_pred, final_gt)

            ng_class_gt = targ_clas[final_neg_indexes]
            ng_class_pred = clas_out[final_neg_indexes]
            loss2 = self.loss_class(ng_class_pred, ng_class_gt)
                        
            loss_c = loss1 + loss2

            ###################################################################
            targ_class  = targ_clas.reshape(-1)
            clas_out    = clas_out.reshape(-1)
            pos_class   = (targ_class==1).nonzero()

            positives   = int( min(pos_class.shape[0], M/2))

            pos_class   = pos_class[torch.randperm(pos_class.shape[0]), :]
            pos_class   = pos_class[:positives, :]

            non_zero=False

            if(pos_class.shape[0]==0):
              non_zero= True

            mask_targ_regr  = targ_regr.permute(1,0,2,3).reshape(4,-1)[:, pos_class]
            mask_regr_pred  = regr_out.permute(1,0,2,3).reshape(4,-1)[:, pos_class]
            loss_r          = self.loss_reg(mask_targ_regr, mask_regr_pred, non_zero)

            loss = loss_c + l*loss_r

            return loss, loss_c, loss_r



    # Post process for the outputs for a batch of images
    # Input:
    #       out_c:  (bz,1,grid_size[0],grid_size[1])}
    #       out_r:  (bz,4,grid_size[0],grid_size[1])}
    #       IOU_thresh: scalar that is the IOU threshold for the NMS
    #       keep_num_preNMS: number of masks we will keep from each image before the NMS
    #       keep_num_postNMS: number of masks we will keep from each image after the NMS
    # Output:
    #       nms_clas_list: list:len(bz){(Post_NMS_boxes)} (the score of the boxes that the NMS kept)
    #       nms_prebox_list: list:len(bz){(Post_NMS_boxes,4)} (the coordinates of the boxes that the NMS kept)
    def postprocess(self,out_c,out_r, IOU_thresh=0.5, keep_num_preNMS=50, keep_num_postNMS=10):
        ####################################
        # TODO postprocess a batch of images
        pre_nms_matrix_list = []
        scores_sorted_list  = []
        nms_clas_list       = []
        nms_prebox_list     = []
        for idx in range(len(out_c)):
            scores_sorted, pre_nms_matrix, nms_clas, nms_prebox = self.postprocessImg(out_c[idx],out_r[idx], IOU_thresh,keep_num_preNMS, keep_num_postNMS)
            scores_sorted_list.append(scores_sorted)
            pre_nms_matrix_list.append(pre_nms_matrix)
            nms_clas_list.append(nms_clas)
            nms_prebox_list.append(nms_prebox)
        #####################################
        return scores_sorted_list, pre_nms_matrix_list, nms_clas_list, nms_prebox_list



    # Post process the output for one image
    # Input:
    #      mat_clas: (1,grid_size[0],grid_size[1])}  (scores of the output boxes)
    #      mat_coord: (4,grid_size[0],grid_size[1])} (encoded coordinates of the output boxes)
    # Output:
    #       nms_clas: (Post_NMS_boxes)
    #       nms_prebox: (Post_NMS_boxes,4) (decoded coordinates of the boxes that the NMS kept)
    def postprocessImg(self,mat_clas, mat_coord, IOU_thresh,keep_num_preNMS, keep_num_postNMS):
            ######################################
            # TODO postprocess a single image

            #convert to x,y,w,h -> x1,y1,x2,y2
            # remove cross boundanry bounding boxes
            # find iou of bounding bozes with each other
            # sort them out 
            # pass them to NMS
            ##################################################

            anchors = self.get_anchors()
            mat_coord = mat_coord.permute(1,2,0)
            
            # x = (tx * wa) + xa
            x = (mat_coord[:,:, 0] * anchors[:,:,2]) + anchors[:,:,0]  

            # y = (ty * ha) + ya
            y = (mat_coord[:,:, 1] * anchors[:,:,3]) + anchors[:,:,1]  
            
            # w = wa * torch.exp(tw) 
            w = anchors[:,:,2] * torch.exp(mat_coord[:,:, 2])

            # h = ha * torch.exp(th)
            h = anchors[:,:,3] * torch.exp(mat_coord[:,:, 3])
                    
            x1          = x - (w/2.0)
            y1          = y - (h/2.0)
            x2          = x + (w/2.0)
            y2          = y + (h/2.0) 

            decoded_boxes_xywh = torch.stack((x,y,w,h))
            decoded_boxes_xy   = torch.stack((x1,y1,x2,y2))
            
            decoded_boxes_xy_reshape = decoded_boxes_xy.permute(2,1,0).reshape(-1,4)


            valid_dec_box_idx = torch.where((decoded_boxes_xy_reshape[:,0] >= 0) & (decoded_boxes_xy_reshape[:,1] >= 0) & (decoded_boxes_xy_reshape[:,2] < 1088) & (decoded_boxes_xy_reshape[:,3] < 800))[0]

            valid_dec_boxes  = decoded_boxes_xy_reshape[valid_dec_box_idx]

            mat_clas_flat   = mat_clas.flatten().reshape(-1)
            sorted_indicies = torch.sort( mat_clas_flat[valid_dec_box_idx], descending=True)

            scores_sorted       = sorted_indicies[0][:keep_num_preNMS]
            pre_nms_matrix      = valid_dec_boxes[sorted_indicies[1][:keep_num_preNMS]]  
            nms_clas,nms_prebox = self.NMS(sorted_indicies[0][:keep_num_preNMS], pre_nms_matrix, thresh = 0.5)


            return scores_sorted, pre_nms_matrix, nms_clas[:keep_num_postNMS], nms_prebox[:keep_num_postNMS]



    # Input:
    #       clas: (top_k_boxes) (scores of the top k boxes)
    #       prebox: (top_k_boxes,4) (coordinate of the top k boxes)
    # Output:
    #       nms_clas: (Post_NMS_boxes)
    #       nms_prebox: (Post_NMS_boxes,4)
    def NMS(self,clas, prebox, thresh):
        ##################################
        # TODO perform NMS
        #call SOLO
        #classifier output 
        #Prebox is the output of 
        #pass the scores and the corresponding bounding boxes
        ##################################
        prebox_cp = torch.clone(prebox)
        clas_cp   = torch.clone(clas)

        ped_iou = torchvision.ops.box_iou(prebox_cp, prebox_cp)
        i = 0
        rows_to_delete = []
        for i in range(ped_iou.shape[0]):
            if i not in rows_to_delete:
                j = i+1
                while j < ped_iou.shape[1]:
                    if (ped_iou[i,j] > thresh):
                        rows_to_delete.append(j)
                    j+=1
        prebox_cp = torch.tensor(np.delete(prebox_cp.cpu().detach().numpy(), list(set(rows_to_delete)), axis = 0))
        clas_cp   = torch.tensor(np.delete(clas_cp.cpu().detach().numpy(), list(set(rows_to_delete))))
        
        return clas_cp,prebox_cp



    def training_step(self, batch, batch_idx):
        images, bounding_boxes, indexes     = batch['images'], batch['bbox'], batch['index']
        images_re                           = torch.stack(images[:])
        gt,ground_coord                     = self.create_batch_truth(bounding_boxes, indexes, (800, 1088))  
        logits, bbox_regs                   = self.forward(images_re)

        loss, loss_c, loss_r = self.compute_loss(logits.to(device), bbox_regs.to(device), gt.to(device), ground_coord.to(device))

        self.log("train_loss",              loss,       prog_bar=True)
        self.log("train_class_loss",        loss_c,     prog_bar=True)
        self.log("train_regression_loss",   loss_r,     prog_bar=True)


        return {"loss": loss, "train_class_loss": loss_c, "train_regression_loss": loss_r}

    def training_epoch_end(self, outputs):

        train_loss              = torch.tensor([output["loss"]                  for output in outputs]).mean().item()
        train_class_loss        = torch.tensor([output["train_class_loss"]      for output in outputs]).mean().item()
        train_regression_loss   = torch.tensor([output["train_regression_loss"] for output in outputs]).mean().item()

        self.train_losses.append((train_loss, train_class_loss, train_regression_loss))

    def validation_step(self, batch, batch_idx):
        images, bounding_boxes, indexes     = batch['images'], batch['bbox'], batch['index']
        images_re                           = torch.stack(images[:])
        gt,ground_coord                     = self.create_batch_truth(bounding_boxes, indexes, (800, 1088))  
        logits, bbox_regs                   = self.forward(images_re)

        val_loss, loss_c, loss_r = self.compute_loss(logits.to(device), bbox_regs.to(device), gt.to(device), ground_coord.to(device))

        self.log("val_loss",              val_loss,   prog_bar=True)
        self.log("val_class_loss",        loss_c,     prog_bar=True)
        self.log("val_regression_loss",   loss_r,     prog_bar=True)

        logits = torch.where(logits >= 0.5, 1, 0)
        point_wise_accuracy = torch.sum(gt.flatten() == logits.flatten())/float(logits.flatten().shape[0])
        print('point_wise_accuracy', point_wise_accuracy)

        return {"val_loss": val_loss, "val_class_loss": loss_c, "val_regression_loss": loss_r}
        
    def validation_epoch_end(self, outputs):

        val_loss                = torch.tensor([output["val_loss"]            for output in outputs]).mean().item()
        val_class_loss          = torch.tensor([output["val_class_loss"]      for output in outputs]).mean().item()
        val_regression_loss     = torch.tensor([output["val_regression_loss"] for output in outputs]).mean().item()

        self.val_losses.append((val_loss, val_class_loss, val_regression_loss))


    def configure_optimizers(self):
        opt     = torch.optim.SGD(self.parameters(), lr=0.01, weight_decay=1e-4, momentum=0.9)
        sched   = {"scheduler": torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[26, 32], gamma=0.1)}

        return {"optimizer": opt, "lr_scheduler": sched}


def plot_bounding_box(image, labels):
    fig = plt.figure(figsize=(30, 10))
    ax = fig.add_subplot()
    ax.imshow(image)
    for i in range(len(labels)):
        rect = patches.Rectangle((labels[i][0], labels[i][1],), labels[i][2] - labels[i][0], labels[i][3] - labels[i][1] , fill=False,color='red')
        ax.add_patch(rect)
    plt.show()

    
# if __name__=="__main__":
#     ######################################################################################################################################
    
#     imgs_path   = '/home/josh/Desktop/CIS680/HW4/FasterRCNN/data/hw3_mycocodata_img_comp_zlib.h5'
#     masks_path  = '/home/josh/Desktop/CIS680/HW4/FasterRCNN/data/hw3_mycocodata_mask_comp_zlib.h5'
#     labels_path = '/home/josh/Desktop/CIS680/HW4/FasterRCNN/data/hw3_mycocodata_labels_comp_zlib.npy'
#     bboxes_path = '/home/josh/Desktop/CIS680/HW4/FasterRCNN/data/hw3_mycocodata_bboxes_comp_zlib.npy'
#     paths = [imgs_path, masks_path, labels_path, bboxes_path]
#     # load the data into data.Dataset
#     dataset = BuildDataset(paths)

#     # build the dataloader
#     # set 20% of the dataset as the training data
#     full_size = len(dataset)
#     train_size = int(full_size * 0.8)
#     test_size = full_size - train_size
#     # random split the dataset into training and testset

#     train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
#     rpn_net = RPNHead()
#     # push the randomized training data into the dataloader

#     train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=0)
#     test_loader  = DataLoader(test_dataset, batch_size=2, shuffle=False, num_workers=0)
#     batch_size = 10
#     train_build_loader = BuildDataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
#     train_loader = train_build_loader.loader()
#     test_build_loader = BuildDataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
#     test_loader = test_build_loader.loader()

#     ########################################################################################################################################################################################
#     model2 = RPNHead()
#     trainer = pl.Trainer()
#     chk_path = "/home/josh/Desktop/CIS680/HW4/FasterRCNN/Code_template_HW4_PartA/32Epochstrain/train_lossepoch=23-train_loss=1.12.ckpt"
#     model2 = RPNHead.load_from_checkpoint(chk_path)

#     # for i,batch in enumerate(test_loader,0):
#     #     images = batch['images']
#     #     images = torch.stack(images[:])
#     #     indexes= batch['index']
#     #     boxes  = batch['bbox']

#     #     gt,ground_coord     = model2.create_batch_truth(boxes, indexes, (800,1088))
#     #     logits, bbox_regs   = model2.forward(images)

#     #     indexes = torch.where(gt == 1)
#     #     num = torch.sum(logits[indexes])
#     #     accuracy = num/len(indexes[0])

#     #     # if i == 10:
#     #     print(accuracy)
#     #     break

#     ######################################################################################################################################
#     #################################################Checkpiint INitialization############################################################

#     # val_checkpoint_callback = ModelCheckpoint(
#     #     monitor="val_loss",
#     #     dirpath="./training_data_new_model_10",
#     #     filename="val_loss{epoch:02d}-{val_loss:.2f}",
#     #     save_top_k=3,
#     #     mode="min",
#     # )
#     # train_checkpoint_callback = ModelCheckpoint(
#     #     monitor="train_loss",
#     #     dirpath="./training_data_new_model_10",
#     #     filename="train_loss{epoch:02d}-{train_loss:.2f}",
#     #     save_top_k=3,
#     #     mode="min",
#     # )
#     # model = RPNHead()
#     # tb_logger = pl_loggers.TensorBoardLogger("logs9/")
#     # trainer = pl.Trainer(gpus=0,logger=tb_logger, max_epochs=36,callbacks=[val_checkpoint_callback,train_checkpoint_callback])
#     # # trainer.fit(model,test_loader)
#     # trainer.validate(model, test_loader)
    

#     #######################################################################################################################################
#     ############################################################PLOTTING###################################################################

#     # model2 = RPNHead()
#     # trainer = pl.Trainer()
#     # chk_path = "/home/josh/Desktop/CIS680/HW4/FasterRCNN/Code_template_HW4_PartA/train_lossepoch=33-train_loss=1.29.ckpt"
#     # model2 = RPNHead.load_from_checkpoint(chk_path)
#     # trainer.test()

#     #Postprocess
#     for i,batch in enumerate(train_loader,0):
#         images = batch['images']
#         images = torch.stack(images[:])
#         # print(images.shape)

#         logits, bbox_regs = model2.forward(images)
#         scores_sorted_list, pre_nms_matrix_list, nms_clas_list, nms_prebox_list = model2.postprocess(logits, bbox_regs, IOU_thresh = 0.5 ,keep_num_preNMS = 50, keep_num_postNMS = 20)

#         # decoded_coord=output_decoding(flatten_coord,flatten_anchors)

#         plot_bounding_box(images[0].permute(1,2,0), pre_nms_matrix_list[0].detach().cpu().numpy())
#         plot_bounding_box(images[0].permute(1,2,0), nms_prebox_list[0].detach().cpu().numpy())

#         if i == 10:
#             break



#     # histogram(bboxes_path)













