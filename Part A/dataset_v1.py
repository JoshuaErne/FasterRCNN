import torch
from torchvision import transforms
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np
from utils import *
# from rpn_final_3 import *
import matplotlib.pyplot as plt
import matplotlib.patches as patches


class BuildDataset(torch.utils.data.Dataset):
    def __init__(self, path):
        #############################################
        # TODO Initialize  Dataset

        self.normalize = transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))
        
        self.images     = h5py.File(path[0],'r')['data']
        self.masks      = h5py.File(path[1],'r')['data']
        self.bboxes     = np.load(path[3], allow_pickle=True)
        self.labels     = np.load(path[2], allow_pickle=True)
        
        self.normalize  = transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))
        
        self.masks_stacked = []
        self.preprocess()

        self.transform  = transforms.Compose([transforms.Resize((800, 1066)), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        self.resize     = transforms.Resize((800, 1066))
        self.pad        = transforms.Pad((11,0))

        self.transform_images = transforms.Compose([transforms.Resize((800,1066)), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  transforms.Pad((11,0))])
        self.transform_masks  = transforms.Compose([transforms.Resize((800,1066)), transforms.Pad((11,0))])

        #############################################

    def preprocess(self):

        count = 0
        for i in range(len(self.labels)):
            temp = []
            for j in range(len(self.labels[i])):
                temp.append(self.masks[count])
                count += 1
            self.masks_stacked.append(temp)

    def __len__(self):
        return len(self.images)

    # In this function for given index we rescale the image and the corresponding  masks, boxes
    # and we return them as output
    # output:
        # transed_img
        # label
        # transed_mask
        # transed_bbox
        # index
    def __getitem__(self, index):
        ################################
        # TODO return transformed images,labels,masks,boxes,index
        ################################
        image = self.images[index]        
        mask  = self.masks_stacked[index]                      
        label = self.labels[index]                                    
        bbox  = self.bboxes[index]                                 
        
        # image = torch.tensor(image, dtype = torch.float)
        # mask  = torch.tensor(mask,  dtype = torch.float)
        # label = torch.tensor(label, dtype = torch.float)
        # bbox  = torch.tensor(bbox,  dtype = torch.float)


        transed_img, transed_mask, transed_bbox = self.pre_process_batch(image, mask, bbox)

        assert transed_img.shape     == (3,800,1088)
        assert transed_bbox.shape[0] == transed_mask.shape[0]

        
        return transed_img, label, transed_mask, transed_bbox, index

    # This function preprocess the given image, mask, box by rescaling them appropriately
    # output:
    #        img: (3,800,1088)
    #        mask: (n_box,800,1088)
    #        box: (n_box,4)
    def pre_process_batch(self, img, mask, bbox):
        #######################################
        # TODO apply the correct transformation to the images,masks,boxes
        #Image Transform
        img_norm = img/255.
        img_norm = torch.tensor(img_norm, dtype = torch.float).unsqueeze(0)
        img_norm = self.transform_images(img_norm)

        #Mask Transform
        mask_norm = torch.zeros((1, len(mask), 800, 1088))
        for idx in range(len(mask)):
            msk = mask[idx]/1.
            msk = torch.tensor(msk, dtype = torch.float).unsqueeze(0)
            msk = self.transform_masks(msk)
            msk[msk > 0.5] = 1
            msk[msk < 0.5] = 0
            mask_norm[:,idx] = msk

        #Box Transform
        bbox[:,0] = bbox[:,0] * (800/300)
        bbox[:,2] = bbox[:,2] * (800/300) 
        bbox[:,1] = (bbox[:,1] * (1066/400)) + 11
        bbox[:,3] = (bbox[:,3] * (1066/400)) + 11
        ######################################

        assert img_norm.squeeze(0).shape == (3, 800, 1088)
        assert bbox.shape[0] == mask_norm.squeeze(0).shape[0]

        return img_norm.squeeze(0), mask_norm.squeeze(0), bbox


class BuildDataLoader(torch.utils.data.DataLoader):
    def __init__(self, dataset, batch_size, shuffle, num_workers):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers


    # output:
    #  dict{images: (bz, 3, 800, 1088)
    #       labels: list:len(bz)
    #       masks: list:len(bz){(n_obj, 800,1088)}
    #       bbox: list:len(bz){(n_obj, 4)}
    #       index: list:len(bz)
    def collect_fn(self, batch):
        out_batch = {"images": [], "labels": [], "masks": [], "bbox": [], "index":[]}
        
        for i in range(len(batch)):
            out_batch["images"].append(batch[i][0])
            out_batch["labels"].append(batch[i][1])
            out_batch["masks"] .append(batch[i][2])
            out_batch["bbox"].append(batch[i][3])
            out_batch["index"].append(batch[i][4])

        return out_batch


    def loader(self):
        return DataLoader(self.dataset,
                          batch_size=self.batch_size,
                          shuffle=self.shuffle,
                          num_workers=self.num_workers,
                          collate_fn=self.collect_fn)


# if __name__ == '__main__':
    # # file path and make a list
    # imgs_path   = 'data/hw3_mycocodata_img_comp_zlib.h5'
    # masks_path  = 'data/hw3_mycocodata_mask_comp_zlib.h5'
    # labels_path = 'data/hw3_mycocodata_labels_comp_zlib.npy'
    # bboxes_path = 'data/hw3_mycocodata_bboxes_comp_zlib.npy'
    # paths = [imgs_path, masks_path, labels_path, bboxes_path]
    # # load the data into data.Dataset
    # dataset = BuildDataset(paths)
  
    # # build the dataloader
    # # set 20% of the dataset as the training data
    # full_size = len(dataset)
    # train_size = int(full_size * 0.8)
    # test_size = full_size - train_size
    # # random split the dataset into training and testset

    # train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    # rpn_net = RPNHead()
    # # push the randomized training data into the dataloader

    # # train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=0)
    # # test_loader  = DataLoader(test_dataset, batch_size=2, shuffle=False, num_workers=0)
    # batch_size = 6
    # train_build_loader = BuildDataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    # train_loader = train_build_loader.loader()
    # test_build_loader = BuildDataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    # test_loader = test_build_loader.loader()

    # for i,batch in enumerate(train_loader,0):
    #     images = batch['images']
    #     images = torch.stack(images[:])
    #     indexes= batch['index']
    #     boxes  = batch['bbox']
        

    #     # fig,ax=plt.subplots(1,1)
    #     # ax.imshow(images.permute(1,2,0))
    #     # rect=patches.Rectangle((boxes[0],boxes[1]),boxes[2]-boxes[0],boxes[3]-boxes[1],fill=False,color='red')
    #     # ax.add_patch(rect)

    #     anchors = rpn_net.create_anchors(rpn_net.anchors_param["ratio"], rpn_net.anchors_param["scale"], rpn_net.anchors_param["grid_size"], rpn_net.anchors_param["stride"])
    #     gt,ground_coord=rpn_net.create_batch_truth(boxes, indexes, (800,1088))
    #     # gt,ground_coord=rpn_net.create_ground_truth(torch.from_numpy(boxes[0]), indexes, rpn_net.anchors_param["grid_size"], anchors, (800,1088))


    #     # Flatten the ground truth and the anchors
    #     flatten_coord,flatten_gt,flatten_anchors=output_flattening(ground_coord.unsqueeze(0),gt.unsqueeze(0),rpn_net.get_anchors())
        
    #     # Decode the ground truth box to get the upper left and lower right corners of the ground truth boxes
    #     decoded_coord=output_decoding(flatten_coord,flatten_anchors)
        
    #     # Plot the image and the anchor boxes with the positive labels and their corresponding ground truth box
    #     images = transforms.functional.normalize(images,
    #                                                   [-0.485/0.229, -0.456/0.224, -0.406/0.225],
    #                                                   [1/0.229, 1/0.224, 1/0.225], inplace=False)
    #     if(i == 0):
    #         break
        
    # for i,each_img in enumerate(images):
    #     fig,ax=plt.subplots(1,1)
    #     ax.imshow(each_img.permute(1,2,0))
    #     # ax.imshow(images.permute(0,2,3,1))
        
    #     find_cor=(flatten_gt[i*3400:(i+1) *3400]==1).nonzero()
    #     find_neg=(flatten_gt[i*3400:(i+1) *3400]==-1).nonzero()
             
    #     for elem in find_cor:
    #         # ax.imshow(images[0].permute(0,2,3,1))
    #         coord=decoded_coord[i*3400:(i+1) *3400][elem,:].view(-1)
    #         anchor=flatten_anchors[i*3400:(i+1) *3400][elem,:].view(-1)

    #         col='r'
    #         rect=patches.Rectangle((coord[0],coord[1]),coord[2]-coord[0],coord[3]-coord[1],fill=False,color=col)
    #         ax.add_patch(rect)
    #         rect=patches.Rectangle((anchor[0]-anchor[2]/2,anchor[1]-anchor[3]/2),anchor[2],anchor[3],fill=False,color='b')
    #         ax.add_patch(rect)

    #     plt.show()
 
    #     if(i == 2):
    #         break
        

 