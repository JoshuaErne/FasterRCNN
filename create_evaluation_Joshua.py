import torchvision
import torch
import numpy as np
from BoxHead import *
from utils import *
from pretrained_models import *

if __name__ == '__main__':

    # Put the path were you save the given pretrained model
    pretrained_path='/home/josh/Desktop/CIS680/HW4/FasterRCNN/HW4_PartB_Code_Template/checkpoint680.pth'
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    backbone, rpn = pretrained_models_680(pretrained_path)

    # we will need the ImageList from torchvision
    from torchvision.models.detection.image_list import ImageList

    # Put the path were the given hold_out_images.npz file is save and load the images
    hold_images_path='/home/josh/Desktop/CIS680/HW4/FasterRCNN/HW4_PartB_Code_Template/hold_out_images.npz'
    test_images=np.load(hold_images_path,allow_pickle=True)['input_images']

    # Load your model here. If you use different parameters for the initialization you can change the following code
    # accordingly
    boxHead=BoxHead()
    boxHead=boxHead.to(device)
    boxHead.eval()

    # Put the path were you have your save network
    train_model_path='/home/josh/Desktop/CIS680/HW4/FasterRCNN/HW4_PartB_Code_Template/model_1.pth'
    checkpoint = torch.load(train_model_path)
    # reload models
    # boxHead.load_state_dict(checkpoint['box_head_state_dict'])
    boxHead.load_state_dict(checkpoint)
    
    keep_topK=200

    cpu_boxes = []
    cpu_scores = []
    cpu_labels = []

    for i, numpy_image in enumerate(test_images, 0):
        # images = numpy_image['images'].to(device)
        images = torch.from_numpy(numpy_image).to(device)
        with torch.no_grad():
            # Take the features from the backbone
            backout = backbone(images)

            # The RPN implementation takes as first argument the following image list
            im_lis = ImageList(images, [(800, 1088)]*images.shape[0])
            # Then we pass the image list and the backbone output through the rpn
            rpnout = rpn(im_lis, backout)

            #The final output is
            # A list of proposal tensors: list:len(bz){(keep_topK,4)}
            proposals=[proposal[0:keep_topK,:] for proposal in rpnout[0]]
            # A list of features produces by the backbone's FPN levels: list:len(FPN){(bz,256,H_feat,W_feat)}
            fpn_feat_list= list(backout.values())


            feature_vectors                 = boxHead.MultiScaleRoiAlign(fpn_feat_list,proposals)
            class_logits,box_pred           = boxHead.forward(feature_vectors, evaluate = True)
            # new_labels,regressor_target     = box_head.create_ground_truth(proposals,labels, bboxes)
            # Do whaterver post processing you find performs best
            boxes,scores,labels,xxxx,yyyy             = boxHead.postprocess_detections(class_logits,box_pred,proposals,conf_thresh=0.5, keep_num_preNMS=200, keep_num_postNMS=3)
            # labels,scores,boxes       = boxHead.postprocess_detections_map_scores(class_logits,box_pred,proposals,conf_thresh=0.5, keep_num_preNMS=200, keep_num_postNMS=3)
            
            fig,ax=plt.subplots(1,1)
            ax.imshow(images[0].permute(1,2,0).cpu().detach().numpy())
            for box, score, label in zip(boxes,scores,labels):
                if box is None:
                    cpu_boxes.append(None)
                    cpu_scores.append(None)
                    cpu_labels.append(None)
                else:
                    cpu_boxes.append(box.to('cpu').detach().numpy())
                    cpu_scores.append(score.to('cpu').detach().numpy())
                    cpu_labels.append(label.to('cpu').detach().numpy())
                    # cpu_boxes.append(box)
                    # cpu_scores.append(score)
                    # cpu_labels.append(label)

            for kk, box_label in enumerate(zip(cpu_boxes[i], cpu_labels[i])):
                if int(box_label[1]) == 1:
                    color = 'r'
                if int(box_label[1]) == 2:
                    color = 'g'
                if int(box_label[1]) == 3:
                    color = 'b'
                box=box_label[0]
                rect=patches.Rectangle((box[0],box[1]),box[2]-box[0],box[3]-box[1],fill=False,color=color)
                ax.add_patch(rect)
            plt.show()

            print('-----')
    np.savez('predictions.npz', predictions={'boxes': cpu_boxes, 'scores': cpu_scores,'labels': cpu_labels})
