from __future__ import print_function, division
import argparse
import os
import cv2
import torch
from skimage import transform
import numpy as np
from PIM import PIM_model
from tqdm import tqdm

device = torch.device('cuda:0' if torch.cuda.is_available()  else 'cpu')
print('device: ', device)

def parse_args():
    parser = argparse.ArgumentParser(description='cq_test')
    parser.add_argument('--threshold', default=0.5, type=float)
    parser.add_argument('--img_dir', default='./img_dir', type=str) 
    parser.add_argument('--save_dir', default='./save_dir', type=str)
    parser.add_argument('--num_workers', default=12, type=int)
    parser.add_argument('--img_size', default=512, type=int)
    parser.add_argument('--mask_size', default=128, type=int)
    parser.add_argument('--model_path', default="./model/ckpt.pth", type=str)
    return parser.parse_args()
    f1s = []
    mccs = []
    ious = []
    aucs = []
    for pred_file in os.listdir(pred_dir):
        pred_path = os.path.join(pred_dir, pred_file)
        gt_path = os.path.join(gt_dir, pred_file)
        #gt_path = gt_dir+pred_file.split('.')[0]+'_gt.png'
        if os.path.exists(gt_path):
            pred = cv2.imread(pred_path, 0) / 255.0
            gt = cv2.imread(gt_path, 0) / 255.0
            if pred.shape[0] == gt.shape[0] and pred.shape[1] == gt.shape[1]:
                gt = (gt > threshold).astype(float)
                auc = auc_measure(gt.flatten(), pred.flatten())
                pred = (pred > threshold).astype(float)
                f1, _, _ = calculate_pixel_f1(pred.flatten(), gt.flatten())
                mcc = matthews_corrcoef(gt.flatten(), pred.flatten())
                iou = iou_measure(gt.flatten(), pred.flatten())
                f1s.append(f1)
                mccs.append(mcc)
                ious.append(iou)
                aucs.append(auc)
        else:
            print(gt_path)
    return np.mean(f1s), np.mean(mccs), np.mean(ious), np.mean(aucs)


def Testing(model, args, input_img, save_path, ori_size):
    model.eval()
    with torch.no_grad():
        images = input_img.to(device)
        logits, _, _  = model(images)
        preds = torch.nn.functional.softmax(logits, 1).cpu().numpy()
        pred_result = np.squeeze(preds[:, 1, :, :])
        fake_seg = 255.0 * pred_result
        fake_seg = fake_seg.astype(np.uint8)
        fake_seg = cv2.resize(fake_seg, (ori_size[1], ori_size[0]))
        cv2.imwrite(save_path, fake_seg.astype(np.uint8))
    

def main(args):
    #Load model
    model = PIM_model(pretrained = False, model_path = None)
    model = torch.nn.DataParallel(model,device_ids=[0,1])
    model.load_state_dict(torch.load(args.model_path, map_location='cuda:0'))
    model = model.to(device)

    mean=np.array([0.485, 0.456, 0.406])
    std=np.array([0.229, 0.224, 0.225])
    #Predict forgery masks one by one
    for img in tqdm(os.listdir(args.img_dir)):
        img_path = os.path.join(args.img_dir, img)
        save_path = os.path.join(args.save_dir, img)
        input_img = cv2.imread(img_path, 1)
        #Image preprocessing
        ori_size = input_img.shape[:2]
        input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
        input_img = transform.resize(input_img, (512, 512))
        input_img = (input_img - mean) / std
        input_img = input_img.transpose(2, 0, 1)
        input_img = torch.from_numpy(input_img.copy()).float()
        input_img = input_img.unsqueeze(0)
        #Predict and save the mask
        Testing(model, args, input_img, save_path, ori_size)

if __name__ == '__main__':
    args = parse_args()
    main(args)

