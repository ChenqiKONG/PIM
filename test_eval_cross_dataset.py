from __future__ import print_function, division
import argparse
import os
import cv2
import torch
import numpy as np
from torch.utils.data import DataLoader
from utils.dataloader_eval import forensics_datareader, forensics_transforms
from PIM import PIM_model
from tqdm import tqdm
import torch.nn.functional as F
import pandas as pd
import shutil
from utils.metrics import get_metrics_forensics
from sklearn.metrics import roc_curve,  auc

device = torch.device('cuda:0' if torch.cuda.is_available()  else 'cpu')
print('device: ', device)

def parse_args():
    parser = argparse.ArgumentParser(description='cq_test')
    parser.add_argument('--threshold', default=0.5, type=float)
    parser.add_argument('--batch_size_test', default=19, type=int)
    parser.add_argument('--num_workers', default=12, type=int)
    parser.add_argument('--img_size', default=512, type=int)
    parser.add_argument('--mask_size', default=128, type=int)
    parser.add_argument('--model_path', default="./model/ckpt.pth", type=str)
    return parser.parse_args()

def matthews_corrcoef(y_true, y_pred):
    tn, tp, fn, fp = get_tn_tp_fn_fp(y_true, y_pred)
    mcc = (tp*tn-fp*fn)/np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)+1e-8)
    if np.isnan(mcc):
        return 0.
    else:
        return mcc

def get_tn_tp_fn_fp(y_true, y_pred):
    tn = np.sum(np.logical_and(np.logical_not(y_true), np.logical_not(y_pred))).astype(np.float64)
    tp = np.sum(np.logical_and(y_true, y_pred )).astype(np.float64)
    fn = np.sum(np.logical_and(y_true, np.logical_not(y_pred))).astype(np.float64)
    fp = np.sum(np.logical_and(np.logical_not(y_true), y_pred )).astype(np.float64)
    return tn, tp, fn, fp

def calculate_pixel_f1(pd, gt):
    if np.max(pd) == np.max(gt) and np.max(pd) == 0:
        f1, iou = 1.0, 1.0
        return f1, 0.0, 0.0
    seg_inv, gt_inv = np.logical_not(pd), np.logical_not(gt)
    true_pos = float(np.logical_and(pd, gt).sum())
    false_pos = np.logical_and(pd, gt_inv).sum()
    false_neg = np.logical_and(seg_inv, gt).sum()
    f1 = 2 * true_pos / (2 * true_pos + false_pos + false_neg + 1e-6)
    precision = true_pos / (true_pos + false_pos + 1e-6)
    recall = true_pos / (true_pos + false_neg + 1e-6)
    return f1, precision, recall

def iou_measure(y_true, y_pred):
    _, tp, fn, fp = get_tn_tp_fn_fp(y_true, y_pred)
    return tp/(tp+fp+fn)

def auc_measure(y_true, y_pred):
    y_true = y_true.astype(int)
    fpr, tpr, _ = roc_curve(y_true, y_pred, pos_label=1)
    #eer = brentq(lambda x : 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    AUC = auc(fpr, tpr)
    return AUC

def evaluate_metric(gt_dir, pred_dir, threshold):
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

def save_results(save_dir_gt, save_dir_pred, mask_paths, preds):
    BS = preds.shape[0]
    for i in range(BS):
        mask_path = mask_paths[i]
        mask = cv2.imread(mask_path)
        ori_size = mask.shape
        save_gt_path = save_dir_gt + mask_path.split('/')[-1]
        shutil.copy(mask_path, save_gt_path)
        save_pred_path = save_dir_pred + mask_path.split('/')[-1]
        pred_result = np.squeeze(preds[i])
        fake_seg = 255.0 * pred_result
        fake_seg = fake_seg.astype(np.uint8)
        fake_seg = cv2.resize(fake_seg, (ori_size[1], ori_size[0]))
        cv2.imwrite(save_pred_path, fake_seg.astype(np.uint8))

def Testing(model, dataloader, args, thre, imgnum, save_dir_gt, save_dir_pred):
    model.eval()
    batch_test_losses = []
    GT = np.zeros((imgnum, args.mask_size, args.mask_size), int)
    PRED = np.zeros((imgnum, args.mask_size, args.mask_size), float)
    for num, data in enumerate(tqdm(dataloader)):
        with torch.no_grad():
            images = data['images'].to(device)
            masks = data['masks'].to(device)
            mask_paths = data['mask_paths']
            logits, _, _  = model(images)
            test_loss = F.cross_entropy(logits, masks)
            preds = torch.nn.functional.softmax(logits, 1)
            batch_test_losses.append(test_loss.item())
            GT[num * args.batch_size_test: (num*args.batch_size_test+masks.size(0))] = masks.cpu().numpy()
            PRED[num * args.batch_size_test: (num*args.batch_size_test+preds.size(0))] = preds[:, 1, :, :].cpu().numpy()
            save_results(save_dir_gt, save_dir_pred, mask_paths, preds[:, 1, :, :].cpu().numpy())
    f1, mcc, iou, acc = get_metrics_forensics(GT, PRED, thre)
    avg_test_loss = round(sum(batch_test_losses) / (len(batch_test_losses)), 5)
    return avg_test_loss, f1, mcc, iou, acc


def test(args, model):
    save_dir_list = ["./pred_results/PIM/NIST16/", "./pred_results/PIM/Columbia/", "./pred_results/PIM/CASIAv1+/", 
                    "./pred_results/PIM/COVER/", "./pred_results/PIM/DEFACTO12k_test/", "./pred_results/PIM/IMD2020_jpg/",
                    "./pred_results/PIM/Carvalho/", "./pred_results/PIM/IFC/", "./pred_results/PIM/In-the-wild/",
                    "./pred_results/PIM/Korus/", "./pred_results/PIM/WildWeb/"]
    test_file_list = ["./csv/test_fake/NIST16.csv", "./csv/test_fake/Columbia.csv", "./csv/test_fake/CASIAv1+.csv",
                    "./csv/test_fake/COVER.csv", "./csv/test_fake/DEFACTO12k_test.csv", "./csv/test_fake/IMD2020_jpg.csv",
                    "./csv/test_fake/Carvalho.csv", "./csv/test_fake/IFC.csv", "./csv/test_fake/In-the-wild.csv",
                    "./csv/test_fake/Korus.csv", "./csv/test_fake/WildWeb.csv"]
    
    f1_list = []
    mcc_list = []
    iou_list = []
    acc_list = []
    resize_f1_list = []
    resize_mcc_list = []
    resize_iou_list = []
    resize_auc_list = []
    for i in range(len(save_dir_list)):
        save_dir = save_dir_list[i]
        test_file = test_file_list[i]

        save_dir_gt = save_dir+'gt/'
        if not os.path.exists(save_dir_gt):
            os.makedirs(save_dir_gt)

        save_dir_pred = save_dir+'pred/'
        if not os.path.exists(save_dir_pred):
            os.makedirs(save_dir_pred)

        test_dataset = forensics_datareader(csv_file=test_file, transform=forensics_transforms(args.img_size, args.mask_size))
        test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size_test, shuffle=False, num_workers=args.num_workers, drop_last=False)
        avg_test_loss, f1, mcc, iou, acc = Testing(model, test_dataloader, args, args.threshold, len(pd.read_csv(test_file, header=None)), save_dir_gt, save_dir_pred)
        
        resize_f1, resize_mcc, resize_iou, resize_auc = evaluate_metric(save_dir_gt, save_dir_pred, args.threshold)
        
        test_msg = 'index: %d| Average test loss: %f | F1: %f| ACC: %f| MCC: %f| IOU: %f| Resize_F1: %f| Resize_MCC: %f| Resize_IOU: %f| Resize_AUC: %f' % (i, avg_test_loss, f1, acc, mcc, iou, resize_f1, resize_mcc, resize_iou, resize_auc)
        print(test_msg)
        f1_list.append(f1)
        mcc_list.append(mcc)
        iou_list.append(iou)
        acc_list.append(acc)
        resize_f1_list.append(resize_f1)
        resize_mcc_list.append(resize_mcc)
        resize_iou_list.append(resize_iou)
        resize_auc_list.append(resize_auc)

    avg_f1 = np.mean(f1_list)
    avg_mcc = np.mean(mcc_list)
    avg_iou = np.mean(iou_list)
    avg_acc = np.mean(acc_list)
    avg_resize_f1 = np.mean(resize_f1_list)
    avg_resize_mcc = np.mean(resize_mcc_list)
    avg_resize_iou = np.mean(resize_iou_list)
    avg_resize_auc = np.mean(resize_auc_list)
    test_msg = 'Average F1: %f| Average ACC: %f| Average MCC: %f| Average IOU: %f| Average Resize_F1: %f| Average Resize_MCC: %f| Average Resize_IOU: %f| Average Resize_AUC: %f' % (avg_f1, avg_acc, avg_mcc, avg_iou, avg_resize_f1, avg_resize_mcc, avg_resize_iou, avg_resize_auc)
    print('\n', test_msg)
    print('Resize F1:', resize_f1_list)
    print('Resize MCC:', resize_mcc_list)
    print('Resize IoU:', resize_iou_list)
    print('Resize AUC:', resize_auc_list)


def main(args):
    model = PIM_model(pretrained = False, model_path = None)
    model = torch.nn.DataParallel(model,device_ids=[0,1])
    model.load_state_dict(torch.load(args.model_path, map_location='cuda:0'))
    model = model.to(device)
    test(args, model)

if __name__ == '__main__':
    args = parse_args()
    main(args)

