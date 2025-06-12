from __future__ import print_function, division
import argparse
import os
import torch
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch.nn.functional as F
from datetime import datetime
from torch.utils.data import DataLoader
from utils.metrics import get_metrics_forensics
from torch.utils.tensorboard import SummaryWriter
from utils.dataloader_train import forensics_datareader_train, forensics_transforms_train
from utils.dataloader_eval import forensics_datareader, forensics_transforms
from PIM import PIM_model

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('device: ', device)

def parse_args():
    parser = argparse.ArgumentParser(description='cq_forensics')
    parser.add_argument('--lr', default=6e-5, type=float)
    parser.add_argument('--random_seed', default=0, type=int)
    parser.add_argument('--lr_decay_rate', default=1.0, type=float)
    parser.add_argument('--threshold', default=0.5, type=float)
    parser.add_argument('--warm_start_epoch', default=0, type=int)
    parser.add_argument('--epochs', default=20, type=int)
    parser.add_argument('--img_size', default=512, type=int)
    parser.add_argument('--mask_size', default=128, type=int)
    parser.add_argument('--batch_size_train', default=28, type=int)
    parser.add_argument('--batch_size_test', default=28, type=int)
    parser.add_argument('--num_workers', default=12, type=int)
    parser.add_argument('--save_step', default=1600, type=int) 
    parser.add_argument('--disp_step', default=800, type=int)
    parser.add_argument('--weight_decay', default=0.00001, type=float) 
    parser.add_argument('--loss_weight_b', default=1.0, type=float)
    parser.add_argument('--loss_weight_c', default=0.001, type=float)
    parser.add_argument('--loss_weight_l1', default=0.1, type=float)
    parser.add_argument('--save_root', default='./Training_results/PIM/', type=str)
    parser.add_argument('--model_name', default="PIM_SwinT_pretrained_6e-5/", type=str)
    parser.add_argument('--pretrained', default=True, type=bool)
    parser.add_argument('--pretrained_path', default='./pretrained_models/moby_upernet_swin_tiny_patch4_window7_512x512.pth', type=str)
    parser.add_argument('--train_csv', default='./csv/CASIAv2_DA_PIDA_train.csv', type=str)
    parser.add_argument('--val_csv', default='./csv/test_fake/DEFACTO84k_val.csv', type=str)
    return parser.parse_args()


def fix_seed(seed):
    torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def Testing(model, dataloader, args, thre, imgnum):
    model.eval()
    batch_test_losses = []
    GT = np.zeros((imgnum, args.mask_size, args.mask_size), int)
    PRED = np.zeros((imgnum, args.mask_size, args.mask_size), float)
    length = len(dataloader)
    for num, data in enumerate(tqdm(dataloader)):
        with torch.no_grad():
            images = data['images'].to(device)
            masks = data['masks'].to(device)
            logits, _ , _ = model(images)
            test_loss = F.cross_entropy(logits, masks)
            preds = torch.nn.functional.softmax(logits, 1)
            batch_test_losses.append(test_loss.item())
            GT[num * args.batch_size_test: (num*args.batch_size_test+masks.size(0))] = masks.cpu().numpy()
            PRED[num * args.batch_size_test: (num*args.batch_size_test+preds.size(0))] = preds[:, 1, :, :].cpu().numpy()
    f1, mcc, iou, acc = get_metrics_forensics(GT, PRED, thre)
    avg_test_loss = round(sum(batch_test_losses) / (len(batch_test_losses)), 5)
    return avg_test_loss, f1, mcc, iou, acc

def train(args, model):
    train_dataset = forensics_datareader_train(csv_file=args.train_csv, transform=forensics_transforms_train(args.img_size, args.mask_size))
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size_train, shuffle=True, num_workers=args.num_workers, drop_last=False)

    val_dataset = forensics_datareader(csv_file=args.val_csv, transform=forensics_transforms(args.img_size, args.mask_size))
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size_test, shuffle=True, num_workers=args.num_workers, drop_last=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    l1_norm = torch.nn.L1Loss()

    # result folder
    res_folder_name = args.save_root + args.model_name
    if not os.path.exists(res_folder_name):
        os.makedirs(res_folder_name)
        os.mkdir(res_folder_name + '/ckpt/')
    else:
        print("WARNING: RESULT PATH ALREADY EXISTED -> " + res_folder_name)
    print('find models here: ', res_folder_name)
    writer = SummaryWriter(res_folder_name)
    file = open(res_folder_name + "/training_log.csv", 'a+')

    # training
    Best_F1 = 0
    steps_per_epoch = len(train_dataloader)
    for epoch in range(args.warm_start_epoch, args.epochs):
        step_loss = np.zeros(steps_per_epoch, dtype=float)
        step_mask_loss = np.zeros(steps_per_epoch, dtype=float)
        step_boundary_loss = np.zeros(steps_per_epoch, dtype=float)
        step_compactness_loss = np.zeros(steps_per_epoch, dtype=float)
        step_recon_loss = np.zeros(steps_per_epoch, dtype=float)
        for step, data in enumerate(tqdm(train_dataloader)):
            model.train()
            optimizer.zero_grad()
            images = data['images'].to(device)
            masks = data['masks'].to(device)
            boundaries = data['boundaries'].to(device)
            img_labels = data['img_labels'].to(device)

            pred_masks, pred_boundaries, recon_imgs = model(images)
            mask_loss = F.cross_entropy(pred_masks, masks)
            boundary_loss = args.loss_weight_b * F.cross_entropy(pred_boundaries, boundaries)
            recon_loss = args.loss_weight_l1 * l1_norm(recon_imgs, img_labels)
            preds_m = torch.nn.functional.softmax(pred_masks, 1)[:, 1, :, :]
            preds_b = torch.nn.functional.softmax(pred_boundaries, 1)[:, 1, :, :]
            
            length = torch.sum(preds_b, dim=(1,2))
            area = torch.sum(preds_m, dim=(1,2))
            compactness_loss = args.loss_weight_c*torch.mean(length**2/(area*4*3.1415926))
            loss = mask_loss + boundary_loss + compactness_loss + recon_loss

            step_loss[step] = loss
            step_mask_loss[step] = mask_loss
            step_boundary_loss[step] = boundary_loss
            step_recon_loss[step] = recon_loss
            step_compactness_loss[step] = compactness_loss

            loss.backward()
            optimizer.step()
            Global_step = epoch * steps_per_epoch + (step + 1)

            if Global_step % args.disp_step == 0:
                avg_loss = np.mean(step_loss[(step + 1) - args.disp_step: (step + 1)])
                avg_mask_loss = np.mean(step_mask_loss[(step + 1) - args.disp_step: (step + 1)])
                avg_boundary_loss = np.mean(step_boundary_loss[(step + 1) - args.disp_step: (step + 1)])
                avg_recon_loss = np.mean(step_recon_loss[(step + 1) - args.disp_step: (step + 1)])
                avg_compactness_loss = np.mean(step_compactness_loss[(step + 1) - args.disp_step: (step + 1)])
                now_time = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
                step_log_msg = '[%s] Epoch: %d/%d | Global_step: %d |average loss: %f |average mask loss: %f |average boundary loss: %f |average compactness loss: %f |average recon loss: %f' % (
                    now_time, epoch + 1, args.epochs, Global_step, avg_loss, avg_mask_loss, avg_boundary_loss, avg_compactness_loss, avg_recon_loss)
                writer.add_scalar('Loss/train', avg_loss, Global_step)
                print('\n', step_log_msg)

            if Global_step % args.save_step == 0:
                now_time = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
                avg_train_loss = np.mean(step_loss[(step + 1) - args.disp_step: (step + 1)])
                log_msg = '[%s] Epoch: %d/%d | average epoch loss: %f' % (now_time, epoch + 1, args.epochs, avg_train_loss)
                print('\n', log_msg)
                file.write(log_msg)
                file.write('\n')

                # validation
                print('Validation...')
                avg_val_loss, val_f1, val_mcc, val_iou, val_acc = Testing(model, val_dataloader, args, args.threshold, len(pd.read_csv(args.val_csv, header=None)))
                val_msg = '[%s] Epoch: %d/%d | Global_step: %d | average val loss: %f | Val_F1: %f| Val_ACC: %f| Val_MCC: %f| Val_IOU: %f' % (
                now_time, epoch + 1, args.epochs, Global_step, avg_val_loss, val_f1, val_acc, val_mcc, val_iou)
                print('\n', val_msg)
                file.write(val_msg)
                file.write('\n')

                #save model
                if val_f1 > Best_F1:
                    Best_F1 = val_f1
                    torch.save(model.state_dict(), res_folder_name + '/ckpt/best.pth')
                    cur_learning_rate = [param_group['lr'] for param_group in optimizer.param_groups]
                    print('Saved model. lr %f' % cur_learning_rate[0])
                    file.write('Saved model. lr %f' % cur_learning_rate[0])
                    file.write('\n')
    file.close()


def main(args):
    model = PIM_model(pretrained = True, model_path = args.pretrained_path)
    model = model.to(device)
    model = torch.nn.DataParallel(model,device_ids=[0,1])
    print(model)
    print("number of model parameters:", sum([np.prod(p.size()) for p in model.parameters()]))
    train(args, model)


if __name__ == '__main__':
    args = parse_args()
    if args.random_seed is not None:
        fix_seed(args.random_seed)
    print(args)
    main(args)
