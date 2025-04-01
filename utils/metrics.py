import numpy as np
from sklearn.metrics import roc_curve,  auc

def auc_measure(y_true, y_pred):
    y_true = y_true.astype(int)
    fpr, tpr, _ = roc_curve(y_true, y_pred, pos_label=1)
    AUC = auc(fpr, tpr)
    return AUC

def get_metrics_forensics(y_true, y_pred, thre):
    sum_f1 = 0.0
    sum_mcc = 0.0
    sum_iou = 0.0
    sum_acc = 0.0
    for idx in range(y_true.shape[0]):
        y_pred_slice = y_pred[idx,:,:]
        y_true_slice = y_true[idx,:,:]

        gt_bool_slice = (y_true_slice>thre)
        pred_bool_slice = (y_pred_slice>thre)
        tp,fp,tn,fn = get_tn_tp_fn_fp(thre, y_pred_slice, gt_bool_slice)
        
        acc = (tp+tn)/(tp+fn+tn+fp)
        f1 = 2*tp/(2*tp+fp+fn+1e-6)
        mcc = (tp*tn-fp*fn)/(np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))+1e-6)
        iou = tp/(fp+tp+fn+1e-6)
        
        if np.max(pred_bool_slice) == np.max(gt_bool_slice) and np.max(pred_bool_slice) == False:
            f1, iou, acc, mcc = 1.0, 1.0, 1.0, 1.0
            
        sum_f1 += f1
        sum_mcc += mcc
        sum_iou += iou
        sum_acc += acc
    
    avg_f1 = sum_f1/y_true.shape[0]
    avg_mcc = sum_mcc/y_true.shape[0]
    avg_iou = sum_iou/y_true.shape[0]
    avg_acc = sum_acc/y_true.shape[0]
    return avg_f1, avg_mcc, avg_iou, avg_acc
 
def get_metrics_forensics_new(y_true, y_pred, thre):
    gt_bool_slice = (y_true>thre)
    pred_bool_slice = (y_pred>thre)
    tp,fp,tn,fn = get_tn_tp_fn_fp(thre, y_pred, gt_bool_slice)
    auc = auc_measure(gt_bool_slice.astype(float).flatten(), y_pred.flatten())
    f1 = 2*tp/(2*tp+fp+fn+1e-6)
    mcc = (tp*tn-fp*fn)/(np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))+1e-6)
    iou = tp/(fp+tp+fn+1e-6)
    if np.max(pred_bool_slice) == np.max(gt_bool_slice) and np.max(pred_bool_slice) == False:
        f1, iou, auc, mcc = 1.0, 1.0, 1.0, 1.0
    elif np.min(pred_bool_slice) == np.min(gt_bool_slice) and np.min(pred_bool_slice) == True:
        f1, iou, auc, mcc = 1.0, 1.0, 1.0, 1.0
    return f1,mcc,iou,auc
    
def get_tn_tp_fn_fp(threshold, dist, actual_issame):
    predict_issame = np.less(1-dist, 1-threshold)
    tp = np.sum(np.logical_and(predict_issame, actual_issame))
    fp = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
    tn = np.sum(np.logical_and(np.logical_not(predict_issame), np.logical_not(actual_issame)))
    fn = np.sum(np.logical_and(np.logical_not(predict_issame), actual_issame))
    return float(tp),float(fp),float(tn),float(fn)


