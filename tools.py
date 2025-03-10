# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Ke Sun (sunk@mail.ustc.edu.cn)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging
import time
from pathlib import Path
import termcolor

from sklearn import metrics
import matplotlib
matplotlib.use('Agg')  # Set the backend before importing pyplot
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix
import scipy.stats

import numpy as np
import cv2
import random
import termcolor

import torch
import torch.nn as nn

from pdb import set_trace as st
import seaborn as sn
import pandas as pd
import re
import seaborn as sns



# convert to colored strings
def toRed(content): return termcolor.colored(content,"red",attrs=["bold"])
def toGreen(content): return termcolor.colored(content,"green",attrs=["bold"])
def toBlue(content): return termcolor.colored(content,"blue",attrs=["bold"])
def toCyan(content): return termcolor.colored(content,"cyan",attrs=["bold"])
def toYellow(content): return termcolor.colored(content,"yellow",attrs=["bold"])
def toMagenta(content): return termcolor.colored(content,"magenta",attrs=["bold"])


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.initialized = False
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None

    def initialize(self, val, weight):
        self.val = val
        self.avg = val
        self.sum = val * weight
        self.count = weight
        self.initialized = True

    def update(self, val, weight=1):
        if not self.initialized:
            self.initialize(val, weight)
        else:
            self.add(val, weight)

    def add(self, val, weight):
        self.val = val
        self.sum += val * weight
        self.count += weight
        self.avg = self.sum / self.count

    def value(self):
        return self.val

    def average(self):
        return self.avg


def get_confusion_matrix(label, pred, size, num_class, ignore=-1):
    """
    Calcute the confusion matrix by given label and pred
    """
    output = pred.cpu().numpy().transpose(0, 2, 3, 1)
    seg_pred = np.asarray(np.argmax(output, axis=3), dtype=np.uint8)
    seg_gt = np.asarray(
    label.cpu().numpy()[:, :size[-2], :size[-1]], dtype=np.int)

    ignore_index = seg_gt != ignore
    seg_gt = seg_gt[ignore_index]
    seg_pred = seg_pred[ignore_index]

    index = (seg_gt * num_class + seg_pred).astype('int32')
    label_count = np.bincount(index)
    confusion_matrix = np.zeros((num_class, num_class))

    for i_label in range(num_class):
        for i_pred in range(num_class):
            cur_index = i_label * num_class + i_pred
            if cur_index < len(label_count):
                confusion_matrix[i_label,
                                 i_pred] = label_count[cur_index]
    return confusion_matrix


def mymetrics_without_avg(gts,probs,preds,avg=None):
    accuracy = metrics.accuracy_score(gts,preds)
    precision=metrics.precision_score(gts,preds, average=avg)
    recall=metrics.recall_score(gts,preds,average=avg)
    f1=metrics.f1_score(gts,preds,average=avg)
    specificity=[]
    con_mat = confusion_matrix(gts, preds)
    for i in range(probs.shape[1]):
        number = np.sum(con_mat[:, :])
        tp = con_mat[i][i]
        fn = np.sum(con_mat[i, :]) - tp
        fp = np.sum(con_mat[:, i]) - tp
        tn = number - tp - fn - fp
        spe1 = tn / (tn + fp)
        specificity.append(spe1)
    
    try:
        auc=metrics.roc_auc_score(gts,probs,average=avg,multi_class='ovr') # one-over-rest
        # only support skicit-learn 1.2 version under python3.9
    except:
        auc=my_roc_auc_score(gts,probs)

    return accuracy, specificity, recall, f1, auc


def calculate_based_on_cm(con_mat):

    fpr,tpr,spec=[],[],[]

    for i in range(con_mat.shape[1]):
        number = np.sum(con_mat[:, :])
        tp = con_mat[i][i]
        fn = np.sum(con_mat[i, :]) - tp
        fp = np.sum(con_mat[:, i]) - tp
        tn = number - tp - fn - fp
        spe1 = tn / (tn + fp)
        spec.append(spe1)

        fpr1 = fp/(fp+tn)
        fpr.append(fpr1)

        tpr1 = tp/(tp+fn)
        tpr.append(tpr1)
    
    return fpr, tpr, spec



def my_roc_auc_score(y_true,y_score):
    roc_labels = []
    for y_i in np.unique(y_true):
        roc_labels.append(
            metrics.roc_auc_score(
                np.vectorize(
                    {y_i: 1, **{y_k: 0 for y_k in np.unique(y_true) if y_k != y_i}}.get
                )(y_true),
                y_score[:, y_i],
            )
        )

    return roc_labels


# support multi classes only
def mymetrics_for_multiclasses(gts,probs,preds,avg='macro'):

    accuracy = metrics.accuracy_score(gts,preds)
    precision=metrics.precision_score(gts,preds, average=avg)
    recall=metrics.recall_score(gts,preds,average=avg)
    f1=metrics.f1_score(gts,preds,average=avg)

    spe=[]
    con_mat = confusion_matrix(gts, preds)
    for i in range(probs.shape[1]):
        number = np.sum(con_mat[:, :])
        tp = con_mat[i][i]
        fn = np.sum(con_mat[i, :]) - tp
        fp = np.sum(con_mat[:, i]) - tp
        tn = number - tp - fn - fp
        spe1 = tn / (tn + fp)
        spe.append(spe1)
    specificity = np.mean(np.array(spe))

    try:
        # average must be one of ('macro', 'weighted') for multiclass problems
        auc=metrics.roc_auc_score(gts,probs,average=avg,multi_class='ovr') # one-over-rest
    except:
        auc=0
    return accuracy, specificity, recall, f1, auc

# support binary or multi classes
def mymetrics(gts,probs,preds,avg='macro'):
    # differiate binary class and multi classes
    num_class=probs.shape[-1]
    if num_class>2:
        accuracy = metrics.accuracy_score(gts,preds)
        precision=metrics.precision_score(gts,preds, average=avg)
        recall=metrics.recall_score(gts,preds,average=avg)
        f1=metrics.f1_score(gts,preds,average=avg)

        spe=[]
        con_mat = confusion_matrix(gts, preds)
        for i in range(probs.shape[1]):
            number = np.sum(con_mat[:, :])
            tp = con_mat[i][i]
            fn = np.sum(con_mat[i, :]) - tp
            fp = np.sum(con_mat[:, i]) - tp
            tn = number - tp - fn - fp
            spe1 = tn / (tn + fp)
            spe.append(spe1)
        specificity = np.mean(np.array(spe))
        try:
            # average must be one of ('macro', 'weighted') for multiclass problems
            auc=metrics.roc_auc_score(gts,probs,average='macro',multi_class='ovr') # one-over-rest
        except:
            auc=0
    if num_class==2:
        accuracy = metrics.accuracy_score(gts,preds)
        precision=metrics.precision_score(gts,preds, average=None)[-1]
        recall=metrics.recall_score(gts,preds,average=None)[-1]
        f1=metrics.f1_score(gts,preds,average=None)[-1]


        spe=[]
        con_mat = confusion_matrix(gts, preds)
        for i in range(probs.shape[1]):
            number = np.sum(con_mat[:, :])
            tp = con_mat[i][i]
            fn = np.sum(con_mat[i, :]) - tp
            fp = np.sum(con_mat[:, i]) - tp
            tn = number - tp - fn - fp
            spe1 = tn / (tn + fp)
            spe.append(spe1)
        specificity = np.array(spe)[-1]
        # check binary classification
        probs=probs[:,-1]
        try:
            # average must be one of ('macro', 'weighted') for multiclass problems
            auc=metrics.roc_auc_score(gts,probs) # one-over-rest
        except:
            auc=0

    return accuracy, specificity, recall, f1, auc




def bootstrap_AUC_CIs(probs, labels):
    probs = np.array(probs)
    labels = np.array(labels)
    N_slide = len(probs)
    index_list = np.arange(0, N_slide)
    AUC_list = []
    i = 0
    while i < 1000:
        sampled_indices = random.choices(index_list, k=N_slide)
        sampled_probs = probs[sampled_indices]
        sampled_labels = labels[sampled_indices]

        if np.unique(sampled_labels).size == 1:
            continue

        auc_bs = metrics.roc_auc_score(sampled_labels, sampled_probs)
        AUC_list.append(auc_bs)
        i += 1

    assert len(AUC_list) == 1000
    AUC_list = np.array(AUC_list)
    auc_avg = np.mean(AUC_list)
    auc_CIs = [np.percentile(AUC_list, 2.5), np.percentile(AUC_list, 97.5)]
    return auc_avg, auc_CIs


def bootstrap_CIs(labels,preds, func):
    preds = np.array(preds)
    labels = np.array(labels)
    N_slide = len(preds)
    index_list = np.arange(0, N_slide)
    result_list = []
    i = 0
    while i < 1000:
        sampled_indices = random.choices(index_list, k=N_slide)
        sampled_preds = preds[sampled_indices]
        sampled_labels = labels[sampled_indices]

        if np.unique(sampled_labels).size == 1:
            continue

        result_bs = func(sampled_labels, sampled_preds)
        result_list.append(result_bs)
        i += 1

    assert len(result_list) == 1000
    result_list = np.array(result_list)
    result_avg = np.mean(result_list)
    result_CIs = [np.percentile(result_list, 2.5), np.percentile(result_list, 97.5)]
    result_std=result_avg-result_CIs[0]
    return result_avg, result_std



def readlines_from_txt(XY):
    all_XY=[]
    for item in XY:
        f=open(item,'r')
        lines_=f.readlines()
        lines=[line.strip() for line in lines_]
        all_XY+=lines
        f.close()
    return all_XY


def produce_heatmap(img,cmap,ratio=0.5,threshold=0,probs=1):
    try:
        heatmap = (cmap - cmap.min())/(cmap.max() - cmap.min())*probs
    except:
        heatmap = heatmap*0.0
    
    heatmap[heatmap<threshold] = 0
    heatmap = cv2.resize(np.float32(heatmap),(img.shape[1],img.shape[0]))
    heatmap = cv2.applyColorMap(np.uint8(255*heatmap), cv2.COLORMAP_JET)
    # heatmap = cv2.applyColorMap(np.uint8(255*heatmap), cv2.COLORMAP_BONE)
    img2heat=heatmap*ratio+img*(1-ratio)
    img2heat=img2heat.astype(np.uint8)
    return img2heat


def produce_transformer_attention(x,get_mask=None):

    att_mat = x.squeeze()

    # Average the attention weights across all heads.
    att_mat = torch.mean(att_mat, dim=1)

    # To account for residual connections, we add an identity matrix to the
    # attention matrix and re-normalize the weights.
    # st()
    residual_att = torch.eye(att_mat.size(1)).cuda()
    aug_att_mat = att_mat + residual_att
    aug_att_mat = aug_att_mat / aug_att_mat.sum(dim=-1).unsqueeze(-1)

    # Recursively multiply the weight matrices
    joint_attentions = torch.zeros(aug_att_mat.size()).cuda()
    joint_attentions[0] = aug_att_mat[0]

    for n in range(1, aug_att_mat.size(0)):
        joint_attentions[n] = torch.matmul(aug_att_mat[n], joint_attentions[n-1])

    v = joint_attentions[-1]
    mask = v[0, 1:]
    grid_size = int(np.sqrt(aug_att_mat.size(-1)))
    # mask = v[0, 1:].reshape(grid_size, grid_size).detach().cpu().numpy()
    # if get_mask:
    #     result = cv2.resize(mask / mask.max(), img.size)
    # else:        
    #     mask = cv2.resize(mask / mask.max(), img.size)[..., np.newaxis]
    #     result = (mask * img).astype("uint8")
    
    return mask



class data_prefetcher():
    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.mean = torch.tensor([0.485 * 255, 0.456 * 255, 0.406 * 255]).cuda().view(1,3,1,1)
        self.std = torch.tensor([0.229 * 255, 0.224 * 255, 0.225 * 255]).cuda().view(1,3,1,1)
        # With Amp, it isn't necessary to manually convert data to half.
        # if args.fp16:
        #     self.mean = self.mean.half()
        #     self.std = self.std.half()
        self.preload()

    def preload(self):
        try:
            self.next_input, self.next_target = next(self.loader)
        except StopIteration:
            self.next_input = None
            self.next_target = None
            return
        # if record_stream() doesn't work, another option is to make sure device inputs are created
        # on the main stream.
        # self.next_input_gpu = torch.empty_like(self.next_input, device='cuda')
        # self.next_target_gpu = torch.empty_like(self.next_target, device='cuda')
        # Need to make sure the memory allocated for next_* is not still in use by the main stream
        # at the time we start copying to next_*:
        # self.stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self.stream):
            self.next_input = self.next_input.cuda(non_blocking=True)
            self.next_target = self.next_target.cuda(non_blocking=True)
            # more code for the alternative if record_stream() doesn't work:
            # copy_ will record the use of the pinned source tensor in this side stream.
            # self.next_input_gpu.copy_(self.next_input, non_blocking=True)
            # self.next_target_gpu.copy_(self.next_target, non_blocking=True)
            # self.next_input = self.next_input_gpu
            # self.next_target = self.next_target_gpu

            # With Amp, it isn't necessary to manually convert data to half.
            # if args.fp16:
            #     self.next_input = self.next_input.half()
            # else:
            self.next_input = self.next_input.float()
            self.next_input = self.next_input.sub_(self.mean).div_(self.std)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        input = self.next_input
        target = self.next_target
        if input is not None:
            input.record_stream(torch.cuda.current_stream())
        if target is not None:
            target.record_stream(torch.cuda.current_stream())
        self.preload()
        return input, target


def sen(Y_test, Y_pred, n):  # n为分类数

    sen = []
    con_mat = confusion_matrix(Y_test, Y_pred)
    for i in range(n):
        tp = con_mat[i][i]
        fn = np.sum(con_mat[i, :]) - tp
        sen1 = tp / (tp + fn)
        sen.append(sen1)

    return sen


def pre(Y_test, Y_pred, n):

    pre = []
    con_mat = confusion_matrix(Y_test, Y_pred)
    for i in range(n):
        tp = con_mat[i][i]
        fp = np.sum(con_mat[:, i]) - tp
        pre1 = tp / (tp + fp)
        pre.append(pre1)

    
    return pre


def spe(Y_test, Y_pred, n):

    spe = []
    con_mat = confusion_matrix(Y_test, Y_pred)
    for i in range(n):
        number = np.sum(con_mat[:, :])
        tp = con_mat[i][i]
        fn = np.sum(con_mat[i, :]) - tp
        fp = np.sum(con_mat[:, i]) - tp
        tn = number - tp - fn - fp
        spe1 = tn / (tn + fp)
        spe.append(spe1)

    return spe


def ACC(Y_test, Y_pred, n):

    acc = []
    con_mat = confusion_matrix(Y_test, Y_pred)
    for i in range(n):
        number = np.sum(con_mat[:, :])
        tp = con_mat[i][i]
        fn = np.sum(con_mat[i, :]) - tp
        fp = np.sum(con_mat[:, i]) - tp
        tn = number - tp - fn - fp
        acc1 = (tp + tn) / number
        acc.append(acc1)

    return acc


def mymetrics_v2(Y_test, Y_pred, n):
    sen = []
    spe = []
    acc = []
    con_mat = confusion_matrix(Y_test, Y_pred)
    for i in range(n):
        number = np.sum(con_mat[:, :])
        tp = con_mat[i][i]
        fn = np.sum(con_mat[i, :]) - tp
        fp = np.sum(con_mat[:, i]) - tp
        tn = number - tp - fn - fp
        acc1 = (tp + tn) / number
        acc.append(acc1)

    for i in range(n):
        number = np.sum(con_mat[:, :])
        tp = con_mat[i][i]
        fn = np.sum(con_mat[i, :]) - tp
        fp = np.sum(con_mat[:, i]) - tp
        tn = number - tp - fn - fp
        spe1 = tn / (tn + fp)
        spe.append(spe1)

    for i in range(n):
        tp = con_mat[i][i]
        fn = np.sum(con_mat[i, :]) - tp
        sen1 = tp / (tp + fn)
        sen.append(sen1)

    acc = np.mean(np.array(acc))
    spe = np.mean(np.array(spe))
    sen = np.mean(np.array(sen))

    return acc, spe, sen





def plot_attention_on_img(img,attentions,patch_size):

    # step-1: preprocess on raw attention matrix
    n_heads = attentions.shape[1]  # number of head
    # keep only the output patch attention
    attW=attH=int(img.shape[0]/patch_size)

    attentions = attentions[0, :, 0, -attH*attW:].reshape(n_heads, -1)

    attentions = attentions.reshape(n_heads, attW, attH)
    attentions = nn.functional.interpolate(attentions.unsqueeze(
        0), scale_factor=patch_size, mode="nearest")[0].detach().cpu().numpy()
    
    # step-2: plot attention on image
    fig1=plt.figure(figsize=(10, 10))
    text = ["Original Image", "Head Mean"]
    for i, fig in enumerate([img, np.mean(attentions, 0)]):
        plt.subplot(1, 2, i+1)
        plt.imshow(fig, cmap='inferno')
        plt.title(text[i])
        plt.axis('off')


    fig2=plt.figure(figsize=(10, 10))
    for i in range(n_heads):
        plt.subplot(n_heads//3, 3, i+1)
        plt.imshow(attentions[i], cmap='inferno')
        plt.title(f"Head n: {i+1}")
        # plt.tight_layout()
        plt.axis('off')
    
    fig2.tight_layout()
    return fig1, fig2


def plot_token_attention_hist(attentions, pool='avg'):
    # default average all prompts to get attention
    # 12 x 7 = (prompt_length_per_layer x num_layers) x num_props
    attention_scores = np.array(attentions)
    attention_scores = np.mean(attention_scores, axis=0)
    # print(np.sum(attention_scores))
    tokens = [ii for ii in range(len(attention_scores))]
    fig = plt.figure(figsize=(4, 3))
    ax = plt.subplot(1, 1, 1)
    plt.bar(tokens, attention_scores)
    plt.xlabel('Quantitative Attributes', fontsize=14)
    plt.ylabel('Significance', fontsize=14)
    props = ['$a^{s}_{1}$', '$a^{s}_{2}$', '$a^{s}_{3}$', '$a^{s}_{4}$', '$a^{s}_{m}$', '$a^{m}_{2}$', '$a^{m}_{3}$']

    ax.set_xticks([ii for ii in range(len(props))])
    ax.set_xticklabels(props)

    ax.yaxis.set_tick_params(labelsize=12)
    ax.xaxis.set_tick_params(labelsize=12)
    ax.set_ylim(0, 1)  # Set y-axis range from 0 to 1
    # plt.title('Histogram of Token Attention')
    # plt.draw()
    plt.show()

def plot_dual_token_attention_hist(atten_x,atten_y, pool='avg'):
    # default average all prompts to get attention
    # 12 x 7 = (prompt_length_per_layer x num_layers) x num_props
    attention_scores_x = np.array(atten_x)
    if len(attention_scores_x.shape)>1:
        attention_scores_x = np.mean(attention_scores_x, axis=0)
    attention_scores_y = np.array(atten_y)
    if len(attention_scores_y.shape)>1:
        attention_scores_y = np.mean(attention_scores_y, axis=0)
    # print(np.sum(attention_scores))
    tokens = [ii for ii in range(attention_scores_x.shape[-1])]
    fig = plt.figure(figsize=(4, 3))
    ax = plt.subplot(1, 1, 1)
    width = 0.9  # the width of the bars
    tokens = np.arange(len(attention_scores_x))
    fig, ax = plt.subplots()
    rects1 = ax.bar(tokens, attention_scores_x, width, label='Positive', color='darkgreen')
    rects2 = ax.bar(tokens, -attention_scores_y, width, label='Negative', color='darkred')

    ax.set_xlabel('Attribute Index')
    ax.set_ylabel('Significance')
    ax.set_xticks(tokens)
    # ax.set_xticklabels(['$a^{s}_{1}$', '$a^{s}_{2}$', '$a^{s}_{3}$', '$a^{s}_{4}$', '$a^{m}_{1}$', '$a^{m}_{2}$', '$a^{m}_{3}$'])
    ax.legend()

    ax.yaxis.set_tick_params(labelsize=12)
    ax.xaxis.set_tick_params(labelsize=12)
    ax.set_ylim(-1, 1)  # Set y-axis range from 0 to 1
    plt.show()




def plot_token_attention_mask(attentions,pool='avg'):
    # default average all prompts to get attention
    # 12 x 7 = (prompt_length_per_layer x num_layers) x num_props
    attention_scores=np.array(attentions)
    # print(attention_scores)
    tokens=[ii for ii in range(len(attention_scores))]
    fig=plt.figure(figsize=(4, 3)) 
    ax=plt.subplot(1,1,1)
    plt.bar(tokens, attention_scores)
    plt.xlabel('Quantitative Attributes',fontsize=14)
    plt.ylabel('Significance',fontsize=14)
    props=['$a^{s}_{1}$','$a^{s}_{2}$','$a^{s}_{3}$','$a^{s}_{4}$','$a^{s}_{m}$','$a^{m}_{2}$','$a^{m}_{3}$',]

    ax.set_xticks([ii for ii in range(len(props))])
    ax.set_xticklabels(props)

    ax.yaxis.set_tick_params(labelsize=12)
    ax.xaxis.set_tick_params(labelsize=12)
    # plt.title('Histogram of Token Attention')
    # plt.draw()
    plt.show()


def plot_orthogonality(tensor1, tensor2):
    """
    Verify if two tensors are orthogonal and plot the dot product matrix.
    
    Args:
    tensor1, tensor2: numpy arrays of shape (sequence_length, dim)
    """
    # Ensure the arrays have the same shape
    assert tensor1.shape == tensor2.shape, "Arrays must have the same shape"
    
    sequence_length, dim = tensor1.shape
    
    # Compute dot product for each pair in the batch and sequence
    dot_products = np.matmul(tensor1, tensor2.T)
    
    # Check if dot products are close to zero (use a small epsilon for numerical stability)
    epsilon = 1e-6
    is_orthogonal = np.allclose(dot_products, np.zeros_like(dot_products), atol=epsilon)
    
    # Plot the dot product matrix
    fig=plt.figure(figsize=(4, 3)) 
    ax=plt.subplot(1,1,1)
    sns.heatmap(dot_products, cmap='coolwarm', center=0)
    plt.tight_layout()





def mkdir_if_needed(x):
    path = os.path.dirname(x)
    if not os.path.exists(path):
        os.makedirs(path)



# plot box-and-whisker plot figure
def plot_box_and_whisker(X,y):
    plt.boxplot(column=X, by=y)
    plt.title('Box-and-Whisker Plots by Class')
    plt.suptitle('')  # Suppress the automatic 'Boxplot grouped by class_column' title
    plt.xlabel('Class')
    plt.ylabel('Values')
    # plt.show()


def extract_model_name(text):
    pattern = r'F\d+-(.+)-seed\d+'
    match = re.search(pattern, text)
    if match:
        return match.group(1)
    else:
        return None



def check_model_status(model):
    for name, param in model.named_parameters():
        if torch.isnan(param).any():
            print(f"NaN parameter in {name}")
        if param.grad is not None and torch.isnan(param.grad).any():
            print(f"NaN gradient in {name}")



def plot_global_token_attention_hist(atten_dict,save_path=None):
    # atten_dict={'0':{'pos':[],'neg':{}},'1':{'pos':[],'neg':{}},'2':{'pos':[],'neg':{}},'3':{'pos':[],'neg':{}}}
    # default average all prompts to get attention

    for key, value in atten_dict.items():
        
        if len(np.array(value['pos']).shape)==4:
            mean_pos = np.mean(np.array(value['pos']), axis=(0,1,2))
            mean_neg = np.mean(np.array(value['neg']), axis=(0,1,2))
        elif len(np.array(value['pos']).shape)==2:
            mean_pos = np.mean(np.array(value['pos']), axis=0)
            mean_neg = np.mean(np.array(value['neg']), axis=0)
        tokens=[ii for ii in range(np.array(value['pos']).shape[-1])]
        fig=plt.figure(figsize=(4, 3)) 
        ax=plt.subplot(1,1,1)
         # print(np.sum(attention_scores))
        width = 0.9  # the width of the bars
        fig, ax = plt.subplots()
        rects1 = ax.bar(tokens, mean_pos, width, label='Positive',color='darkgreen')
        rects2 = ax.bar(tokens, -mean_neg, width, label='Negative',color='darkred')
   
        plt.xlabel('Attributes Index',fontsize=14)
        plt.ylabel('Significance',fontsize=14)
        

        ax.set_xticks(tokens)
        # ax.set_xticklabels(props)

        ax.yaxis.set_tick_params(labelsize=12)
        ax.xaxis.set_tick_params(labelsize=12)
        ax.set_ylim(-1, 1)  # Set y-axis range from 0 to 1

        plt.savefig(save_path.replace('XXX',key))


def plot_global_token_attention_matrix(atten_dict,save_path=None):
    # atten_dict={'0':{'pos':[],'neg':{}},'1':{'pos':[],'neg':{}},'2':{'pos':[],'neg':{}},'3':{'pos':[],'neg':{}}}
    # default average all prompts to get attention

    for key, value in atten_dict.items():
  
        mean_pos = np.mean(np.array(value['pos']), axis=(0,1))
        mean_neg = np.mean(np.array(value['neg']), axis=(0,1))

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

        sns.heatmap(mean_pos, ax=ax1, cmap='YlOrRd')
        ax1.set_title('Positive Group Attention Pattern')
        ax1.set_xlabel('Attribute Index')
        ax1.set_ylabel('Prompt Index')

        sns.heatmap(mean_neg, ax=ax2, cmap='YlOrRd')
        ax2.set_title('Negative Group Attention Pattern')
        ax2.set_xlabel('Attribute Index')
        ax2.set_ylabel('Prompt Index')

        plt.tight_layout()

        plt.savefig(os.path.join(save_path, f"atten_matrix_{key}.png"))
        
def plot_global_uncertainty_histogram(atten_dict,pool='avg',save_path=None):
    
    for key, value in atten_dict.items():
        
        # Calculate the mean and std of the data along N dimension
        data=np.array(value)
        if pool=='mean':
            data=np.mean(data,axis=-1)
        elif pool=='sum':
            data=np.sum(data,axis=-1)
            
        N, S = data.shape
    
        # First step: Filter extreme values
        data_mean = np.mean(data, axis=0)
        data_std = np.std(data, axis=0)
        z_scores = (data - data_mean) / data_std

        # Create mask for non-extreme values
        mask = np.abs(z_scores) <= 2
        filtered_data = np.copy(data)
        filtered_data[~mask] = np.nan  # Set outliers to NaN

        # Second step: Normalize remaining data to [-1, 1] for each attribute
        normalized_data = np.zeros_like(filtered_data)
        for i in range(S):
            valid_data = filtered_data[:, i][~np.isnan(filtered_data[:, i])]
            if len(valid_data) > 0:
                min_val = np.min(valid_data)
                max_val = np.max(valid_data)
                # Avoid division by zero
                if max_val != min_val:
                    normalized_data[:, i] = 2 * (filtered_data[:, i] - min_val) / (max_val - min_val) - 1
                else:
                    normalized_data[:, i] = filtered_data[:, i] - min_val

        # Create figure with specified size
        plt.figure(figsize=(5, 4))

        # Create x positions for attributes
        x = np.arange(S)

        # Create box plots
        bp = plt.boxplot([normalized_data[:, i][~np.isnan(normalized_data[:, i])] for i in range(S)], 
                         positions=x,
                         patch_artist=True,
                         medianprops=dict(color="black", linewidth=1.5),
                         boxprops=dict(facecolor='lightblue', alpha=0.5),
                         widths=0.4,
                         showmeans=True,
                         meanprops={'marker':'D',
                                  'markerfacecolor':'red',
                                  'markeredgecolor':'darkred',
                                  'markersize':8})

        # Plot scatter points for non-outlier samples
        for i in range(N):
            valid_points = ~np.isnan(normalized_data[i])
            if np.any(valid_points):
                plt.scatter(x[valid_points], normalized_data[i, valid_points], 
                           color='green', s=20, alpha=0.5, zorder=3)

        # Add horizontal line at y=0
        plt.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

        # Customize plot
        plt.grid(True, alpha=0.3)
        plt.xticks(x, [str(i) for i in range(S)])
    
        plt.tight_layout()

        plt.savefig(os.path.join(save_path.replace('XXX',f'{key}-{pool}')))

    



def positive_softmax(x):
    # Convert input to numpy array if it isn't already
    x = np.asarray(x)
    
    # Create a mask for strictly positive values (> 0)
    positive_mask = (x > 0)
    
    # Initialize output array with zeros
    output = np.zeros_like(x, dtype=np.float32)
    
    # Only process if there are any positive values
    if np.any(positive_mask):
        # Extract positive values
        positive_values = x[positive_mask]
        
        # Compute softmax only on positive values
        # Subtract max for numerical stability
        exp_values = np.exp(positive_values - np.max(positive_values))
        softmax_values = exp_values / np.sum(exp_values)
        
        # Place the computed values back in the output array
        output[positive_mask] = softmax_values
    
    return output


# plot positive and negative attention on image
def plot_pos_neg_attention_on_img(x, y, mode='all', alpha=0.5):
    """
    Plot both positive and negative attributions on the same image using cv2.imshow.
    
    Args:
        x (torch.Tensor): Input image tensor of shape [H, W, 3]
        y (torch.Tensor): Attribution tensor of same spatial dimensions as x
        mode (str): 'all' for both, 'pos' for positive only, 'neg' for negative only
        alpha (float): Transparency for the overlay (0-1)
    """
    
    img_cv = x
    
    # Prepare attribution
    attr = y
    if len(attr.shape) == 3:  # If attribution has channels
        attr = np.mean(attr, axis=-1)  # Average across channels
    
    # Create masks for positive and negative values
    pos_mask = np.maximum(attr, 0)
    neg_mask = np.minimum(attr, 0)
    
    # Normalize masks
    if pos_mask.max() != 0:
        pos_mask = pos_mask / pos_mask.max()
    if neg_mask.min() != 0:
        neg_mask = -neg_mask / -neg_mask.min()
    
    # Convert masks to heatmap format
    pos_heatmap = (pos_mask * 255).astype(np.uint8)
    neg_heatmap = (neg_mask * 255).astype(np.uint8)
    
    # Create empty overlay image
    overlay = np.zeros_like(img_cv)
    
    if mode in ['all', 'pos']:
        # Add positive attributions in red
        pos_colored = cv2.applyColorMap(pos_heatmap, cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(overlay, 1, pos_colored, alpha, 0)
    
    if mode in ['all', 'neg']:
        # Add negative attributions in blue
        neg_colored = cv2.applyColorMap(neg_heatmap, cv2.COLORMAP_BLUE)
        overlay = cv2.addWeighted(overlay, 1, neg_colored, alpha, 0)
    
    result = cv2.addWeighted(img_cv, 1, overlay, alpha, 0)
    result = result.astype(np.uint8)
        
    return result

