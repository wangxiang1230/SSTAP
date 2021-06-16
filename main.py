import sys
from dataset import VideoDataSet, VideoDataSet_unlabel
from loss_function import bmn_loss_func, get_mask
import os
import json
import torch
import torch.nn.parallel
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import numpy as np
import opts
from ipdb import set_trace
from models import BMN, TemporalShift, TemporalShift_random
import pandas as pd
import random
from post_processing import BMN_post_processing
from eval import evaluation_proposal
from ipdb import set_trace

seed = 400
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)  # Numpy module.
random.seed(seed)  # Python random module.
torch.manual_seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3'
blue = lambda x: '\033[94m' + x + '\033[0m'
sys.dont_write_bytecode = True
global_step = 0
eval_loss = []
consistency_rampup = 5
consistency = 6  # 30  # 3  # None


def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)


def softmax_mse_loss(input_logits, target_logits):
    """Takes softmax on both sides and returns MSE loss

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    # input_softmax = F.softmax(input_logits, dim=1)
    # target_softmax = F.softmax(target_logits, dim=1)
    # num_classes = input_logits.size()[1]
    # return F.mse_loss(input_softmax, target_softmax, reduction='sum') / num_classes   # size_average=False
    return F.mse_loss(input_logits, target_logits, reduction='mean')


def softmax_kl_loss(input_logits, target_logits):
    """Takes softmax on both sides and returns KL divergence

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    # input_log_softmax = F.log_softmax(input_logits, dim=1)
    # target_softmax = F.softmax(target_logits, dim=1)
    # return F.kl_div(input_log_softmax, target_softmax, reduction='sum')
    return F.kl_div(input_logits, target_logits, reduction='mean')


def Motion_MSEloss(output,clip_label,motion_mask=torch.ones(100).cuda()):
    z = torch.pow((output-clip_label),2)
    loss = torch.mean(motion_mask*z)
    return loss


def sigmoid_rampup(current, rampup_length):
    """Exponential rampup from https://arxiv.org/abs/1610.02242"""
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))


def linear_rampup(current, rampup_length):
    """Linear rampup"""
    assert current >= 0 and rampup_length >= 0
    if current >= rampup_length:
        return 1.0
    else:
        return current / rampup_length


def cosine_rampdown(current, rampdown_length):
    """Cosine rampdown from https://arxiv.org/abs/1608.03983"""
    assert 0 <= current <= rampdown_length
    return float(.5 * (np.cos(np.pi * current / rampdown_length) + 1))


def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return consistency * sigmoid_rampup(epoch, consistency_rampup)


def train_BMN(data_loader, model, optimizer, epoch, bm_mask):
    model.train()
    epoch_pemreg_loss = 0
    epoch_pemclr_loss = 0
    epoch_tem_loss = 0
    epoch_loss = 0
    for n_iter, (input_data, label_confidence, label_start, label_end) in enumerate(data_loader): 
        input_data = input_data.cuda()                  
        label_start = label_start.cuda()
        label_end = label_end.cuda()
        label_confidence = label_confidence.cuda()
        confidence_map, start, end = model(input_data)     # [B, 2, 100, 100], [B,100],[B,100]
        loss = bmn_loss_func(confidence_map, start, end, label_confidence, label_start, label_end, bm_mask.cuda())  # loss = tem_loss + 10 * pem_reg_loss + pem_cls_loss
                                                                                                                # return loss, tem_loss, pem_reg_loss, pem_cls_loss
        optimizer.zero_grad()
        loss[0].backward()
        optimizer.step()

        epoch_pemreg_loss += loss[2].cpu().detach().numpy()
        epoch_pemclr_loss += loss[3].cpu().detach().numpy()
        epoch_tem_loss += loss[1].cpu().detach().numpy()
        epoch_loss += loss[0].cpu().detach().numpy()

    print(
        "BMN training loss(epoch %d): tem_loss: %.03f, pem class_loss: %.03f, pem reg_loss: %.03f, total_loss: %.03f" % (
            epoch, epoch_tem_loss / (n_iter + 1),
            epoch_pemclr_loss / (n_iter + 1),
            epoch_pemreg_loss / (n_iter + 1),
            epoch_loss / (n_iter + 1)))


def train_BMN_Semi(data_loader, train_loader_unlabel, model, model_ema, optimizer, epoch, bm_mask):
    global global_step
    model.train()
    epoch_pemreg_loss = 0
    epoch_pemclr_loss = 0
    epoch_tem_loss = 0
    epoch_loss = 0
    consistency_loss_all = 0
    consistency_loss_ema_all = 0
    consistency_criterion = softmax_mse_loss  # softmax_kl_loss
    
    temporal_perb = TemporalShift_random(400, 64)   
    order_clip_criterion = nn.CrossEntropyLoss()
    consistency = True
    clip_order = True
    dropout2d = True
    temporal_re = True
    unlabeled_train_iter = iter(train_loader_unlabel)
    for n_iter, (input_data, label_confidence, label_start, label_end) in enumerate(data_loader):  
        input_data = input_data.cuda()  
        label_start = label_start.cuda()
        label_end = label_end.cuda()
        label_confidence = label_confidence.cuda()

        input_data_student = temporal_perb(input_data)
        if dropout2d:
            input_data_student = F.dropout2d(input_data_student, 0.2)
        else:
            input_data_student = F.dropout(input_data_student, 0.2)
        confidence_map, start, end = model(input_data_student)  # [B, 2, 100, 100], [B,100],[B,100]
        loss = bmn_loss_func(confidence_map, start, end, label_confidence, label_start, label_end, bm_mask.cuda())
        confidence_map = confidence_map * bm_mask.cuda()
        if temporal_re:
            input_recons = F.dropout2d(input_data.permute(0,2,1), 0.2).permute(0,2,1)
        else:
            input_recons = F.dropout2d(input_data, 0.2)
        recons_feature = model(input_recons, recons=True)

        try:
            input_data_unlabel= unlabeled_train_iter.next()
            input_data_unlabel = input_data_unlabel.cuda()
        except:
            unlabeled_train_iter = iter(train_loader_unlabel)
            input_data_unlabel = unlabeled_train_iter.next()
            input_data_unlabel = input_data_unlabel.cuda()
        
        input_data_unlabel_student = temporal_perb(input_data_unlabel)
        if dropout2d:
            input_data_unlabel_student = F.dropout2d(input_data_unlabel_student, 0.2)
        else:
            input_data_unlabel_student = F.dropout(input_data_unlabel_student, 0.2)
        confidence_map_unlabel_student, start_unlabel_student, end_unlabel_student = model(input_data_unlabel_student)
        confidence_map_unlabel_student = confidence_map_unlabel_student * bm_mask.cuda()

        # label
        input_data_label_student_flip = F.dropout2d(input_data.flip(2).contiguous(), 0.1)
        confidence_map_label_student_flip, start_label_student_flip, end_label_student_flip = model(
            input_data_label_student_flip)
        confidence_map_label_student_flip = confidence_map_label_student_flip * bm_mask.cuda()
        # unlabel
        input_data_unlabel_student_flip = F.dropout2d(input_data_unlabel.flip(2).contiguous(), 0.1)
        confidence_map_unlabel_student_flip, start_unlabel_student_flip, end_unlabel_student_flip = model(
            input_data_unlabel_student_flip)
        confidence_map_unlabel_student_flip = confidence_map_unlabel_student_flip * bm_mask.cuda()
        
        if temporal_re:
            recons_input_student = F.dropout2d(input_data_unlabel.permute(0,2,1), 0.2).permute(0,2,1)
        else:
            recons_input_student = F.dropout2d(input_data_unlabel, 0.2)

        recons_feature_unlabel_student = model(recons_input_student, recons=True)
        
        loss_recons = 0.0005 * (
                Motion_MSEloss(recons_feature, input_data) + Motion_MSEloss(recons_feature_unlabel_student,
                                                                            input_data_unlabel))  # 0.0001

        with torch.no_grad():
            # input_data_unlabel = input_data_unlabel.cuda()
            input_data_ema = F.dropout(input_data, 0.05)  # 0.3
            confidence_map_teacher, start_teacher, end_teacher = model_ema(input_data_ema)
            confidence_map_teacher = confidence_map_teacher * bm_mask.cuda()
            input_data_unlabel_teacher = F.dropout(input_data_unlabel, 0.05)  # 0.3
            confidence_map_unlabel_teacher, start_unlabel_teacher, end_unlabel_teacher = model_ema(
                input_data_unlabel_teacher)
            confidence_map_unlabel_teacher = confidence_map_unlabel_teacher * bm_mask.cuda()

            # flip (label)
            out = torch.zeros_like(confidence_map_unlabel_teacher)
            out_m = confidence_map_unlabel_teacher.flip(3).contiguous()
            for i in range(100):
                out[:, :, i, :100 - i] = out_m[:, :, i, i:]
            confidence_map_unlabel_teacher_flip = out

            # flip (unlabel)
            out = torch.zeros_like(confidence_map_teacher)
            out_m = confidence_map_teacher.flip(3).contiguous()
            for i in range(100):
                out[:, :, i, :100 - i] = out_m[:, :, i, i:]
            confidence_map_label_teacher_flip = out
            # start_unlabel_teacher_flip = start_unlabel_teacher.flip(1).contiguous()
            # end_unlabel_teacher_flip = end_unlabel_teacher.flip(1).contiguous()

            # add mask
            start_unlabel_teacher[start_unlabel_teacher >= 0.9] = 1.0
            start_unlabel_teacher[start_unlabel_teacher <= 0.1] = 0.0  # 2_add
            end_unlabel_teacher[end_unlabel_teacher >= 0.9] = 1.0
            end_unlabel_teacher[end_unlabel_teacher <= 0.1] = 0.0

            # flip (label)
            start_label_teacher_flip = start_teacher.flip(1).contiguous()
            end_label_teacher_flip = end_teacher.flip(1).contiguous()

            # flip (unlabel)
            start_unlabel_teacher_flip = start_unlabel_teacher.flip(1).contiguous()
            end_unlabel_teacher_flip = end_unlabel_teacher.flip(1).contiguous()

            mask = torch.eq(
                (start_unlabel_teacher.max(1)[0] > 0.6).float() + (end_unlabel_teacher.max(1)[0] > 0.6).float(), 2.)
            confidence_map_unlabel_teacher = confidence_map_unlabel_teacher[mask]
            start_unlabel_teacher = start_unlabel_teacher[mask]
            end_unlabel_teacher = end_unlabel_teacher[mask]

            # flip
            confidence_map_unlabel_teacher_flip = confidence_map_unlabel_teacher_flip[mask]
            start_unlabel_teacher_flip = start_unlabel_teacher_flip[mask]
            end_unlabel_teacher_flip = end_unlabel_teacher_flip[mask]

        # add mask
        confidence_map_unlabel_student = confidence_map_unlabel_student[mask]
        start_unlabel_student = start_unlabel_student[mask]
        end_unlabel_student = end_unlabel_student[mask]

        # flip add mask
        confidence_map_unlabel_student_flip = confidence_map_unlabel_student_flip[mask]
        start_unlabel_student_flip = start_unlabel_student_flip[mask]
        end_unlabel_student_flip = end_unlabel_student_flip[mask]

        if consistency:
            consistency_weight = get_current_consistency_weight(epoch)
            # meters.update('cons_weight', consistency_weight)
            # set_trace()
            consistency_loss = consistency_weight * (consistency_criterion(confidence_map, confidence_map_teacher) +
                                                     consistency_criterion(start, start_teacher) +
                                                     consistency_criterion(end, end_teacher))

            consistency_loss_ema = consistency_weight * (
                    consistency_criterion(confidence_map_unlabel_teacher, confidence_map_unlabel_student) +
                    consistency_criterion(start_unlabel_teacher, start_unlabel_student) +
                    consistency_criterion(end_unlabel_teacher, end_unlabel_student))
            # set_trace()
            if torch.isnan(consistency_loss_ema):
                consistency_loss_ema = torch.tensor(0.).cuda()

            consistency_loss_ema_flip = 0.1 * consistency_weight * (
                    consistency_criterion(confidence_map_unlabel_teacher_flip, confidence_map_unlabel_student_flip) +
                    consistency_criterion(start_unlabel_teacher_flip, start_unlabel_student_flip) +
                    consistency_criterion(end_unlabel_teacher_flip, end_unlabel_student_flip)) + 0.1 * consistency_weight * (
                    consistency_criterion(confidence_map_label_teacher_flip, confidence_map_label_student_flip) +
                    consistency_criterion(start_label_teacher_flip, start_label_student_flip) +
                    consistency_criterion(end_label_teacher_flip, end_label_student_flip))

            # meters.update('cons_loss', consistency_loss.item())

        else:
            consistency_loss = torch.tensor(0).cuda()
            consistency_loss_ema = torch.tensor(0).cuda()
            consistency_loss_ema_flip = torch.tensor(0).cuda()
            # meters.update('cons_loss', 0)

        if clip_order:
            input_data_all = torch.cat([input_data, input_data_unlabel], 0)
            batch_size, C, T = input_data_all.size()
            idx = torch.randperm(batch_size)
            input_data_all_new = input_data_all[idx]
            forw_input = torch.cat(
                [input_data_all_new[:batch_size // 2, :, T // 2:], input_data_all_new[:batch_size // 2, :, :T // 2]], 2)
            back_input = input_data_all_new[batch_size // 2:, :, :]
            input_all = torch.cat([forw_input, back_input], 0)
            label_order = [0] * (batch_size // 2) + [1] * (batch_size - batch_size // 2)
            label_order = torch.tensor(label_order).long().cuda()
            out = model(input_all, clip_order=True)
            loss_clip_order = order_clip_criterion(out, label_order)

        loss_all = loss[0] + consistency_loss + consistency_loss_ema + loss_recons + 0.01 * loss_clip_order + consistency_loss_ema_flip
        optimizer.zero_grad()
        loss_all.backward()
        optimizer.step()
        global_step += 1
        update_ema_variables(model, model_ema, 0.999, float(global_step/20))   # //5  //25

        epoch_pemreg_loss += loss[2].cpu().detach().numpy()
        epoch_pemclr_loss += loss[3].cpu().detach().numpy()
        epoch_tem_loss += loss[1].cpu().detach().numpy()
        epoch_loss += loss[0].cpu().detach().numpy()
        consistency_loss_all += consistency_loss.cpu().detach().numpy()
        consistency_loss_ema_all += consistency_loss_ema.cpu().detach().numpy()
        if n_iter % 10 == 0:
            print(
                "training %d (epoch %d): tem_loss: %.03f, pem class_loss: %.03f, pem reg_loss: %.03f, consistency_loss: %.05f, consistency_loss_ema: %.05f, total_loss: %.03f" % (global_step,
                    epoch, epoch_tem_loss / (n_iter + 1),
                    epoch_pemclr_loss / (n_iter + 1),
                    epoch_pemreg_loss / (n_iter + 1),
                    consistency_loss_all / (n_iter + 1),
                    consistency_loss_ema_all / (n_iter + 1),
                    epoch_loss / (n_iter + 1)))

    print(
        blue("BMN training loss(epoch %d): tem_loss: %.03f, pem class_loss: %.03f, pem reg_loss: %.03f, total_loss: %.03f" % (
            epoch, epoch_tem_loss / (n_iter + 1),
            epoch_pemclr_loss / (n_iter + 1),
            epoch_pemreg_loss / (n_iter + 1),
            epoch_loss / (n_iter + 1))))


def train_BMN_Semi_Full(data_loader, model, model_ema, optimizer, epoch, bm_mask):
    global global_step
    model.train()
    epoch_pemreg_loss = 0
    epoch_pemclr_loss = 0
    epoch_tem_loss = 0
    epoch_loss = 0
    consistency_loss_all = 0
    consistency_loss_ema_all = 0
    consistency_criterion = softmax_mse_loss  # softmax_kl_loss
    # perturbance = nn.dropout(0.3)
    temporal_perb = TemporalShift_random(400, 64)   # TemporalShift(400, 8)  16
    order_clip_criterion = nn.CrossEntropyLoss()
    consistency = True
    clip_order = True
    dropout2d = True
    temporal_re = True
    # unlabeled_train_iter = iter(train_loader_unlabel)
    for n_iter, (input_data, label_confidence, label_start, label_end) in enumerate(data_loader):
        input_data = input_data.cuda()
        label_start = label_start.cuda()
        label_end = label_end.cuda()
        label_confidence = label_confidence.cuda()

        input_data_student = temporal_perb(input_data)
        if dropout2d:
            input_data_student = F.dropout2d(input_data_student, 0.2)
        else:
            input_data_student = F.dropout(input_data_student, 0.2)
        confidence_map, start, end = model(input_data_student)  # [B, 2, 100, 100], [B,100],[B,100]
        loss = bmn_loss_func(confidence_map, start, end, label_confidence, label_start, label_end, bm_mask.cuda())
        confidence_map = confidence_map * bm_mask.cuda()
        if temporal_re:
            input_recons = F.dropout2d(input_data.permute(0, 2, 1), 0.2).permute(0, 2, 1)
        else:
            input_recons = F.dropout2d(input_data, 0.2)
        recons_feature = model(input_recons, recons=True)

        # try:
        #     input_data_unlabel= unlabeled_train_iter.next()
        #     input_data_unlabel = input_data_unlabel.cuda()
        # except:
        #     unlabeled_train_iter = iter(train_loader_unlabel)
        #     input_data_unlabel = unlabeled_train_iter.next()
        #     input_data_unlabel = input_data_unlabel.cuda()
        # input_data_unlabel = F.dropout2d(input_data_unlabel.cuda(), 0.2)
        # input_data_unlabel_student = temporal_perb(input_data_unlabel)
        # if dropout2d:
        #     input_data_unlabel_student = F.dropout2d(input_data_unlabel_student, 0.2)
        # else:
        #     input_data_unlabel_student = F.dropout(input_data_unlabel_student, 0.2)
        # confidence_map_unlabel_student, start_unlabel_student, end_unlabel_student = model(input_data_unlabel_student)
        # confidence_map_unlabel_student = confidence_map_unlabel_student * bm_mask.cuda()

        input_data_label_student_flip = F.dropout2d(input_data.flip(2).contiguous(), 0.1)
        confidence_map_label_student_flip, start_label_student_flip, end_label_student_flip = model(
            input_data_label_student_flip)
        confidence_map_label_student_flip = confidence_map_label_student_flip * bm_mask.cuda()

        # recons_input_student = F.dropout2d(input_data_unlabel.cuda(), 0.2)
        # recons_feature_unlabel_student = model(recons_input_student, recons=True)
        # set_trace()
        loss_recons = 0.0005 * (
                Motion_MSEloss(recons_feature, input_data))  # 0.0001

        with torch.no_grad():
            # input_data_unlabel = input_data_unlabel.cuda()
            input_data_ema = F.dropout(input_data, 0.05)  # 0.3
            confidence_map_teacher, start_teacher, end_teacher = model_ema(input_data_ema)
            confidence_map_teacher = confidence_map_teacher * bm_mask.cuda()
            # input_data_unlabel_teacher = F.dropout(input_data_unlabel, 0.05)  # 0.3
            # confidence_map_unlabel_teacher, start_unlabel_teacher, end_unlabel_teacher = model_ema(
            #     input_data_unlabel_teacher)
            # confidence_map_unlabel_teacher = confidence_map_unlabel_teacher * bm_mask.cuda()

            # flip
            out = torch.zeros_like(confidence_map_teacher)
            out_m = confidence_map_teacher.flip(3).contiguous()
            for i in range(100):
                out[:, :, i, :100 - i] = out_m[:, :, i, i:]
            confidence_map_label_teacher = out
            # start_unlabel_teacher_flip = start_unlabel_teacher.flip(1).contiguous()
            # end_unlabel_teacher_flip = end_unlabel_teacher.flip(1).contiguous()

            # add mask
            # start_label_teacher[start_label_teacher >= 0.9] = 1.0
            # start_label_teacher[start_label_teacher <= 0.1] = 0.0  # 2_add
            # end_unlabel_teacher[end_unlabel_teacher >= 0.9] = 1.0
            # end_unlabel_teacher[end_unlabel_teacher <= 0.1] = 0.0

            # flip
            start_label_teacher_flip = label_start.flip(1).contiguous()
            end_label_teacher_flip = label_end.flip(1).contiguous()

            # mask = torch.eq(
            #     (start_unlabel_teacher.max(1)[0] > 0.6).float() + (end_unlabel_teacher.max(1)[0] > 0.6).float(), 2.)
            # confidence_map_unlabel_teacher = confidence_map_unlabel_teacher[mask]
            # start_unlabel_teacher = start_unlabel_teacher[mask]
            # end_unlabel_teacher = end_unlabel_teacher[mask]

            # flip
            # confidence_map_unlabel_teacher_flip = confidence_map_unlabel_teacher_flip[mask]
            # start_unlabel_teacher_flip = start_unlabel_teacher_flip[mask]
            # end_unlabel_teacher_flip = end_unlabel_teacher_flip[mask]

        # add mask
        # confidence_map_unlabel_student = confidence_map_unlabel_student[mask]
        # start_unlabel_student = start_unlabel_student[mask]
        # end_unlabel_student = end_unlabel_student[mask]

        # flip add mask
        # confidence_map_unlabel_student_flip = confidence_map_label_student_flip[mask]
        # start_unlabel_student_flip = start_label_student_flip[mask]
        # end_unlabel_student_flip = end_label_student_flip[mask]

        if consistency:
            consistency_weight = get_current_consistency_weight(epoch)
            # meters.update('cons_weight', consistency_weight)
            # set_trace()
            consistency_loss = consistency_weight * (consistency_criterion(confidence_map, confidence_map_teacher) +
                                                     consistency_criterion(start, start_teacher) +
                                                     consistency_criterion(end, end_teacher))

            consistency_loss_ema_flip = 0.1 * consistency_weight * (
                    consistency_criterion(confidence_map_label_student_flip, confidence_map_label_teacher) +
                    consistency_criterion(start_label_student_flip, start_label_teacher_flip) +
                    consistency_criterion(end_label_student_flip, end_label_teacher_flip))

            # consistency_loss_ema_flip = 0.1 * consistency_weight * (
            #         consistency_criterion(confidence_map_label_teacher, confidence_map_label_student_flip) +
            #         consistency_criterion(start_label_teacher_flip, start_label_student_flip) +
            #         consistency_criterion(end_label_teacher_flip, end_label_student_flip))

            # meters.update('cons_loss', consistency_loss.item())

        else:
            consistency_loss = torch.tensor(0).cuda()
            consistency_loss_ema = torch.tensor(0).cuda()
            consistency_loss_ema_flip = torch.tensor(0).cuda()
            # meters.update('cons_loss', 0)

        if clip_order:
            input_data_all = input_data  # torch.cat([input_data, input_data_unlabel], 0)
            batch_size, C, T = input_data_all.size()
            idx = torch.randperm(batch_size)
            input_data_all_new = input_data_all[idx]
            forw_input = torch.cat(
                [input_data_all_new[:batch_size // 2, :, T // 2:], input_data_all_new[:batch_size // 2, :, :T // 2]], 2)
            back_input = input_data_all_new[batch_size // 2:, :, :]
            input_all = torch.cat([forw_input, back_input], 0)
            label_order = [0] * (batch_size // 2) + [1] * (batch_size - batch_size // 2)
            label_order = torch.tensor(label_order).long().cuda()
            out = model(input_all, clip_order=True)
            loss_clip_order = order_clip_criterion(out, label_order)

        loss_all = loss[0] + consistency_loss + loss_recons + 0.01 * loss_clip_order + consistency_loss_ema_flip
        optimizer.zero_grad()
        loss_all.backward()
        optimizer.step()
        global_step += 1
        update_ema_variables(model, model_ema, 0.999, float(global_step/20))   # //5  //25

        epoch_pemreg_loss += loss[2].cpu().detach().numpy()
        epoch_pemclr_loss += loss[3].cpu().detach().numpy()
        epoch_tem_loss += loss[1].cpu().detach().numpy()
        epoch_loss += loss[0].cpu().detach().numpy()
        consistency_loss_all += consistency_loss.cpu().detach().numpy()
        # consistency_loss_ema_all += consistency_loss_ema.cpu().detach().numpy()
        if n_iter % 10 == 0:
            print(
                "training %d (epoch %d): tem_loss: %.03f, pem class_loss: %.03f, pem reg_loss: %.03f, consistency_loss: %.05f, total_loss: %.03f" % (global_step,
                    epoch, epoch_tem_loss / (n_iter + 1),
                    epoch_pemclr_loss / (n_iter + 1),
                    epoch_pemreg_loss / (n_iter + 1),
                    consistency_loss_all / (n_iter + 1),
                    # consistency_loss_ema_all / (n_iter + 1),
                    epoch_loss / (n_iter + 1)))

    print(
        blue("BMN training loss(epoch %d): tem_loss: %.03f, pem class_loss: %.03f, pem reg_loss: %.03f, total_loss: %.03f" % (
            epoch, epoch_tem_loss / (n_iter + 1),
            epoch_pemclr_loss / (n_iter + 1),
            epoch_pemreg_loss / (n_iter + 1),
            epoch_loss / (n_iter + 1))))


def test_BMN(data_loader, model, epoch, bm_mask):
    global eval_loss
    model.eval()
    best_loss = 1e10
    epoch_pemreg_loss = 0
    epoch_pemclr_loss = 0
    epoch_tem_loss = 0
    epoch_loss = 0
    for n_iter, (input_data, label_confidence, label_start, label_end) in enumerate(data_loader):
        input_data = input_data.cuda()
        label_start = label_start.cuda()
        label_end = label_end.cuda()
        label_confidence = label_confidence.cuda()

        confidence_map, start, end = model(input_data)
        loss = bmn_loss_func(confidence_map, start, end, label_confidence, label_start, label_end, bm_mask.cuda())

        epoch_pemreg_loss += loss[2].cpu().detach().numpy()
        epoch_pemclr_loss += loss[3].cpu().detach().numpy()
        epoch_tem_loss += loss[1].cpu().detach().numpy()
        epoch_loss += loss[0].cpu().detach().numpy()

    print(
        blue("BMN val loss(epoch %d): tem_loss: %.03f, pem class_loss: %.03f, pem reg_loss: %.03f, total_loss: %.03f" % (
            epoch, epoch_tem_loss / (n_iter + 1),
            epoch_pemclr_loss / (n_iter + 1),
            epoch_pemreg_loss / (n_iter + 1),
            epoch_loss / (n_iter + 1))))

    eval_loss.append(epoch_loss / (n_iter + 1))

    state = {'epoch': epoch + 1,
             'state_dict': model.state_dict()}
    torch.save(state, opt["checkpoint_path"] + "/BMN_checkpoint.pth.tar")       # ./checkpoint
    if epoch_loss < model.module.tem_best_loss:
        model.module.tem_best_loss = epoch_loss
        torch.save(state, opt["checkpoint_path"] + "/BMN_best.pth.tar")
    # eval_loss.append(epoch_loss / (n_iter + 1))
    opt_file = open(opt["checkpoint_path"] + "/output_eval_loss.json", "w")
    json.dump(eval_loss, opt_file)
    opt_file.close()


def test_BMN_ema(data_loader, model, epoch, bm_mask):
    model.eval()
    best_loss = 1e10
    epoch_pemreg_loss = 0
    epoch_pemclr_loss = 0
    epoch_tem_loss = 0
    epoch_loss = 0
    for n_iter, (input_data, label_confidence, label_start, label_end) in enumerate(data_loader):
        input_data = input_data.cuda()
        label_start = label_start.cuda()
        label_end = label_end.cuda()
        label_confidence = label_confidence.cuda()

        confidence_map, start, end = model(input_data)
        loss = bmn_loss_func(confidence_map, start, end, label_confidence, label_start, label_end, bm_mask.cuda())

        epoch_pemreg_loss += loss[2].cpu().detach().numpy()
        epoch_pemclr_loss += loss[3].cpu().detach().numpy()
        epoch_tem_loss += loss[1].cpu().detach().numpy()
        epoch_loss += loss[0].cpu().detach().numpy()

    print(
        blue("BMN val_ema loss(epoch %d): tem_loss: %.03f, pem class_loss: %.03f, pem reg_loss: %.03f, total_loss: %.03f" % (
            epoch, epoch_tem_loss / (n_iter + 1),
            epoch_pemclr_loss / (n_iter + 1),
            epoch_pemreg_loss / (n_iter + 1),
            epoch_loss / (n_iter + 1))))

    state = {'epoch': epoch + 1,
             'state_dict': model.state_dict()}
    torch.save(state, opt["checkpoint_path"] + "/BMN_checkpoint_ema.pth.tar")       # ./checkpoint
    if epoch_loss < model.module.tem_best_loss:                                                    
        model.module.tem_best_loss = epoch_loss
        torch.save(state, opt["checkpoint_path"] + "/BMN_best_ema.pth.tar")


def BMN_Train(opt):
    model = BMN(opt)
    model = torch.nn.DataParallel(model, device_ids=[0, 1, 2, 3]).cuda()
    model_ema = BMN(opt)
    model_ema = torch.nn.DataParallel(model_ema, device_ids=[0, 1, 2, 3]).cuda()
    for param in model_ema.parameters():
        param.detach_()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=opt["training_lr"],         
                           weight_decay=opt["weight_decay"])                               # 1e-4

    train_loader = torch.utils.data.DataLoader(VideoDataSet(opt, subset="train"),             # [16,400,100]
                                               batch_size=opt["batch_size"], shuffle=True, drop_last=True,
                                               num_workers=8, pin_memory=True)
    if opt['use_semi'] and opt['unlabel_percent'] > 0.:
        train_loader_unlabel = torch.utils.data.DataLoader(VideoDataSet_unlabel(opt, subset="unlabel"),  # [16,400,100]
                                                   batch_size=min(max(round(opt["batch_size"]*opt['unlabel_percent']/(4*(1.-opt['unlabel_percent'])))*4, 4), 24), shuffle=True,drop_last=True,
                                                   num_workers=8, pin_memory=True)

    test_loader = torch.utils.data.DataLoader(VideoDataSet(opt, subset="validation"),
                                              batch_size=opt["batch_size"], shuffle=False,
                                              num_workers=8, pin_memory=True)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opt["step_size"], gamma=opt["step_gamma"])    # 7     0.1
    bm_mask = get_mask(opt["temporal_scale"])
    use_semi = opt['use_semi']
    print('use {} label for training!!!'.format(1-opt['unlabel_percent']))
    print('training batchsize : {}'.format(opt["batch_size"]))
    print('unlabel_training batchsize : {}'.format(min(max(round(opt["batch_size"]*opt['unlabel_percent']/(4*(1.-opt['unlabel_percent'])))*4, 4), 24)))
    for epoch in range(opt["train_epochs"]):          # 9
        # scheduler.step()
        if use_semi:
            if opt['unlabel_percent'] == 0.:
                print('use Semi !!! use all label !!!')
                train_BMN_Semi_Full(train_loader, model, model_ema, optimizer, epoch, bm_mask)
                test_BMN(test_loader, model, epoch, bm_mask)
                test_BMN_ema(test_loader, model_ema, epoch, bm_mask)
            else:
                print('use Semi !!!')
                train_BMN_Semi(train_loader, train_loader_unlabel, model, model_ema, optimizer, epoch, bm_mask)
                test_BMN(test_loader, model, epoch, bm_mask)
                test_BMN_ema(test_loader, model_ema, epoch, bm_mask)
        else:
            print('use Fewer label !!!')
            train_BMN(train_loader, model, optimizer, epoch, bm_mask)
            test_BMN(test_loader, model, epoch, bm_mask)
        scheduler.step()


def BMN_inference(opt, eval_name):
    model = BMN(opt)
    model = torch.nn.DataParallel(model, device_ids=[0, 1, 2, 3]).cuda()
    model_checkpoint_dir = opt["checkpoint_path"] + eval_name   # BMN_checkpoint.pth.tar  BMN_best.pth.tar
    checkpoint = torch.load(model_checkpoint_dir)       # BMN_best.pth.tar
    print('load :', model_checkpoint_dir, ' OK !')
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()                                                 

    test_loader = torch.utils.data.DataLoader(VideoDataSet(opt, subset="validation"),            
                                              batch_size=8, shuffle=False,
                                              num_workers=8, pin_memory=True, drop_last=False)   
    tscale = opt["temporal_scale"]
    with torch.no_grad():
        for idx, input_data in test_loader:
            # set_trace()
            length = idx.shape[0]
            # for ii in range(length):
            video_name = []
            for ii in range(length):
                video_name_video = test_loader.dataset.video_list[idx[ii]]
                video_name.append(video_name_video)
            input_data = input_data.cuda()
            confidence_map, start, end = model(input_data)
            # set_trace()
            for ii in range(length):

                start_scores = start[ii].detach().cpu().numpy()
                end_scores = end[ii].detach().cpu().numpy()
                clr_confidence = (confidence_map[ii][1]).detach().cpu().numpy()
                reg_confidence = (confidence_map[ii][0]).detach().cpu().numpy()

                max_start = max(start_scores)
                max_end = max(end_scores)

                ####################################################################################################
                # generate the set of start points and end points
                start_bins = np.zeros(len(start_scores))
                start_bins[0] = 1  # [1,0,0...,0,1] 
                for idx in range(1, tscale - 1):                                       
                    if start_scores[idx] > start_scores[idx + 1] and start_scores[idx] > start_scores[idx - 1]:
                        start_bins[idx] = 1
                    elif start_scores[idx] > (0.5 * max_start):
                        start_bins[idx] = 1

                end_bins = np.zeros(len(end_scores))
                end_bins[-1] = 1
                for idx in range(1, tscale - 1):
                    if end_scores[idx] > end_scores[idx + 1] and end_scores[idx] > end_scores[idx - 1]:
                        end_bins[idx] = 1
                    elif end_scores[idx] > (0.5 * max_end):
                        end_bins[idx] = 1
                ########################################################################################################

                #########################################################################
                # 
                new_props = []
                for idx in range(tscale):
                    for jdx in range(tscale):
                        start_index = jdx
                        end_index = start_index + idx+1
                        if end_index < tscale and start_bins[start_index] == 1 and end_bins[end_index] == 1:
                            xmin = start_index/tscale
                            xmax = end_index/tscale
                            xmin_score = start_scores[start_index]
                            xmax_score = end_scores[end_index]
                            clr_score = clr_confidence[idx, jdx]
                            reg_score = reg_confidence[idx, jdx]
                            score = xmin_score * xmax_score * clr_score*reg_score              
                            new_props.append([xmin, xmax, xmin_score, xmax_score, clr_score, reg_score, score])
                new_props = np.stack(new_props)
                #########################################################################

                col_name = ["xmin", "xmax", "xmin_score", "xmax_score", "clr_score", "reg_socre", "score"]
                new_df = pd.DataFrame(new_props, columns=col_name)
                new_df.to_csv("./output/BMN_results/" + video_name[ii] + ".csv", index=False)


def BMN_inference_ema(opt, eval_name):
    model = BMN(opt)
    model = torch.nn.DataParallel(model, device_ids=[0, 1, 2, 3]).cuda()
    model_checkpoint_dir = opt["checkpoint_path"] + eval_name   # BMN_checkpoint.pth.tar  BMN_best.pth.tar
    checkpoint = torch.load(model_checkpoint_dir)       # BMN_best.pth.tar
    print('load :', model_checkpoint_dir, ' OK !')
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()                                                 

    test_loader = torch.utils.data.DataLoader(VideoDataSet(opt, subset="validation"),            
                                              batch_size=8, shuffle=False,
                                              num_workers=8, pin_memory=True, drop_last=False)    
    tscale = opt["temporal_scale"]
    with torch.no_grad():
        for idx, input_data in test_loader:
            # set_trace()
            length = idx.shape[0]
            # for ii in range(length):
            video_name = []
            for ii in range(length):
                video_name_video = test_loader.dataset.video_list[idx[ii]]
                video_name.append(video_name_video)
            input_data = input_data.cuda()
            confidence_map, start, end = model(input_data)
            # set_trace()
            for ii in range(length):

                start_scores = start[ii].detach().cpu().numpy()
                end_scores = end[ii].detach().cpu().numpy()
                clr_confidence = (confidence_map[ii][1]).detach().cpu().numpy()
                reg_confidence = (confidence_map[ii][0]).detach().cpu().numpy()

                max_start = max(start_scores)
                max_end = max(end_scores)

                ####################################################################################################
                # generate the set of start points and end points
                start_bins = np.zeros(len(start_scores))
                start_bins[0] = 1  # [1,0,0...,0,1] 
                for idx in range(1, tscale - 1):                                       
                    if start_scores[idx] > start_scores[idx + 1] and start_scores[idx] > start_scores[idx - 1]:
                        start_bins[idx] = 1
                    elif start_scores[idx] > (0.5 * max_start):
                        start_bins[idx] = 1

                end_bins = np.zeros(len(end_scores))
                end_bins[-1] = 1
                for idx in range(1, tscale - 1):
                    if end_scores[idx] > end_scores[idx + 1] and end_scores[idx] > end_scores[idx - 1]:
                        end_bins[idx] = 1
                    elif end_scores[idx] > (0.5 * max_end):
                        end_bins[idx] = 1
                ########################################################################################################

                #########################################################################
                
                new_props = []
                for idx in range(tscale):
                    for jdx in range(tscale):
                        start_index = jdx
                        end_index = start_index + idx+1
                        if end_index < tscale and start_bins[start_index] == 1 and end_bins[end_index] == 1:
                            xmin = start_index/tscale
                            xmax = end_index/tscale
                            xmin_score = start_scores[start_index]
                            xmax_score = end_scores[end_index]
                            clr_score = clr_confidence[idx, jdx]
                            reg_score = reg_confidence[idx, jdx]
                            score = xmin_score * xmax_score * clr_score*reg_score              
                            new_props.append([xmin, xmax, xmin_score, xmax_score, clr_score, reg_score, score])
                new_props = np.stack(new_props)
                #########################################################################

                col_name = ["xmin", "xmax", "xmin_score", "xmax_score", "clr_score", "reg_socre", "score"]
                new_df = pd.DataFrame(new_props, columns=col_name)
                new_df.to_csv("./output/BMN_results/" + video_name[ii] + ".csv", index=False)


def main(opt):
    if opt["mode"] == "train":
        BMN_Train(opt)
    elif opt["mode"] == "inference":
        if not os.path.exists("output/BMN_results"):
            os.makedirs("output/BMN_results")
        print('unlabel percent: ', opt['unlabel_percent'])
        print('eval student model !!')
        for eval_name in ['/BMN_checkpoint.pth.tar', '/BMN_best.pth.tar']:
            BMN_inference(opt, eval_name)
            print("Post processing start")
            BMN_post_processing(opt)
            print("Post processing finished")
            evaluation_proposal(opt)
        print('eval teacher model !!')
        for eval_name in ['/BMN_checkpoint_ema.pth.tar', '/BMN_best_ema.pth.tar']:
            BMN_inference_ema(opt, eval_name)
            print("Post processing start")
            BMN_post_processing(opt)
            print("Post processing finished")
            evaluation_proposal(opt)


if __name__ == '__main__':
    opt = opts.parse_opt()
    opt = vars(opt)
    if not os.path.exists(opt["checkpoint_path"]):
        os.makedirs(opt["checkpoint_path"])
    if not os.path.exists('./output'):
        os.makedirs('./output')
    opt_file = open(opt["checkpoint_path"] + "/opts.json", "w")
    json.dump(opt, opt_file)
    opt_file.close()

    main(opt)
