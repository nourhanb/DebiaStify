from __future__ import print_function

import torch
import torch.nn.functional as F
import torch.nn as nn
import time

from torch import optim

from utils import get_accuracy
from trainer.kd_hinton import Trainer as hinton_Trainer
from trainer.loss_utils import compute_hinton_loss
from arguments import get_args

args = get_args()


class Trainer(hinton_Trainer):
    def __init__(self, args, **kwargs):
        super().__init__(args=args, **kwargs)
        self.lambh = args.lambh
        self.lambf = args.lambf
        self.sigma = args.sigma
        self.kernel = args.kernel
        self.jointfeature = args.jointfeature

    def train(self, train_loader, test_loader, epochs):

        num_classes = train_loader.dataset.num_classes
        num_groups = train_loader.dataset.num_groups
        best_acc = 0

        #distiller = MMDLoss(w_m=self.lambf, sigma=self.sigma,
        #                    num_classes=num_classes, num_groups=num_groups, kernel=self.kernel)
        distiller = KLLoss(w_m=self.lambf, num_classes=num_classes, num_groups=num_groups)

        for epoch in range(self.epochs):
            self._train_epoch(epoch, train_loader, self.model, self.teacher, distiller=distiller)
            eval_start_time = time.time()
            eval_loss, eval_acc, eval_deopp = self.evaluate(self.model, test_loader, self.criterion)
            eval_end_time = time.time()
            print('[{}/{}] Method: {} '
                  'Test Loss: {:.3f} Test Acc: {:.2f} Test DEopp {:.2f} [{:.2f} s]'.format
                  (epoch + 1, epochs, self.method,
                   eval_loss, eval_acc, eval_deopp, (eval_end_time - eval_start_time)))

            if eval_acc > best_acc:
                best_acc = eval_acc  # Update the best accuracy
                filename = f'best_student_model_acc_wearing_necklace_no_kl_{best_acc:.2f}.pth'
                torch.save(self.model.state_dict(), filename)
                print('Saved new best student model weights with accuracy: {:.2f}'.format(best_acc))

            if self.scheduler != None:
                self.scheduler.step(eval_loss)

        print('Training Finished!')

    def _train_epoch(self, epoch, train_loader, model, teacher, distiller=None):
        model.train()
        teacher.eval()
        init = False
        running_acc = 0.0
        running_loss = 0.0
        batch_start_time = time.time()

        for i, data in enumerate(train_loader):
            # Get the inputs
            inputs, _, groups, targets, _ = data
            labels = targets

            if self.cuda:
                inputs = inputs.cuda(self.device)
                labels = labels.cuda(self.device)
                groups = groups.long().cuda(self.device)
            t_inputs = inputs.to(self.t_device)

            outputs = model(inputs, get_inter=True)

            if args.model == 'resaligned':

                if init is False:
                    layer_list = []
                    teacher_feature_size = outputs[0].size(1)
                    student_feature_size = outputs[0].size(1)
                    layer_list.append(nn.Linear(student_feature_size, teacher_feature_size))
                    layer_list.append(nn.Linear(student_feature_size, teacher_feature_size))
                    layer_list.append(nn.Linear(student_feature_size, teacher_feature_size))
                    model.adaptation_layers = nn.ModuleList(layer_list)
                    model.adaptation_layers.cuda()
                    self.optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=5e-4, momentum=0.9)
                    init = True

            stu_logits = outputs[-1]
            t_outputs = teacher(t_inputs, get_inter=True)
            tea_logits = t_outputs[-1]
            kd_loss = compute_hinton_loss(stu_logits, t_outputs=tea_logits,
                                          kd_temp=self.kd_temp, device=self.device) if self.lambh != 0 else 0

            if args.model == 'resaligned':
                loss = self.criterion(stu_logits, labels) + self.criterion(outputs[-2], labels) + \
                   self.criterion(outputs[-3], labels) + self.criterion(outputs[-4], labels)
            else:
                loss = self.criterion(stu_logits, labels)

            loss =loss + self.lambh * kd_loss

            f_s1 = outputs[0]
            f_t1 = t_outputs[3]
            mmd_loss1 = distiller.forward(f_s1, f_t1, groups=groups, labels=labels, jointfeature=self.jointfeature)

            f_s2 = outputs[1]
            f_t2 = t_outputs[3]
            mmd_loss2 = distiller.forward(f_s2, f_t2, groups=groups, labels=labels, jointfeature=self.jointfeature)

            f_s3 = outputs[2]
            f_t3 = t_outputs[3]
            mmd_loss3 = distiller.forward(f_s3, f_t3, groups=groups, labels=labels, jointfeature=self.jointfeature)

            f_s4 = outputs[3]
            f_t4 = t_outputs[3]
            mmd_loss4 = distiller.forward(f_s4, f_t4, groups=groups, labels=labels, jointfeature=self.jointfeature)

            #loss = loss + mmd_loss1 + mmd_loss2 + mmd_loss3 + mmd_loss4
            loss = loss + mmd_loss2 
            running_loss += loss.item()
            running_acc += get_accuracy(stu_logits, labels)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if i % self.term == self.term - 1:  # print every self.term mini-batches
                avg_batch_time = time.time() - batch_start_time
                print('[{}/{}, {:5d}] Method: {} Train Loss: {:.3f} Train Acc: {:.2f} '
                      '[{:.2f} s/batch]'.format
                      (epoch + 1, self.epochs, i + 1, self.method, running_loss / self.term, running_acc / self.term,
                       avg_batch_time / self.term))

                running_loss = 0.0
                running_acc = 0.0
                batch_start_time = time.time()

        if not self.no_annealing:
            self.lambh = self.lambh - 3 / (self.epochs - 1)


class MMDLoss(nn.Module):
    def __init__(self, w_m, sigma, num_groups, num_classes, kernel):
        super(MMDLoss, self).__init__()
        self.w_m = w_m
        self.sigma = sigma
        self.num_groups = num_groups
        self.num_classes = num_classes
        self.kernel = kernel

    def forward(self, f_s, f_t, groups, labels, jointfeature=False):
        if self.kernel == 'poly':
            student = F.normalize(f_s.view(f_s.shape[0], -1), dim=1)
            teacher = F.normalize(f_t.view(f_t.shape[0], -1), dim=1).detach()
        else:
            student = f_s.view(f_s.shape[0], -1)
            teacher = f_t.view(f_t.shape[0], -1).detach()

        mmd_loss = 0

        if jointfeature:
            K_TS, sigma_avg = self.pdist(teacher, student,
                                         sigma_base=self.sigma, kernel=self.kernel)
            K_TT, _ = self.pdist(teacher, teacher, sigma_base=self.sigma, sigma_avg=sigma_avg, kernel=self.kernel)
            K_SS, _ = self.pdist(student, student,
                                 sigma_base=self.sigma, sigma_avg=sigma_avg, kernel=self.kernel)

            mmd_loss += K_TT.mean() + K_SS.mean() - 2 * K_TS.mean()

        else:
            with torch.no_grad():
                _, sigma_avg = self.pdist(teacher, student, sigma_base=self.sigma, kernel=self.kernel)

            for c in range(self.num_classes):
                if len(teacher[labels == c]) == 0:
                    continue
                for g in range(self.num_groups):
                    if len(student[(labels == c) * (groups == g)]) == 0:
                        continue
                    K_TS, _ = self.pdist(teacher[labels == c], student[(labels == c) * (groups == g)],
                                         sigma_base=self.sigma, sigma_avg=sigma_avg, kernel=self.kernel)
                    K_SS, _ = self.pdist(student[(labels == c) * (groups == g)], student[(labels == c) * (groups == g)],
                                         sigma_base=self.sigma, sigma_avg=sigma_avg, kernel=self.kernel)

                    K_TT, _ = self.pdist(teacher[labels == c], teacher[labels == c], sigma_base=self.sigma,
                                         sigma_avg=sigma_avg, kernel=self.kernel)

                    mmd_loss += K_TT.mean() + K_SS.mean() - 2 * K_TS.mean()

        loss = (1 / 2) * self.w_m * mmd_loss

        return loss

    @staticmethod
    def pdist(e1, e2, eps=1e-12, kernel='rbf', sigma_base=1.0, sigma_avg=None):
        if len(e1) == 0 or len(e2) == 0:
            res = torch.zeros(1)
        else:
            if kernel == 'rbf':
                e1_square = e1.pow(2).sum(dim=1)
                e2_square = e2.pow(2).sum(dim=1)
                prod = e1 @ e2.t()
                res = (e1_square.unsqueeze(1) + e2_square.unsqueeze(0) - 2 * prod).clamp(min=eps)
                res = res.clone()
                sigma_avg = res.mean().detach() if sigma_avg is None else sigma_avg
                res = torch.exp(-res / (2 * (sigma_base) * sigma_avg))
            elif kernel == 'poly':
                res = torch.matmul(e1, e2.t()).pow(2)

        return res, sigma_avg

class KLLoss(nn.Module):
    def __init__(self, w_m, num_groups, num_classes):
        super(KLLoss, self).__init__()
        self.w_m = w_m
        self.num_groups = num_groups
        self.num_classes = num_classes

    def forward(self, f_s, f_t, groups, labels, jointfeature=False):
        # Normalize the features
        student = F.log_softmax(f_s.view(f_s.shape[0], -1), dim=1)
        teacher = F.softmax(f_t.view(f_t.shape[0], -1), dim=1).detach()

        kl_loss = 0

        if jointfeature:
            # Check if both tensors have the same shape
            if student.size() == teacher.size():
                kl_loss = F.kl_div(student, teacher, reduction='batchmean')
            else:
                raise ValueError("Student and Teacher features must have the same shape for jointfeature=True")
        else:
            for c in range(self.num_classes):
                teacher_mask = labels == c
                teacher_features = teacher[teacher_mask]
                if teacher_features.size(0) == 0:
                    continue
                for g in range(self.num_groups):
                    student_mask = (labels == c) & (groups == g)
                    student_features = student[student_mask]
                    if student_features.size(0) == 0:
                        continue
                    # Check if both tensors have the same shape
                    if student_features.size(1) != teacher_features.size(1):
                        raise ValueError("Mismatch in feature dimensions between student and teacher")
                    # Resample student features to match the size of teacher features
                    if student_features.size(0) > teacher_features.size(0):
                        student_features = student_features[:teacher_features.size(0)]
                    elif student_features.size(0) < teacher_features.size(0):
                        teacher_features = teacher_features[:student_features.size(0)]
                    kl_loss += F.kl_div(student_features, teacher_features, reduction='batchmean')

        loss = self.w_m * kl_loss

        return loss