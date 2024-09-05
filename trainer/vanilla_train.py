from __future__ import print_function

import time
import torch
from torch import optim
from arguments import get_args
from utils import get_accuracy
import trainer
import torch.nn as nn

args = get_args()
model_name = args.model


class Trainer(trainer.GenericTrainer):
    def __init__(self, args, **kwargs):
        super().__init__(args=args, **kwargs)

    def train(self, train_loader, test_loader, epochs):
        model = self.model
        model.train()
        best_acc = 0
        for epoch in range(epochs):
            self._train_epoch(epoch, train_loader, model)

            eval_start_time = time.time()
            eval_loss, eval_acc, eval_deopp = self.evaluate(model, test_loader, self.criterion)
            eval_end_time = time.time()
            print('[{}/{}] Method: {} '
                  'Test Loss: {:.3f} Test Acc: {:.2f} Test DEopp {:.2f} [{:.2f} s]'.format
                  (epoch + 1, epochs, self.method,
                   eval_loss, eval_acc, eval_deopp, (eval_end_time - eval_start_time)))
            if eval_acc > best_acc:
                best_acc = eval_acc  # Update the best accuracy
                filename = f'best_model_acc_wearing_necklace_{best_acc:.2f}.pth'
                torch.save(model.state_dict(), filename)
                print('Saved new best model weights with accuracy: {:.2f}'.format(best_acc))

            if self.scheduler != None and 'Multi' not in type(self.scheduler).__name__:
                self.scheduler.step(eval_loss)
            else:
                self.scheduler.step()

        print('Training Finished!')

    def _train_epoch(self, epoch, train_loader, model):
        model.train()
        init = False
        running_acc = 0.0
        running_loss = 0.0

        batch_start_time = time.time()
        for i, data in enumerate(train_loader):
            # Get the inputs
            inputs, _, groups, targets, _ = data

            labels = targets

            if self.cuda:
                inputs = inputs.cuda(device=self.device)
                labels = labels.cuda(device=self.device)

            if model_name == 'resaligned':
                outputs = model(inputs, get_inter=True)
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
                loss = self.criterion(outputs[-1], labels) + self.criterion(outputs[-2], labels) \
                       + self.criterion(outputs[-3], labels) + self.criterion(outputs[-4], labels)
                running_loss += loss.item()
                running_acc += get_accuracy(outputs[-1], labels)

            else:
                outputs = model(inputs)
                loss = self.criterion(outputs, labels)
                running_loss += loss.item()
                running_acc += get_accuracy(outputs, labels)

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
