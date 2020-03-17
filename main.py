import os
import numpy as np
import torch
from draw_process import *
from torch.optim import SGD
from torch.optim.lr_scheduler import MultiStepLR
import models
import opts
from datasets import *

def create_model(opt):
    if opt.netType == 'EnvNet':
        model = models.EnvNet(opt.nClasses)
    else:
        model = models.EnvNet2(opt.nClasses)
    return model

def create_optimizer(model, opt):
    opt_params = {"weight_decay": opt.weightDecay,
                  "momentum": opt.momentum,
                  "nesterov": opt.nesterov,
                  "lr": opt.LR}
    return SGD(model.parameters(), **opt_params)

def kldiv_loss(input, target):
    logsoftmax = torch.nn.LogSoftmax(dim=1)
    idx = torch.nonzero(target.data).split(1, dim = 1)
    entropy = - torch.sum(target[idx] * torch.log(target[idx]))
    crossEntropy = - torch.sum(target * logsoftmax(input))

    return (crossEntropy - entropy) / input.shape[0]

def crossentropy_loss(input, target):
    logsoftmax = torch.nn.LogSoftmax(dim=1)
    crossEntropy = - torch.sum(target * logsoftmax(input))
    return crossEntropy / input.shape[0]

if __name__ == '__main__':
    opt = opts.parse()

    global_train_loss = []
    global_val_loss = []
    global_train_error = []
    global_val_error = []
    n_folds = opt.nFolds

    for test_fold in opt.splits:
        print('+------------------------------+')
        print("test validation is {}".format(test_fold))
        print('+------------------------------+')

        device = torch.device("cuda:0" if torch.cuda.is_available else "cpu")
        print("{} is used.".format(device))

        model = create_model(opt)
        model.to(device)
        optimizer = create_optimizer(model, opt)
        scheduler = MultiStepLR(optimizer, milestones=opt.milestones, gamma=0.1)

        train_loss = []
        train_error_rate = []
        val_loss = []
        val_error_rate = []

        training_generator, validation_generator = get_data_generators(opt, test_fold)
        print("Data Get fold{}".format(test_fold))

        for epoch in range(opt.nEpochs):
            model.train()
            running_loss = 0.0
            for i, data in enumerate(training_generator, 0):
                input_array, label_array = data[0], data[1]
                inputs = input_array[:,None,None,:].to(device)
                labels = label_array.to(device)

                optimizer.zero_grad()

                outputs = model(inputs)
                if opt.BC:
                    loss = kldiv_loss(outputs, labels)
                else:
                    loss = crossentropy_loss(outputs, labels)

                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            model.eval()
            eval_loss = 0.0
            correct = 0.0
            total = 0.0
            with torch.no_grad():
                for data in validation_generator:
                    input_array, label_array = data[0], data[1]
                    input_array = input_array.reshape(input_array.shape[0]*opt.nCrops, input_array.shape[2])
                    inputs = input_array[:,None,None,:].to(device)
                    labels = label_array.to(device)
                    outputs = model(inputs)

                    outputs_total = outputs.data.reshape(outputs.shape[0] // opt.nCrops, opt.nCrops, outputs.shape[1])
                    outputs_total = torch.mean(outputs_total,1)

                    _, answer = torch.max(labels, 1)
                    _, predicted = torch.max(outputs_total, 1)
                    total += labels.size(0)
                    correct += (predicted == answer).sum().item()

                    loss = crossentropy_loss(outputs_total.data, labels.data)
                    eval_loss += loss.item()

            scheduler.step()
            running_loss = running_loss/len(training_generator)
            eval_loss = eval_loss/len(validation_generator)

            print("Epoch [%d/%d], train_loss: %.4f, val_loss: %.4f, Accuracy: %.4f" % (epoch + 1,opt.nEpochs ,running_loss, eval_loss , 100 * correct / total))
            train_loss.append(running_loss)
            val_loss.append(eval_loss)
            val_error_rate.append(100 - 100 * correct / total)

        print ("Finished Training")

        if opt.save != "None":
            draw_progress(train_loss, val_loss, val_error_rate, opt.nEpochs, "validation_{}".format(test_fold), opt.save)
            torch.save(model.state_dict(), os.path.join(opt.save, "model_{}.bin".format(test_fold)))
