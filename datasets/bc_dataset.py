import os
import numpy as np
import torch
import random
import datasets.transforms as T
import datasets.utils as U

class BCDatasets(torch.utils.data.Dataset):
    def __init__(self, sounds, labels, opt, train = True):
        self.X = sounds
        self.Y = labels
        self.opt = opt
        self.transform = T.Transform(self.opt)
        self.train = train
        self.mix = (opt.BC and train)
        self.preprocess_funcs = self.preprocess_setup()

    def __len__(self):
        #'Denotes the total number of samples'
        return len(self.Y)

    #poreprocess
    def preprocess_setup(self):
        if self.train:
            funcs = []
            if self.opt.strongAugment:
                funcs += [U.random_scale(1.25)]

            funcs += [U.padding(self.opt.inputLength // 2),
                      U.random_crop(self.opt.inputLength),
                      U.normalize(32768.0),
                      ]

        else:
            funcs = [U.padding(self.opt.inputLength // 2),
                     U.normalize(32768.0),
                     U.multi_crop(self.opt.inputLength, self.opt.nCrops),
                     ]

        return funcs

    def preprocess(self, sound):
        for f in self.preprocess_funcs:
            sound = f(sound)

        return sound

    def __getitem__(self, idx):
        if self.mix:  # Training phase of BC learning
            # Select two training examples
            while True:
                indice1 = random.randint(0, len(self.X) - 1)
                indice2 = random.randint(0, len(self.X) - 1)
                sound1 = self.X[indice1]
                label1 = self.Y[indice1]
                sound2 = self.X[indice2]
                label2 = self.Y[indice2]
                if label1 != label2:
                    break

            sound1 = self.preprocess(sound1)
            sound2 = self.preprocess(sound2)

            # Mix two examples
            r = np.array(random.random())
            sound = U.mix(sound1, sound2, r, self.opt.fs).astype(np.float32)
            eye = np.eye(self.opt.nClasses)
            label = (eye[label1] * r + eye[label2] * (1 - r)).astype(np.float32)

        else:  # Training phase of standard learning or testing phase
            sound = self.X[idx]
            label_num = self.Y[idx]
            eye = np.eye(self.opt.nClasses).astype(np.float32)
            sound = self.preprocess(sound).astype(np.float32)
            #label = np.zeros(self.opt.nClasses).astype(np.int32)
            label = eye[label_num]


        if self.train and self.opt.strongAugment:
            sound = U.random_gain(6)(sound).astype(np.float32)

        return sound, label

def get_data_generators(opt, test_fold):
    dataset = np.load(os.path.join(opt.data, opt.dataset, 'wav{}.npz'.format(opt.fs // 1000)), allow_pickle=True)

    # Split to train and val
    train_sounds = []
    train_labels = []
    val_sounds = []
    val_labels = []
    for i in range(1, opt.nFolds + 1):
        sounds = dataset['fold{}'.format(i)].item()['sounds']
        labels = dataset['fold{}'.format(i)].item()['labels']
        if i == test_fold:
            val_sounds.extend(sounds)
            val_labels.extend(labels)
        else:
            train_sounds.extend(sounds)
            train_labels.extend(labels)

    # Iterator setup
    train_data = BCDatasets(train_sounds, train_labels, opt, train=True)
    val_data = BCDatasets(val_sounds, val_labels, opt, train=False)
    train_iter = torch.utils.data.DataLoader(train_data, batch_size = opt.batchSize, shuffle=True)
    val_iter = torch.utils.data.DataLoader(val_data, batch_size = opt.batchSize//opt.nCrops, shuffle=False)

    return train_iter, val_iter
