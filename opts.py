import os
import argparse


def parse():
    def get_degault_milistones(milistones, nEpochs):
        out = []
        for i in range(len(milistones)):
            out.append(int(milistones[i] * nEpochs))
        return out

    parser = argparse.ArgumentParser(description='BC learning for sounds')

    # General settings
    parser.add_argument('--dataset', required=True, choices=['esc10', 'esc50'])
    parser.add_argument('--netType', required=True, choices=['EnvNet', 'EnvNet2'])
    parser.add_argument('--data', required=True, help='Path to dataset')
    parser.add_argument('--split', type=int, default=-1, help='esc: 1-5, urbansound: 1-10 (-1: run on all splits)')
    parser.add_argument('--save', default='None', help='Directory to save the results')

    # Learning settings (default settings are defined below)
    parser.add_argument('--BC', action='store_true', help='BC learning')
    parser.add_argument('--strongAugment', action='store_true', help='Add scale and gain augmentation')
    parser.add_argument('--nEpochs', type=int, default=-1)
    parser.add_argument('--LR', type=float, default=-1, help='Initial learning rate')
    parser.add_argument('--milestones', type=float, nargs='*', default=-1, help='When decrease LR')
    parser.add_argument('--gamma', type=float, default=0.1, help='decreasing coeff')
    parser.add_argument('--batchSize', type=int, default=64)
    parser.add_argument('--weightDecay', type=float, default=5e-4)
    parser.add_argument('--nesterov', type=bool, default=True)
    parser.add_argument('--momentum', type=float, default=0.9)

    parser.add_argument('--nCrops', type=int, default=10)

    opt = parser.parse_args()

    # Dataset details
    if opt.dataset == 'esc50':
        opt.nClasses = 50
        opt.nFolds = 5
    elif opt.dataset == 'esc10':
        opt.nClasses = 10
        opt.nFolds = 5

    if opt.split == -1:
        opt.splits = range(1, opt.nFolds + 1)
    else:
        opt.splits = [opt.split]

    # Model details
    if opt.netType == 'EnvNet':
        opt.fs = 16000
        opt.inputLength = 24014
    elif opt.netType == 'EnvNet2':
        opt.fs = 44100
        opt.inputLength = 66650
    # Default settings (nEpochs will be doubled if opt.BC)
    default_settings = dict()
    default_settings['esc50'] = {'EnvNet': {'nEpochs': 600, 'LR': 0.01, 'milestones': [0.5, 0.75]},
                                 'EnvNet2': {'nEpochs': 1000, 'LR': 0.1, 'milestones': [0.3, 0.6, 0.9]}}
    default_settings['esc10'] = {'EnvNet': {'nEpochs': 600, 'LR': 0.01, 'milestones': [0.5, 0.75]},
                                 'EnvNet2': {'nEpochs': 600, 'LR': 0.01, 'milestones': [0.5, 0.75]}}

    for key in ['nEpochs', 'LR', 'milestones']:
        if eval('opt.{}'.format(key)) == -1:
            setattr(opt, key, default_settings[opt.dataset][opt.netType][key])
            if key == 'nEpochs' and opt.BC:
                opt.nEpochs *= 2
    opt.milestones = get_degault_milistones(opt.milestones, opt.nEpochs)
    if opt.save != 'None' and not os.path.isdir(opt.save):
        os.makedirs(opt.save)

    display_info(opt)

    return opt


def display_info(opt):
    if opt.BC:
        learning = 'BC'
    else:
        learning = 'standard'

    print('+------------------------------+')
    print('| Sound classification')
    print('+------------------------------+')
    print('| dataset  : {}'.format(opt.dataset))
    print('| netType  : {}'.format(opt.netType))
    print('| learning : {}'.format(learning))
    print('| augment  : {}'.format(opt.strongAugment))
    print('| nEpochs  : {}'.format(opt.nEpochs))
    print('| LRInit   : {}'.format(opt.LR))
    print('| batchSize: {}'.format(opt.batchSize))
    print('| nesterov   : {}'.format(opt.nesterov))
    print('| milestones : {}'.format(opt.milestones))
    print('| gamma : {}'.format(opt.gamma))
    print('+------------------------------+')
