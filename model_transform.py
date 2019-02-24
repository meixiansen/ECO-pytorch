import argparse
import os
import time
import shutil
import torch
import torchvision
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
from torch.nn.utils import clip_grad_norm_

from dataset import TSNDataSet
from models import TSN
from transforms import *
from opts import parser
import sys
import torch.utils.model_zoo as model_zoo
from torch.nn.init import constant_, xavier_uniform_

best_prec1 = 0

# python3 -u main.py ${dataset_name} RGB ${train_path} ${val_path} --arch ${netType} --num_segments ${num_segments} --gd 50 --lr ${learning_rate} --lr_steps 15 30 --
# epochs 40 -b ${batch_size} -i ${iter_size} -j ${num_workers} --dropout ${dropout} --snapshot_pref ${mainFolder}/${subFolder}/${snap_pref} --consensus_type identity
#  --eval-freq 1 --rgb_prefix 0 --pretrained_parts finetune --no_partialbn --nesterov "True" 2>&1 | tee -a ${mainFolder}/${subFolder}/training/log.txt

#global args, best_prec1
args = parser.parse_args()

print("------------------------------------")
print("Environment Versions:")
print("- Python: {}".format(sys.version))
print("- PyTorch: {}".format(torch.__version__))
print("- TorchVison: {}".format(torchvision.__version__))

args_dict = args.__dict__
print("------------------------------------")
print(args.arch + " Configurations:")
for key in args_dict.keys():
    print("- {}: {}".format(key, args_dict[key]))
print("------------------------------------")

if args.dataset == 'ucf101':
    num_class = 101
    rgb_read_format = "{:05d}.jpg"
elif args.dataset == 'hmdb51':
    num_class = 51
elif args.dataset == 'kinetics':
    num_class = 400
    rgb_read_format = "{:05d}.jpg"
elif args.dataset == 'something':
    num_class = 174
    rgb_read_format = "{:04d}.jpg"
else:
    raise ValueError('Unknown dataset ' + args.dataset)

model = TSN(num_class, args.num_segments, args.pretrained_parts, args.modality,
            base_model=args.arch,
            consensus_type=args.consensus_type, dropout=args.dropout, partial_bn=not args.no_partialbn)

checkpoint = torch.load('./logs/eco_lite_finetune_UCF101_rgb_epoch_40_checkpoint.pth.tar')

from collections import OrderedDict
new_state_dict = OrderedDict()
for k, v in checkpoint['state_dict'].items():
    name = k[7:] # remove `module.`
    new_state_dict[name] = v
# load params
model.load_state_dict(new_state_dict)
model.eval()

example = torch.rand(8, 3, 224, 224)

# Use torch.jit.trace to generate a torch.jit.ScriptModule via tracing.
traced_script_module = torch.jit.trace(model, example)

# ScriptModule
output = traced_script_module(torch.ones(8, 3, 224, 224))
traced_script_module.save('eco_finetune_ucf101.pt')
print(output)
