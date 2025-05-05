from torch.utils.data import DataLoader, Dataset
from datetime import datetime
import random, torch, torchvision, cv2, glob, json, os, collections
import numpy as np
from PIL import Image


def set_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def load_model(args, net):
    params = torch.load(args.load_path)['state_dict']
    net.load_state_dict(params)
    print('Model loaded successfully!')


def set_save_path(args):
    os.makedirs(args.save_path, exist_ok=True)
    with open(os.path.join(args.save_path, 'args.txt'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)
        print('save_path: ' + str(args.save_path) + '. Joined successfully!')
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')