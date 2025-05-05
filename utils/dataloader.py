from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from datetime import datetime
import torchvision, glob
import numpy as np
import operators.single_coil_mri as mrimodel

import operators.operator as op

class CustomMRI(Dataset):
    def __init__(self, data_path, transform=None):
        self.data_path = data_path
        self.transform = transform

        file_list = glob.glob(data_path + "*")
        self.data = []
        for single_data_path in file_list:
            self.data.append(single_data_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        container = np.load(self.data[idx], allow_pickle=True)
        e = {name: container[name] for name in container}['arr_0'].item()
        X, y = e['X'].squeeze(), e['y'].squeeze()
        return X, y

class CustomTumorMRI(Dataset):
    def __init__(self, data_path, transform=None):
        self.data_path = data_path
        self.transform = transform

        file_list = glob.glob(data_path + "*")
        self.data = []
        for single_data_path in file_list:
            self.data.append(single_data_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = torchvision.io.read_image(self.data[idx])
        img = op.normalize(img, 1)
        if self.transform:
            img = self.transform(img)
        return img


class CustomSeisDeconv(Dataset):
    def __init__(self, data_path, transform=None):
        self.data_path = data_path
        self.transform = transform

        file_list = glob.glob(data_path + "*")
        self.data = []
        for single_data_path in file_list:
            self.data.append(single_data_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        container = np.load(self.data[idx], allow_pickle=True)
        e = {name: container[name] for name in container}['arr_0'].item()
        X, y = e['refl'], e['trace']
        return X, y


def load_data(args):
    timeStamp = datetime.now().strftime("%m%d-%H%M")
    args.save_path += args.file_name + timeStamp
    if args.train:
        if args.train_mode == 'AT':
            args.save_path += '_epsilon=' + str(args.epsilon)
        elif args.train_mode == 'input-jitter':
            args.save_path += '_sigma_W=' + str(args.sigma_w)
        elif args.train_mode == 'sgd-jitter':
            args.save_path += '_sigma_Wk=' + str(args.sigma_wk)

    if not args.train:
        args.save_path += '_val_mode_' + args.test_mode

    if args.dataset == 'MRI':
        transform = transforms.Compose([transforms.ToTensor()])
        train_dataset = CustomMRI(args.data_path['train'], transform)
        val_dataset = CustomMRI(args.data_path['val'], transform)
        test_dataset = CustomMRI(args.data_path['test'], transform)

        tr_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
        val_loader = DataLoader(dataset=val_dataset, batch_size=args.batch_size_val, shuffle=False, drop_last=True)
        ts_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size_val, shuffle=False, drop_last=True)

        return tr_loader, len(train_dataset), val_loader, len(val_dataset), ts_loader, len(test_dataset)

    elif args.dataset == 'seis':
        transform = transforms.Compose([transforms.ToTensor()])
        train_dataset = CustomSeisDeconv(args.data_path['train'], transform)
        val_dataset = CustomSeisDeconv(args.data_path['val'], transform)
        test_dataset = CustomSeisDeconv(args.data_path['test'], transform)

        tr_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)
        val_loader = DataLoader(dataset=val_dataset, batch_size=args.batch_size_val, shuffle=True, drop_last=False)
        ts_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size_val, shuffle=False, drop_last=False)

        return tr_loader, len(train_dataset), val_loader, len(val_dataset), ts_loader, len(test_dataset)


def load_mri_knee_tumor(args):
    """ data is obtained from https://radiopaedia.org/cases/giant-cell-tumour-knee """
    load_path = '../datasets/knee_tumor/'
    transform = transforms.Compose([transforms.Resize((320, 320))])
    dataset = CustomTumorMRI(load_path, transform)
    dataloader = DataLoader(dataset=dataset, batch_size=args.batch_size_val, shuffle=False, drop_last=True)
    return dataloader, len(dataset)