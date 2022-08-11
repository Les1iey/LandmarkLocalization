import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
from config import Config
from dataset import LandmarkDataset
from model.model import Swin_transfusion
from model.model_confused import swin_fusion
from test import get_mre_sdr
from train import train
import argparse
Config = Config()


def get_args():
    parser = argparse.ArgumentParser()
    # optional
    parser.add_argument('--val', help='if validation',default= False)
    parser.add_argument('-p', "--phase", choices=['train', 'test', 'pretrained'], required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    if args.phase == 'train':
        # training setting
        model = Swin_transfusion()
        model.cuda(Config.GPU)
        criterion = nn.SmoothL1Loss(beta=1.5)
        criterion = criterion.cuda(Config.GPU)
        optimizer = torch.optim.Adam(model.parameters(), lr=Config.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.75, last_epoch=-1)

        # train
        val = args.val
        model_trained = train(model, criterion, optimizer, scheduler, Config.num_epochs,val)
        save_last_model_path = os.path.join(Config.save_model_path,'last.pth')
        torch.save(model_trained.state_dict(), save_last_model_path)


    elif args.phase == 'test':

        test1_dataset = LandmarkDataset(Config.test1_dir, Config.gt_dir, Config.height, Config.width,
                                        Config.num_classes,
                                        Config.sigma, Config.alpha)
        test1_loader = DataLoader(dataset=test1_dataset, batch_size=1, shuffle=False, num_workers=4)

        test2_dataset = LandmarkDataset(Config.test2_dir, Config.gt_dir, Config.height, Config.width,
                                        Config.num_classes,
                                        Config.sigma, Config.alpha)
        test2_loader = DataLoader(dataset=test2_dataset, batch_size=1, shuffle=False, num_workers=4)

        model = Swin_transfusion()
        model.cuda(Config.GPU)
        #model_test = torch.load(os.path.join(Config.save_model_path, 'last.pth') ,map_location={'cuda:1': 'cuda:{}'.format(Config.GPU)})  # ,map_location={'cuda:0': 'cuda:7'}
        state_dict = torch.load(os.path.join(Config.save_model_path, 'last.pth'))
        model.load_state_dict(state_dict)
        model.eval()
        get_mre_sdr(model, test1_loader, Config.gt_dir, os.path.join(Config.save_results_path, 'err_1.xls'),Config.test1_dir)
        get_mre_sdr(model, test2_loader, Config.gt_dir, os.path.join(Config.save_results_path, 'err_2.xls'),Config.test2_dir)



    elif args.phase == 'pretrained':

        test1_dataset = LandmarkDataset(Config.test1_dir, Config.gt_dir, Config.height, Config.width,
                                        Config.num_classes,
                                        Config.sigma, Config.alpha)
        test1_loader = DataLoader(dataset=test1_dataset, batch_size=1, shuffle=False, num_workers=4)

        test2_dataset = LandmarkDataset(Config.test2_dir, Config.gt_dir, Config.height, Config.width,
                                        Config.num_classes,
                                        Config.sigma, Config.alpha)
        test2_loader = DataLoader(dataset=test2_dataset, batch_size=1, shuffle=False, num_workers=4)

        model = swin_fusion()
        model.cuda(Config.GPU)
        state_dict = torch.load(os.path.join(Config.save_model_path, 'pretrained.pth'))
        model.load_state_dict(state_dict)
        model.eval()
        get_mre_sdr(model, test1_loader, Config.gt_dir, os.path.join(Config.save_results_path, 'err_1.xls'),Config.test1_dir)
        get_mre_sdr(model, test2_loader, Config.gt_dir, os.path.join(Config.save_results_path, 'err_2.xls'),Config.test2_dir)