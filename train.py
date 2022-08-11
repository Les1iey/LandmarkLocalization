import torch
import time
import os
from dataset import LandmarkDataset
from config import Config
from test import val_error
from torch.utils.data import DataLoader
import SimpleITK as sitk
Config = Config()

def train(model, criterion,optimizer, scheduler, num_epochs=30, val = False):

    best_loss = 100
    best_error = 20
    loss_temp = 0

    # train dataset
    train_dataset = LandmarkDataset(Config.img_dir, Config.gt_dir,Config.height, Config.width, Config.num_classes,
                                    Config.sigma,Config.alpha)
    train_loader = DataLoader(dataset=train_dataset, batch_size=2, shuffle=True, num_workers=4)

    # val dataset
    if val:
        val_dataset = LandmarkDataset(Config.val_dir, Config.gt_dir, Config.height, Config.width, Config.num_classes,
                                      Config.sigma, Config.alpha)
        val_loader = DataLoader(dataset=val_dataset, batch_size=1, shuffle=False, num_workers=4)


    for epoch in range(0, num_epochs):
        epoch = epoch + 1
        print('Epoch{}/{}'.format(epoch, num_epochs))
        print('-' * 10)
        begin_t = time.time()
        model.train()

        for i, (img, heatmaps,_) in enumerate(train_loader):
            img = img.cuda(Config.GPU)
            heatmaps = heatmaps.cuda(Config.GPU)
            outputs = model(img)

            loss = criterion(outputs, heatmaps)
            loss_temp += loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()

        epoch_loss = loss_temp / len(train_loader)
        #  保存训练时loss最小的model
        if epoch >= 5 and epoch_loss < best_loss:
            best_loss = epoch_loss
            file_name = 'best_loss.pth'
            save_file_path = os.path.join(Config.save_model_path,file_name)
            torch.save(model.state_dict(), save_file_path)

        # 保存val中error最小的model
        if val and epoch >= 5:
            model.eval()
            val_err = val_error(model, val_loader, Config.num_classes)
            print('val上误差:', val_err)
            if val_err < best_error:
                best_error = val_err
                file_name = 'best_error.pth'
                save_file_path = os.path.join(Config.save_model_path, file_name)
                torch.save(model.state_dict(), save_file_path)


        tm = int(time.time() - begin_t)
        print('一个epoch的时间:',tm)

    return model
