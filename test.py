import cv2
import numpy as np
import os
import xlwt
import SimpleITK as sitk
from config import Config
from utils import get_pre_coordinates
import matplotlib.pyplot as plt
from PIL import Image

Config = Config()


def visualise(img_file,gt,pre,save_file,MRE,nums):
    img = Image.open(img_file)
    img = np.array(img)
    plt.imshow(img, cmap='gray')
    for i in range(nums):
        x, y = gt[i]
        x1, y1 = pre[i]
        plt.scatter(x, y, color='blue', s=1)
        plt.scatter(x1, y1, color='lime', s=1)
        s = str(round(MRE,3))
        #plt.text(x + 30, y + 30, str(i + 1), fontsize=5, verticalalignment='top',horizontalalignment='center', color='red')
        plt.text(5, 50, s='MRE: '+ s, fontsize=10, color='red')

    plt.axis('off')
    plt.savefig(save_file,dpi = 500)
    plt.close()



def get_mre_sdr(model, test_loader, gt_dir, save_path,test_dir):
    nums = Config.num_classes
    err_max = np.zeros(nums)
    num_err_below_20 = np.zeros(nums)
    num_err_below_25 = np.zeros(nums)
    num_err_below_30 = np.zeros(nums)
    num_err_below_40 = np.zeros(nums)
    radial_error = np.zeros((len(test_loader), nums))

    for index, (img, heatmaps, img_name) in enumerate(test_loader):
        img = img.cuda(Config.GPU)
        outputs = model(img)
        outputs = outputs[0].cpu().detach().numpy()
        out = np.copy(outputs)
        outputs, pred = get_pre_coordinates(outputs,nums)

        # get gt coordinate

        gt_file = os.path.join(gt_dir,img_name[0].split('.')[0] + '.txt')
        gt_array = np.loadtxt(gt_file, delimiter=",")
        ratio = gt_array[-1][0] * gt_array[-1][1]

        #  start
        for j in range(nums):
            error = np.linalg.norm(gt_array[j] - pred[j]) * ratio
            radial_error[index, j] = error

            if error >= err_max[j]:
                err_max[j] = error

            if error <= 2:
                num_err_below_20[j] += 1
            elif error <= 2.5:
                num_err_below_25[j] += 1
            elif error <= 3:
                num_err_below_30[j] += 1
            elif error <= 4:
                num_err_below_40[j] += 1

            # save the huge error image and output
            if error >= 10:
                print('{}存在误差大于10mm的点:'.format(img_name[0]),  '{}号点'.format(j + 1), '误差为:{}mm'.format(error))

                #save the gt heatmap
                path_heatmap = os.path.join(Config.save_huge_path, img_name[0].split('.')[0] + '_hp' + '.nii.gz')
                sitk.WriteImage(sitk.GetImageFromArray(heatmaps[0].cpu().detach().numpy()), path_heatmap)

                # save the original outputs
                path_out = os.path.join(Config.save_huge_path, img_name[0].split('.')[0] + '_original_outputs' + '.nii.gz')
                sitk.WriteImage(sitk.GetImageFromArray(out), path_out)

                # save the post-processing outputs
                path_output = os.path.join(Config.save_huge_path, img_name[0].split('.')[0] + '_processing_outputs' + '.nii.gz')
                sitk.WriteImage(sitk.GetImageFromArray(outputs), path_output)


        single_err = np.mean(radial_error[index])
        print('图片', img_name[0].split('.')[0], '平均误差:', single_err)

        img_file = os.path.join(test_dir,img_name[0].split('.')[0] + '.bmp')
        reference_file = os.path.join(Config.reference_path,img_name[0].split('.')[0] + '_' + str(round(single_err, 3)) + '.png')
        visualise(img_file, gt_array, pred, reference_file,single_err,nums)

    radial_error_average = np.mean(radial_error)
    std = np.std(radial_error)
    print('测试集平均误差:',radial_error_average)

    num_err_below_25 = num_err_below_25 + num_err_below_20
    num_err_below_30 = num_err_below_30 + num_err_below_25
    num_err_below_40 = num_err_below_40 + num_err_below_30

    row0 = ['NO', '<=2mm', '<=2.5mm', '<=3mm', '<=4mm', 'mean_err','std', 'err_max']
    f = xlwt.Workbook()
    sheet1 = f.add_sheet('sheet1', cell_overwrite_ok=True)
    for i in range(0, len(row0)):
        sheet1.write(0, i, row0[i])
    for i in range(0, nums):
        sheet1.write(i + 1, 0, i + 1)
        sheet1.write(i + 1, 1, num_err_below_20[i] / (len(test_loader)) * 100)
        sheet1.write(i + 1, 2, num_err_below_25[i] / (len(test_loader)) * 100)
        sheet1.write(i + 1, 3, num_err_below_30[i] / (len(test_loader)) * 100)
        sheet1.write(i + 1, 4, num_err_below_40[i] / (len(test_loader)) * 100)
        sheet1.write(i + 1, 5, np.mean(radial_error[:, i]))
        sheet1.write(i + 1, 6, np.std(radial_error[:, i]))
        sheet1.write(i + 1, 7, err_max[i])

    sheet1.write(nums+1, 1, np.mean(num_err_below_20) / (len(test_loader)) * 100)
    sheet1.write(nums+1, 2, np.mean(num_err_below_25) / (len(test_loader)) * 100)
    sheet1.write(nums+1, 3, np.mean(num_err_below_30) / (len(test_loader)) * 100)
    sheet1.write(nums+1, 4, np.mean(num_err_below_40) / (len(test_loader)) * 100)
    sheet1.write(nums+1, 5, radial_error_average)
    sheet1.write(nums+1, 6, std)

    f.save(save_path)


def val_error(model, val_loader,nums):
    radial_error = np.zeros((len(val_loader), nums))
    huge_num = 0
    max = 0

    for index, (img, heatmaps, img_name) in enumerate(val_loader):
        img = img.cuda(Config.GPU)
        #print('图片', img_name[0])

        outputs = model(img)
        outputs = outputs[0].cpu().detach().numpy()

        _, pred_np_array = get_pre_coordinates(outputs,nums)

        cache_data_dir = os.path.join(Config.gt_dir)
        cached_average_path = os.path.join(cache_data_dir, img_name[0].split('.')[0] + ".txt")
        gt_array = np.loadtxt(cached_average_path, delimiter=",")
        ratio = gt_array[-1][0] * gt_array[-1][1]

        for i in range(nums):

            gt = gt_array[i]
            pred = pred_np_array[i]

            localisation_error = np.linalg.norm(pred - gt) *ratio

            if localisation_error > max:
                max = localisation_error

            #radial_error[int(img_name[0].split('.')[0])-151,i] = localisation_error
            radial_error[index, i] = localisation_error
            if localisation_error >= 100:
                huge_num += 1

    radial_error_average = np.mean(radial_error)
    #print('误差大于100的test1共有:',huge_num)
    #print('test1最大的误差为:',max)

    return radial_error_average