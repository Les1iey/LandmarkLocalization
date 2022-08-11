import numpy as np
from scipy.ndimage.measurements import label as scipy_label
from scipy.ndimage.measurements import center_of_mass

def get_pre_coordinates(heatmaps,nums):
    out = np.zeros_like(heatmaps[0])
    #heatmaps[heatmaps < 0] = 0
    pred = np.zeros((nums, 2))

    for i in range(nums):
        heatmap = heatmaps[i]
        if np.max(heatmap) <= 0:
            #heatmap = -heatmap

            heatmap[heatmap < -0.1] = 0
            heatmap[heatmap != 0] = 1

            structure = np.ones((3, 3), dtype=np.int)  # 8连通域
            components_labels, ncomponents = scipy_label(heatmap, structure)

            count_max = 0
            label = 0
            for l in range(1, ncomponents + 1):
                component = (components_labels == l)
                count = np.count_nonzero(component)  # 某个残差块的大小

                if count > count_max:
                    count_max = count
                    label = l


            heatmap[components_labels != label] = 0
            heatmaps[i] = heatmap

            y_array, x_array = np.where(heatmap > 0.88 * np.max(heatmap)) #np.where(heatmap == np.max(heatmap))
            pred[i][0] = np.mean(x_array)  # .astype(int)
            pred[i][1] = np.mean(y_array)  # .astype(int)

        else:
            heatmaps[i][heatmaps[i] < 0.25 * np.max(heatmaps[i])] = 0
            structure = np.ones((3, 3), dtype=np.int)  # 8连通域
            components_labels, ncomponents = scipy_label(heatmap, structure)

            count_max = 0
            label = 0
            for l in range(1, ncomponents + 1):
                component = (components_labels == l)
                count = np.count_nonzero(component)  # 某个残差块的大小

                if count > count_max:
                    count_max = count
                    label = l

            count_1 = 0
            for l in range(1, ncomponents + 1):
                component = (components_labels == l)
                count = np.count_nonzero(component)  # 某个残差块的大小

                if count > count_max/5:
                    count_1+=1

            if count_1 > 1:
                print('存在两个',count_1,i)


            #print(count_max)
            heatmap[components_labels != label] = 0
            #out[heatmap > 0] = heatmap[heatmap > 0]
            heatmaps[i] = heatmap

            y_array, x_array = np.where(heatmap > 0.88 * np.max(heatmap))
            pred[i][0] = np.mean(x_array)  # .astype(int)
            pred[i][1] = np.mean(y_array)  # .astype(int

            heatmaps[i][heatmaps[i] < 0.88 * np.max(heatmaps[i])] = 0


        # pre_y, pre_x = np.where(heatmap == np.max(heatmap))
        # pred[i][1] = pre_y[0]
        # pred[i][0] = pre_x[0]

        # y_array, x_array = np.where(heatmap > 0.85 *np.max(heatmap))
        # pred[i][0] = np.mean(x_array)#.astype(int)
        # pred[i][1] = np.mean(y_array)#.astype(int)

    return heatmaps,pred
