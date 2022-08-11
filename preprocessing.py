import os
import glob
import cv2
from PIL import Image
from tqdm import tqdm
from skimage import io
from skimage import img_as_ubyte
from imgaug.augmentables import Keypoint
from imgaug.augmentables import KeypointsOnImage
import numpy as np
import imgaug.augmenters as iaa
from config import Config

Config = Config()

def preprocessing(img_dir, annotation_dir):

    downsampled_image_width = Config.width  # 768
    downsampled_image_height = Config.height

    # Aspect ratio is defined as width / height
    downsampled_aspect_ratio = downsampled_image_width / downsampled_image_height

    # Define how to downsample and pad images
    preprocessing_steps = [
        iaa.PadToAspectRatio(downsampled_aspect_ratio, position='right-bottom'),  # 右下
        iaa.Resize({"width": downsampled_image_width, "height": downsampled_image_height}),
    ]
    seq = iaa.Sequential(preprocessing_steps)

    # establish the processing_dir
    img_dir_name = img_dir.split('/')[-1]
    processing_dir = os.path.join(Config.processing_dir,
                                  "{}_{}".format(downsampled_image_width, downsampled_image_height))
    img_resized_dir = os.path.join(processing_dir,img_dir_name)
    gt_resized_dir = os.path.join(processing_dir,'gt')

    if not os.path.exists(img_resized_dir):
        os.makedirs(img_resized_dir)
    if not os.path.exists(gt_resized_dir):
        os.makedirs(gt_resized_dir)

    # get the file names of all images in the directory

    image_paths = sorted(glob.glob(img_dir + "/*.bmp"))

    for image_path in tqdm(image_paths):

        # Get the file name with no extension
        file_name = os.path.basename(image_path).split(".")[0]

        # Get sub-directories for annotations
        annotation_sub_dirs = sorted(glob.glob(annotation_dir + "/*"))

        # Keep track of where we will be saving the downsampled image and the meta data
        cache_image_path = os.path.join(img_resized_dir, file_name + ".bmp")  # 依然保存为bmp
        cache_annotation_paths = []

        annotation_paths = []
        average_path = os.path.join(gt_resized_dir, file_name + ".txt")
        for annotation_sub_dir in annotation_sub_dirs:
            annotation_paths.append(os.path.join(annotation_sub_dir, file_name + ".txt"))
            cache_annotation_paths.append(os.path.join(gt_resized_dir, file_name + ".txt"))



        # Don't need to create them if they already exist
        if not os.path.exists(cache_image_path):

            # -----------Image-----------

            # Get image
            image = io.imread(image_path, as_gray=True)

            # Augment image
            image_resized = seq(image=image)

            # Save new image
            image_resized = np.clip(image_resized, 0.0, 1.0)  # 0-1之间
            image_as_255 = img_as_ubyte(image_resized)  # 0-255
            im = Image.fromarray(image_as_255)
            im.save(cache_image_path)

            # -----------Annotations-----------

            # Use pandas to extract the key points from the txt file
            average_txt = np.zeros((2, Config.num_classes, 2))
            gt_txt = np.zeros((Config.num_classes+1, 2))

            for i, annotation_path, cache_annotation_path in zip((0, 1), annotation_paths, cache_annotation_paths):
                # Get annotations
                kps_np_array = np.loadtxt(annotation_path, delimiter=",", max_rows=Config.num_classes)

                # Augment annotations
                kps = KeypointsOnImage.from_xy_array(kps_np_array, shape=image.shape)
                kps_resized = seq(keypoints=kps)

                # Save annotations
                kps_np_array = kps_resized.to_xy_array()  # (19, 2)
                average_txt[i] = kps_np_array
                np.savetxt(cache_annotation_path, kps_np_array, fmt="%.8g", delimiter=",")

            # average_txt[0][15] = average_txt[1][15]

            gt_txt[:-1] = np.mean(average_txt, axis=0)

            original_image_height, original_image_width = image.shape
            original_aspect_ratio = original_image_width / original_image_height

            if original_aspect_ratio > downsampled_aspect_ratio:
                scale_factor = original_image_width / downsampled_image_width
            else:
                scale_factor = original_image_height / downsampled_image_height

            gt_txt[-1] = [scale_factor, Config.physical_factor]

            np.savetxt(average_path, gt_txt, fmt="%.8f", delimiter=",")





def main(raw_dir,annotation_dir):

    files= os.listdir(raw_dir)
    for file in files:
        img_dir = os.path.join(raw_dir,file)
        preprocessing(img_dir,annotation_dir)

    if not os.path.exists(Config.save_model_path):
        os.makedirs(Config.save_model_path)

    if not os.path.exists(Config.save_results_path):
        os.makedirs(Config.save_results_path)

    if not os.path.exists(Config.save_huge_path):
        os.makedirs(Config.save_huge_path)

    if not os.path.exists(Config.reference_path):
        os.makedirs(Config.reference_path)

if __name__ == "__main__":
    raw_dir = Config.raw_dir
    annotation_dir = Config.annotation_dir
    main(raw_dir,annotation_dir)