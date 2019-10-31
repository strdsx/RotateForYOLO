import Augmentor
import os
import random
import shutil

from data_aug.data_aug import *
from data_aug.bbox_util import *
import cv2 
import pickle as pkl
import numpy as np 
import matplotlib.pyplot as plt
import multiprocessing
from multiprocessing import Process
multiprocessing.set_start_method('spawn', True)

# Reference : https://github.com/Paperspace/DataAugmentationForObjectDetection
# Reference : https://github.com/mdbloice/Augmentor

def Augmentation_Operator(input_path, output_path, sample_num):
    # Pipe Init.
    ############################ Fix !!! ############################
    p = Augmentor.Pipeline(input_path, output_path)
    p.random_color(0.6, 0.2, 0.7)

    p.random_erasing(probability=0.3, rectangle_area=0.10000001)
    p.gaussian_distortion(probability=0.6, grid_width=3, grid_height=3, magnitude=3, corner="bell", method="in")
    p.random_distortion(probability=0.2, grid_width=2, grid_height=2, magnitude=2)
    p.histogram_equalisation(probability=0.1)
    p.random_brightness(probability=0.8, min_factor=0.3, max_factor=0.8)

    # Scale Transform
    p.scale(probability=0.7, scale_factor=1.5)
    p.scale(probability=0.1, scale_factor=1.3)

    p.sample(sample_num)
    ############################ Fix !!! ############################

# Augmentation
def augmentation(input_path, sample_number):
    aug_path = input_path + "_aug/" # <== Augmentation 이미지 저장 경로
    output_path = "../" + aug_path
    
    Augmentation_Operator(input_path, aug_path, sample_number)

    # Merge
    imglist = os.listdir(aug_path)

    cnt = 0

    # Augmentation img list.
    # Augmentation 완료 이후, 원본 이미지와 파일 이름 매칭 후,
    # 해당 txt 파일 가져와서 merge
    orig_folder = input_path.split("/")[-1]
    for i in imglist:
        tmp = i.split(orig_folder +"_original_")[1]
        i_num = tmp.split(".")[0]

        new_name = aug_path + str(cnt) + ".jpg"

        # GaussianBlur
        if cnt % 5 == 0:
            img = cv2.imread(aug_path + i)
            blur_img = cv2.GaussianBlur(img, (3, 3), 0)
            cv2.imwrite(aug_path + i, blur_img)

        # Rename augmentation image files.
        shutil.move(aug_path + i, new_name)

        # Get .txt files in the origin piap folder.
        shutil.copy(input_path + "/" + i_num + ".txt", aug_path + str(cnt) + ".txt")

        print("\tAugmentation ==> {}/{}".format(cnt, sample_number))

        cnt += 1


# to TL RB
def trans_bbox(bbox_info, origin_width, origin_height):
    return_list=[]
    for i in bbox_info:
        c_index = int(i[0])
        rcx = float(i[1])
        rcy = float(i[2])
        rw = float(i[3])
        rh = float(i[4])

        # to abs coordinate
        x1 = int((rcx - (rw/2))*origin_width)
        y1 = int((rcy - (rh/2))*origin_height)
        x2 = x1 + int(rw * origin_width)
        y2 = y1 + int(rh * origin_height)
        return_list.append([x1,y1,x2,y2,c_index])

    return return_list

def rotate_func(image_list, image_folder, start, end):    
    for im in image_list[start:end]:
        img_name = image_folder + im
        txt_name = image_folder + im.split(".")[0] + ".txt"

        # Read image
        img = cv2.imread(img_name) #OpenCV uses BGR channels
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        height, width = img.shape[:2]

        # Read txt for YOLO training
        with open(txt_name, "r") as f:
            lines = f.readlines()

        if len(lines) > 0:
            for i in range(len(lines)):
                lines[i] = lines[i].split("\n")[0]
                lines[i] = lines[i].split(" ")

            # transform bbox
            bboxes = trans_bbox(lines, width, height)
            bboxes = np.array(bboxes, dtype=np.float)
            
            try:
                transforms = Sequence([RandomRotate(15)])

                # transformed image
                trans_img, trans_bboxes = transforms(img, bboxes)
                tH, tW = trans_img.shape[:2]

                # init save path & save image, txt files
                save_img = image_folder + "rot_" + im
                save_txt = image_folder + "rot_" + im.split(".")[0] + ".txt"

                trans_img = cv2.cvtColor(trans_img, cv2.COLOR_RGB2BGR)
                cv2.imwrite(save_img, trans_img)
                print("\tRotated ==> ", save_img)
                with open(save_txt, "w") as f:
                    for i in trans_bboxes:
                        x1 = int(i[0])
                        y1 = int(i[1])
                        x2 = int(i[2])
                        y2 = int(i[3])

                        # check
                        '''
                        # trans_img = cv2.rectangle(trans_img, (x1, y1), (x2, y2), (0,255,0), 1)
                        crop_img = trans_img[y1:y2, x1:x2]
                        cv2.imshow("crop", crop_img)
                        cv2.waitKey()
                        cv2.destroyAllWindows()
                        '''

                        c_index = str(int(i[4])) # class index

                        # to relative coordinate for YOLO training
                        rcx = str(round(float(((x2 - x1) / 2 + x1) / tW), 6))
                        rcy = str(round(float(((y2 - y1) / 2 + y1) / tH), 6))
                        rw = str(round(float((x2-x1)/tW), 6))
                        rh = str(round(float((y2-y1)/tH), 6))

                        yolo_txt = c_index + " " + rcx + " " + rcy + " " + rw + " " + rh + "\n"
                        f.write(yolo_txt)
                # check
                '''
                # image show
                trans_img = cv2.resize(trans_img, dsize=(0,0),fx=4, fy=4)
                cv2.imshow("test", trans_img)
                cv2.waitKey()
                '''
            except OSError as err:
                print("error => ", err)

def make_train_txt():
    base_path = "/home/cvserver3/project/JH/Dataset/piap/"
    folder_list = ["LPC/", "LPC_aug/"]

    image_list = []
    for fol in folder_list:
        folder_path = base_path + fol

        img_list = [x for x in os.listdir(folder_path) if x.endswith("jpg")]

        for i in img_list:
            img_path = folder_path + i + "\n"
            image_list.append(img_path)
    
    # Suffle
    random.shuffle(image_list)
    
    total_num = len(image_list)
    train_num = int(total_num * 0.7)

    trainset = image_list[:train_num]
    validset = image_list[train_num:]

    print("\tNumber of train image : {}\n\tNumber of valid image : {}".format(len(trainset), len(validset)))
    
    # Write
    with open("191031_SLPCtrain.txt", "w") as f:
        for t in trainset:
            f.write(t)
    
    with open("191031_SLPCvalid.txt", "w") as f:
        for v in validset:
            f.write(v)



if __name__ == '__main__':
    '''
    # Augmentor
    origin_folder = "/home/cvserver3/project/JH/Dataset/piap/LPC"
    num_of_image = 1000000
    augmentation(origin_folder, num_of_image)

    # Rotation
    aug_folder = "/home/cvserver3/project/JH/Dataset/piap/LPC_aug/"
    image_list = [x for x in os.listdir(aug_folder) if x.endswith(".jpg")]
    image_num = len(image_list)
    split_num = 4
    proc_list = []

    for i in range(split_num):
        start = int(image_num * i / split_num)
        end = int(image_num * (i + 1) / split_num)
        proc = Process(target=rotate_func, args=(image_list, aug_folder, start, end,))
        proc_list.append(proc)
        proc.start()

    for p in proc_list:
        p.join()
    print("Finished rotate...")
    '''

    make_train_txt()
    print("Success make yolo train file")
