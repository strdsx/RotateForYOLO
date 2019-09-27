from data_aug.data_aug import *
from data_aug.bbox_util import *
import cv2 
import pickle as pkl
import numpy as np 
import matplotlib.pyplot as plt
from multiprocessing import Process

# Reference : https://github.com/Paperspace/DataAugmentationForObjectDetection

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

def main(image_folder, start, end):    
    save_path = image_folder

    image_list = [x for x in os.listdir(image_folder) if x.split(".")[1] == "jpg"]

    for cnt, im in enumerate(image_list):
        if start < cnt < end:
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
                    transforms = Sequence([RandomRotate(15),
                        RandomHorizontalFlip(1),
                        RandomScale(0.2, diff=False)])

                    # transformed image
                    trans_img, trans_bboxes = transforms(img, bboxes)
                    tH, tW = trans_img.shape[:2]

                    # init save path & save image, txt files
                    save_img = save_path + "rot_" + im
                    save_txt = save_path + "rot_" + im.split(".")[0] + ".txt"

                    cv2.imwrite(save_img, trans_img)
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
                            # print(rcx, rcy, rw, rh, c_index)
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


if __name__ == '__main__':
    image_folder = "/home/jaehyeon/data/Drowner/drowner_train/"
    image_list = [x for x in os.listdir(image_folder) if x.endswith(".jpg")]
    image_num = len(image_list)
    split_num = 4
    proc_list = []

    for i in range(split_num):
        start = int(image_num * i / 4)
        end = int(image_num * (i + 1) / 4)
        proc = Process(target=main, args=(image_folder, start, end,))
        proc_list.append(proc)
        proc.start()

    for p in proc_list:
        p.join()