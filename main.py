from data_aug.data_aug import *
from data_aug.bbox_util import *
import cv2 
import pickle as pkl
import numpy as np 
import matplotlib.pyplot as plt

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

def main():
    get_number = 250000
    image_count = 0

    image_folder = "img_bolt_aug/"

    image_list = [x for x in os.listdir(image_folder) if x.split(".")[1] == "jpg"]

    for im in image_list:
        img_name = image_folder + im
        txt_name = image_folder + im.split(".")[0] + ".txt"

        # Read image
        img = cv2.imread(img_name)[:,:,::-1] #OpenCV uses BGR channels
        height, width = img.shape[:2]

        # Read txt for YOLO training
        with open(txt_name, "r") as f:
            lines = f.readlines()
            
        for i in range(len(lines)):
            lines[i] = lines[i].split("\n")[0]
            lines[i] = lines[i].split(" ")

        # transform bbox
        bboxes = trans_bbox(lines, width, height)
        bboxes = np.array(bboxes, dtype=np.float)

        transforms = Sequence([RandomRotate(24)])

        # transformed image
        trans_img, trans_bboxes = transforms(img, bboxes)
        tH, tW = trans_img.shape[:2]

        # init save path & save image, txt files
        save_img = "img_rotate/rotate_" + im
        save_txt = "img_rotate/rotate_" + im.split(".")[0] + ".txt"

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
        print("\t ===> Rotated {} / {}".format(image_count, (get_number-1)))
        image_count += 1
        if image_count == (get_number - 1):
            break


if __name__ == '__main__':
    main()