import cv2
import os
import sys
import numpy as np

if __name__ == "__main__":
    path = '/home/park/ORB_SLAM/myslam2/kitti08_mask/'
    dst = '/home/park/ORB_SLAM/myslam2/kitti08_mask.mp4'   

    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    sample = cv2.imread(path + os.listdir(path)[0])
    height, width, depth = sample.shape

    out = cv2.VideoWriter(dst, fourcc, 10.0, (width,height))
    if not out.isOpened():
        print('video writer error')
        sys.exit()

    img_array = []
    for i in range(1, len(os.listdir(path)) + 1):
        img_path = path + "masking{}.png".format(i)
        img = cv2.imread(img_path,cv2.IMREAD_GRAYSCALE)
        img = cv2.cvtColor(img , cv2.COLOR_GRAY2BGR)
        #img = cv2.imread(img_path)

        print(img_path, type(img), img.shape)
        img_array.append(img)
        # cv2.imshow('img', img)
        # cv2.waitKey()
        
        
    for i in range(len(img_array)):
       out.write(img_array[i])
    out.release()
    
    # while cnt<=len(os.listdir(path)):
    #     #img_path = os.path.join(path, f"masking{cnt}.png")
    #     img_path = path + "/masking{}.png".format(cnt)   cnt += 1  
    #     #print(img_path)   
    #     img = cv2.imread(img_path)
    #     cv2.imshow('img', img)
        
    #     #out.write(img)
    #     cnt += 1
    

    
    
    
    

    


    
