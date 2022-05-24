import cv2
import numpy as np
import sys


if __name__ == "__main__":
    img_path = "/home/park/ORB_SLAM2/Examples/Stereo/dataset/sequences/00/image_0/"
    prvs = cv2.imread(img_path + "000000.png")
    prvs = cv2.cvtColor(prvs, cv2.COLOR_BGR2GRAY)

    cnt = 1
    while True:
        if cnt < 10:
            img_name = "0"*5 + str(cnt) + ".png"
        elif 10<=cnt<=99:
            img_name = "0"*4 + str(cnt) + ".png"
        elif 100<=cnt<=999:
            img_name = "0"*3 + str(cnt) + ".png"
        else:
            img_name = "0"*2 + str(cnt) + ".png"
        
        try:
            next = cv2.imread(img_path + img_name)
            next = cv2.cvtColor(next, cv2.COLOR_BGR2GRAY)
            draw_img = next.copy()
        except FileNotFoundError:
            print('file none exist!')
            sys.exit()

        flow = cv2.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        h,w = next.shape[:2]
        for y in range(0,h,15):
            for x in range(0,w,15):
                fx,fy = flow[y][x]
                cv2.line(draw_img, (x,y), (int(x+fx), int(y+fy)), (255,0,0))
                cv2.circle(draw_img, (x,y), 1, (0,0,0),-1)
        
        cv2.imshow("farneback", draw_img)
        cv2.imshow("frame", next)
        cv2.waitKey(30)
        prvs = next
        cnt += 1
        
        
    