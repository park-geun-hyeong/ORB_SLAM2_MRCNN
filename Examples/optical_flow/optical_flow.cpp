#include<iostream>
#include<vector>
#include<string>
#include<opencv2/opencv.hpp>

using namespace std;
using namespace cv;

int main(){

    string vid_path = "/home/park/optical_flow/vtest.avi"; 
    VideoCapture capture(vid_path);
    if(!capture.isOpened()){
        cerr << "unable to open file" <<endl;
    }
    Mat frame, prvs, optical, drawing; 
    UMat flowMat;
    capture>>prvs;
    cvtColor(prvs, prvs, COLOR_BGR2GRAY);
    
    while(1){
        capture >> frame;
        
        if(frame.empty()){
            cout<<"empty frame"<<endl;
            break;
        }
        frame.copyTo(optical);
        frame.copyTo(drawing);
        cvtColor(optical, optical, COLOR_BGR2GRAY);
        
        calcOpticalFlowFarneback(prvs, optical, flowMat, 0.5, 3, 15, 3, 5, 1.2, 0);
        Mat flow;
        flowMat.copyTo(flow);
        
        for(int y = 0; y<optical.rows; y+=5){
            for(int x = 0; x<optical.cols; x+=5){
                cv::Point2f f = flow.at<cv::Point2f>(y,x);
                line(drawing, Point(x,y), Point(cvRound(x+f.x), cvRound(y+f.y)), Scalar(255,0,0));
                circle(drawing, Point(x,y),1, Scalar(0,0,0),-1);
            }
        }
        cv::imshow("frame", frame);
        cv::imshow("farnaback", drawing);

        prvs = frame;
        cvtColor(prvs, prvs , COLOR_BGR2GRAY);
        int keyboard = waitKey(30);
        if(keyboard == 'q' || keyboard == 27){break;}
    }
    return 0;
}