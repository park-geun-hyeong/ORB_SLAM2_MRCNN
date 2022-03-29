#include<iostream>
#include<vector>
#include<opencv2/opencv.hpp>
#include<string>

using namespace std;
using namespace cv;

void show_img(Mat& img, Mat& Mask);
void keypoint_filter(Mat& img, Mat& Mask);
void DetectMasking(Mat& img, Mat& Mask);

Ptr<FeatureDetector> detector = ORB::create();

int main(int argc, char *argv[]){

    if(argc != 3){
        cerr<<"./build_file image_path masking_path"<<endl;
        exit(0);
        return -1;
    }

    Mat img = imread(argv[1], CV_LOAD_IMAGE_COLOR);
    Mat mask = imread(argv[2]);

    // Mat img = imread("./data/john_wick01.jpg",CV_LOAD_IMAGE_COLOR);
    // Mat mask = imread("./result/dynamic_mask.png");
    
    cout<< "show_img"<<endl;
    show_img(img, mask);

    cout<< "keypoint filtering"<<endl;
    keypoint_filter(img, mask);

    cout<<"DetectMasking"<<endl;
    DetectMasking(img,mask);

    return 0;
}


void show_img(Mat& img, Mat& Mask){
    Mat img_masked;
    img.copyTo(img_masked, Mask);

    imshow("img", img);
    imshow("Mask", Mask);
    imshow("masked_ img", img_masked);
    waitKey(0);
}

void keypoint_filter(Mat& img, Mat& Mask){
    
    vector<KeyPoint> keypoint1, keypoint2;
    Mat draw_img1, draw_img2, draw_masked;
    img.copyTo(draw_img1);
    img.copyTo(draw_img2);
    img.copyTo(draw_masked, Mask);

    detector->detect(img, keypoint1);
    detector->detect(img, keypoint2);
    drawKeypoints(draw_img1, keypoint1, draw_img1, Scalar(0,255,0), DrawMatchesFlags::DEFAULT);
    imshow("oridinal keypoint", draw_img1);

    KeyPointsFilter filter;
    filter.runByPixelsMask(keypoint1, Mask);
    filter.runByPixelsMask(keypoint2, 1-Mask);

    drawKeypoints(draw_img2, keypoint1, draw_img2, Scalar(0,255,0), DrawMatchesFlags::DEFAULT);
    drawKeypoints(draw_img2, keypoint2, draw_img2, Scalar(0,0,255), DrawMatchesFlags::DEFAULT);

    imshow("keypoint_filtered_img", draw_img2); 

    drawKeypoints(draw_masked, keypoint1, draw_masked, Scalar(0,255,0), DrawMatchesFlags::DEFAULT);
    drawKeypoints(draw_masked, keypoint2, draw_masked, Scalar(0,0,255), DrawMatchesFlags::DEFAULT);

    imshow("keypoint_filtered_img_masked", draw_masked); 
    waitKey(0);  
};

void DetectMasking(Mat& img, Mat& Mask){

    Mat draw_masked;
    Mat out_img;

    img.copyTo(draw_masked, Mask);  
   
    vector<KeyPoint> keypoint3,keypoint4;
    detector->detect(img, keypoint3, Mask);
    detector->detect(img, keypoint4, 1-Mask);

    cout<< img.size() <<", "<<Mask.size()<<endl;
    cout<<keypoint3.size() <<", " << keypoint4.size()<<endl;
    
    drawKeypoints(img, keypoint3, out_img, Scalar(255,0,0), DrawMatchesFlags::DEFAULT);
    drawKeypoints(out_img, keypoint4, out_img, Scalar(0,255,0), DrawMatchesFlags::DEFAULT);
    imshow("img1", out_img);

    drawKeypoints(draw_masked, keypoint3, out_img, Scalar(255,0,0), DrawMatchesFlags::DEFAULT);
    drawKeypoints(out_img, keypoint4, out_img, Scalar(0,255,0), DrawMatchesFlags::DEFAULT);
    imshow("img2", out_img);

    cv::waitKey(0);
}