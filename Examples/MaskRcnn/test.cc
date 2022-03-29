#include <vector>
#include <fstream>
#include <sstream>
#include <iostream>
#include <string.h>
#include <string>
#include <unordered_set>

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/video/tracking.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>


cv::Mat DynamicExtract(const cv::Mat &im, cv::Mat &dynamic_mask,float confThreshold, float maskThreshold){

    std::vector<std::string> classes; // classId --> className
    std::unordered_set<std::string> dynamicClasses; // name of dynamic classes
    cv::dnn::Net net; // mask-rcnn model
    
    std::string strModelPath = "/home/park/ORB_SLAM/myslam2/ORB_SLAM2/Examples/MaskRcnn/pretrained/";
    // define inference parameter                                                                                                        
    std::string textGraph = strModelPath + "mask_rcnn_inception_v2_coco_2018_01_28.pbtxt";
    std::string modelWeights = strModelPath +  "mask_rcnn_inception_v2_coco_2018_01_28/frozen_inference_graph.pb";
    std::string classesFile = strModelPath + "mscoco_labels.names";
    std::string dynamicClassFile = strModelPath + "dynamic.names";

    // Load names of classes
    std::ifstream ifs(classesFile.c_str());
    std::string line;
    while (getline(ifs, line)) classes.push_back(line);

    // load names of dynamic classes
    std::ifstream ifs2(dynamicClassFile.c_str());
    while (getline(ifs2, line)) dynamicClasses.insert(line);
    
    // Load the network
    net = cv::dnn::readNetFromTensorflow(modelWeights, textGraph);
    net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
    cv::Mat blob;
    // Create 4D blob from a frame as input
    
    cv::dnn::blobFromImage(im, blob, 1.0, cv::Size(im.cols, im.rows), cv::Scalar(), true, false);
    net.setInput(blob);

    // Runs the forward pass to get output from the output layers
    std::vector<cv::String> outNames(2);
    outNames[0] = "detection_out_final";
    outNames[1] = "detection_masks";

    std::vector<cv::Mat> outs;
    net.forward(outs, outNames);
 
    cv::Mat outDetections = outs[0];
    cv::Mat outMasks = outs[1];

    // Output size of masks is NxCxHxW where
    // N - number of detected boxes
    // C - number of classes (excluding background)
    // HxW - segmentation shape
    const int numClasses = outMasks.size[1];
    const int numDetections = outDetections.size[2];
    outDetections = outDetections.reshape(1, outDetections.total() / 7);

    // aggregate binary mask of dynamic objects into dynamic_mask
    // dynamic part should be zero
    dynamic_mask = cv::Mat(im.size(), CV_8U, cv::Scalar(255));
    cv::Mat mat_zeros = cv::Mat::zeros(im.size(), CV_8U);
    for (int i = 0; i < numDetections; ++i) {
        float score = outDetections.at<float>(i, 2);
        if (score > confThreshold) {
            // Extract class id
            int classId = static_cast<int>(outDetections.at<float>(i, 1));

            // Extract bounding box
            int left = static_cast<int>(im.cols * outDetections.at<float>(i, 3));
            int top = static_cast<int>(im.rows * outDetections.at<float>(i, 4));
            int right = static_cast<int>(im.cols * outDetections.at<float>(i, 5));
            int bottom = static_cast<int>(im.rows * outDetections.at<float>(i, 6));
            
            left = std::max(0, std::min(left, im.cols - 1));
            top = std::max(0, std::min(top, im.rows - 1));
            right = std::max(0, std::min(right, im.cols - 1));
            bottom = std::max(0, std::min(bottom, im.rows - 1));
            cv::Rect box = cv::Rect(left, top, right - left + 1, bottom - top + 1);
            
            // Extract the mask for the object
            cv::Mat objectMask(outMasks.size[2], outMasks.size[3], CV_32F, outMasks.ptr<float>(i, classId));
            // Resize the mask, threshold, color and apply it on the image
            cv::resize(objectMask, objectMask, cv::Size(box.width, box.height));
            // threshold mask into binary 255/0 mask
            cv::Mat mask = (objectMask > maskThreshold);
            mask.convertTo(mask, CV_8U);

            if (dynamicClasses.count(classes[classId])) {
                // copy ones into the corresponding mask region
                mat_zeros(box).copyTo(dynamic_mask(box), mask);
            }
        }
    }
    return dynamic_mask;
}

int main(int argc, char* argv[]){

    std::string img_path = "/home/park/ORB_SLAM/myslam2/ORB_SLAM2/Examples/MaskRcnn/data/000030.png";
    cv::Mat dynamic_mask;
    cv::Mat result;
    cv::Mat frame = cv::imread(img_path, CV_LOAD_IMAGE_COLOR);
    std::cout<<frame.size()<<std::endl;

    result = DynamicExtract(frame, dynamic_mask,0.5,0.3);
    cv::imwrite("/home/park/ORB_SLAM/myslam2/ORB_SLAM2/Examples/MaskRcnn/result/test_mask.png", result);
    return 0;
}




