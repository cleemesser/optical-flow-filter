/**
 * \file flowWebCam.cpp
 * \brief Optical flow demo using OpenCV VideoCapture to compute flow from webcam.
 * \copyright 2015, Juan David Adarve, ANU. See AUTHORS for more details
 * \license 3-clause BSD, see LICENSE for more details
 */

#include <iostream>
#include <vector>

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/types_c.h>

#include <flowfilter/gpu/flowfilter.h>
#include <flowfilter/gpu/display.h>

using namespace std;
using namespace cv;
using namespace flowfilter;
using namespace flowfilter::gpu;


void wrapCVMat(Mat& cvMat, image_t& img) {

    img.height = cvMat.rows;
    img.width = cvMat.cols;
    img.depth = cvMat.channels();
    img.pitch = cvMat.cols*cvMat.elemSize();
    img.itemSize = cvMat.elemSize1();
    img.data = cvMat.ptr();
}

/**
 * MODE OF USE
 * ./processVideo  <video filename>
 *
 *
 */
int main(int argc, char** argv) {
    string video_file = string("/home/clee/code/eegml/data/goodvideo_lpch/myoclonic-atonic-vid02.mp4");
    
    // int cameraIndex = 0;
    //string video_file = string("/home/clee/code/eegml/data/goodvideo_lpch/myoclonic-atonic-vid02.mp4");

    // if user provides camera index
    if(argc > 1) {
      // cameraIndex = atoi(argv[1]);
      video_file = argv[1];
    } else {
      video_file = "/home/clee/code/eegml/data/goodvideo_lpch/myoclonic-atonic-vid02.mp4";
    }
    
    // VideoCapture cap(cameraIndex); // open the default camera
    //VideoCapture cap("/home/clee/code/eegml/data/goodvideo_lpch/myoclonic-atonic-vid02.mp4"); // open a file
    cout << "try to open: " << video_file << endl;
    VideoCapture cap(video_file); // open a file
    if(!cap.isOpened()){
      cout << "Could not open " << video_file << endl;
        return -1;
    }
    
    Mat frame;

    //  captura a frame to get image width and height
    cap >> frame;
    int width = frame.cols;
    int height = frame.rows;
    cout << "frame shape: [" << height << ", " << width << "]" << endl;

    Mat frameGray(height, width, CV_8UC1);
    Mat fcolor(height, width, CV_8UC4);

    // structs used to wrap cv::Mat images and transfer to flowfilter
    image_t hostImageGray;
    image_t hostFlowColor;

    wrapCVMat(frameGray, hostImageGray);
    wrapCVMat(fcolor, hostFlowColor);
    

    //#################################
    // Filter parameters
    //#################################
    float maxflow = 40.0f;
    vector<float> gamma = {500.0f, 50.0f, 5.0f};
    vector<int> smoothIterations = {2, 8, 20};

    //#################################
    // Filter creation with
    // 3 pyramid levels
    //#################################
    PyramidalFlowFilter filter(height, width, 3);
    filter.setMaxFlow(maxflow);
    filter.setGamma(gamma);
    filter.setSmoothIterations(smoothIterations);

    //#################################
    // To access optical flow
    // on the host
    //#################################
    Mat flowHost(height, width, CV_32FC2);
    image_t flowHostWrapper;
    wrapCVMat(flowHost, flowHostWrapper);
    

    // Color encoder connected to optical flow buffer in the GPU
    FlowToColor flowColor(filter.getFlow(), maxflow);

    int ret =0;
    // Capture loop
    for(;;) {

        // capture a new frame from the camera
        // and convert it to gray scale (uint8)
        // cap >> frame;
      ret = cap.read(frame);
      if (!ret) {
	cout << "breaking out\n";
	break;
      }
        cvtColor(frame, frameGray, CV_BGR2GRAY);
        
        // transfer image to flow filter and compute
        filter.loadImage(hostImageGray);
        filter.compute();

        // cout << "elapsed time: " << filter.elapsedTime() << " ms" << endl;

        // transfer the optical flow from GPU to
        // host memory allocated by flowHost cvMat.
        // After this, optical flow values
        // can be accessed using OpenCV pixel
        // access methods.
        filter.downloadFlow(flowHostWrapper);

        // computes color encoding (RGBA) and download it to host
        flowColor.compute();
        flowColor.downloadColorFlow(hostFlowColor);
        cvtColor(fcolor, fcolor, CV_RGBA2BGRA);
	// cvtColor(fcolor, fcolor, RGBA2BGRA);//nope

        imshow("image", frameGray);
        imshow("optical flow", fcolor);

        if ( (waitKey(10) & 0xFF) == 27) break; // allows time for fetch of events so images are displayed

    }
    cout << "finished\n";
    // waitKey(10000000);
    // the camera will be deinitialized automatically in VideoCapture destructor
    return 0;
}
