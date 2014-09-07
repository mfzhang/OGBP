
#include "opencv2/core/core.hpp"
#include "opencv2/contrib/contrib.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include "OGBP.h"


#include <stdio.h>
#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>

#include "opencv2/opencv.hpp"

#include "detectObject.h"       // Easily detect faces or eyes (using LBP or Haar Cascades).
#include "preprocessFace.h"     // Easily preprocess face images, for face recognition.



using namespace cv;
using namespace std;

const char *faceCascadeFilename = "lbpcascade_frontalface.xml";     // LBP face detector.


const int faceWidth = 100;
const int faceHeight = faceWidth;


void initDetectors(CascadeClassifier &faceCascade)
{
        faceCascade.load(faceCascadeFilename);

}

void initWebcam(VideoCapture &videoCapture, int cameraNumber)
{
        videoCapture.open(cameraNumber);
}



static void read_csv(const string& filename, vector<Mat>& images, vector<int>& labels, char separator = ';') {
    std::ifstream file(filename.c_str(), ifstream::in);
    if (!file) {
        string error_message = "No valid input file was given, please check the given filename.";
        CV_Error(CV_StsBadArg, error_message);
    }
    string line, path, classlabel;
    while (getline(file, line)) {
        stringstream liness(line);
        getline(liness, path, separator);
        getline(liness, classlabel);
        if(!path.empty() && !classlabel.empty()) {
            images.push_back(imread(path, 0));
            labels.push_back(atoi(classlabel.c_str()));
        }
    }
}


/**********************video***********************************************/

void collectfromvideo()
{


    CascadeClassifier faceCascade;

    VideoCapture videoCapture;

    int cameraNumber = 0;

    initDetectors(faceCascade);
    initWebcam(videoCapture, cameraNumber);


      while(1) {

        Mat cameraFrame;
        Mat displayedFrame;

        videoCapture >> cameraFrame;
        cameraFrame.copyTo(displayedFrame);


        Mat preprocessedFace = getPreprocessedFace(displayedFrame, faceWidth, faceCascade);



        if (!preprocessedFace.empty())
        {

          imshow("test",preprocessedFace);
          waitKey(20);

          imwrite("test.jpg",preprocessedFace);
          //namedWindow("test");
          //imshow("test",preprocessedFace);
          //waitKey(20);

          cout << "collect ok !" << endl;

        }

    }

}

/**********************image***********************************************/


void collectfromimage(Mat& src)
{

    CascadeClassifier faceCascade;

    initDetectors(faceCascade);

    Mat preprocessedFace = getPreprocessedFace(src, faceWidth, faceCascade);

    imwrite("test.jpg", preprocessedFace );


}


/**********************retina***********************************************/

void writeretinaprocess(Mat& src)
{

     cv::Ptr<cv::Retina> myRetina;
     myRetina = new cv::Retina(src.size());
     cv::Mat retinaOutput_parvo;

     myRetina->run(src);
     myRetina->getParvo(retinaOutput_parvo);

     imwrite("retina.jpg",retinaOutput_parvo);

}

Mat retinaprocessout(Mat& src)
{

     cv::Ptr<cv::Retina> myRetina;
     myRetina = new cv::Retina(src.size());
     cv::Mat retinaOutput_parvo;

     myRetina->run(src);
     myRetina->getParvo(retinaOutput_parvo);

     //imwrite("retina.jpg",retinaOutput_parvo);
     return retinaOutput_parvo;
}


/**********************train and save***********************************************/


void trainandsave() {

    string fn_csv = string("a.txt");
    //cout << fn_csv << endl;
    //while(1);

    vector<Mat> images;
    vector<int> labels;

    try {
        read_csv(fn_csv, images, labels);
    } catch (cv::Exception& e) {
        cerr << "Error opening file \"" << fn_csv << "\". Reason: " << e.msg << endl;

        exit(1);
    }

    Ptr<FaceRecognizer> model0 = createEigenFaceRecognizer();

    model0->train(images, labels);
    cout << "train done "<< endl;

    model0->save("OGBPfaces_at.yml");


}



/**********************load and recognition***********************************************/


void loadandrecognition() {


    CascadeClassifier faceCascade;

    VideoCapture videoCapture;

    int count=0;
    int num=0;

    int cameraNumber = 0;
    initDetectors(faceCascade);
    initWebcam(videoCapture, cameraNumber);

    Ptr<FaceRecognizer> model = createEigenFaceRecognizer();
    model->load("OGBPfaces_at.yml");



      while(1) {

        Mat cameraFrame;
        Mat displayedFrame;

        videoCapture >> cameraFrame;
        cameraFrame.copyTo(displayedFrame);

        Mat preprocessedFace = getPreprocessedFace(displayedFrame, faceWidth, faceCascade);
        //cout << preprocessedFace.size() << endl;
        //Mat test = retinaprocessout(preprocessedFace);

        if(!preprocessedFace.empty()){

            int prediction = model->predict(preprocessedFace);

            num++;
            cout << num << endl;

            if(prediction==2){

                     count++;
                     //cout << "cyy" << endl;
            }


            string correctrate = format("correct rate  = %f ", (double)count/(double)num);
            cout << correctrate << endl;


        }

        if(num==500)
            break;

    }

}


/**********************main***********************************************/

int main(int argc, const char *argv[])
{

    string a = string(argv[1]);
    Mat src;

    if(a == "cv"){

             collectfromvideo();
    }
    else if(a == "ci"){

             src = imread(argv[2],1);
             collectfromimage(src);

    }
    else if(a == "p"){

             src = imread(argv[2],1);
             writeretinaprocess(src);

         }
    else if(a == "t"){

            trainandsave();
    }
    else if(a == "r"){

            loadandrecognition();
    }




    return 0;
}
