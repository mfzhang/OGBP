#ifndef OGBP_H
#define OGBP_H


#include <cmath>
#include <set>
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include "opencv2/core/core.hpp"
#include "opencv2/contrib/contrib.hpp"


#include <cctype>
#include <iostream>
#include <iterator>
#include <stdio.h>

using namespace cv;
using namespace std;


Mat  myOGBP(Mat& src);


CV_EXPORTS_W Ptr<FaceRecognizer> createEigenFaceRecognizer(int num_components = 0, double threshold = DBL_MAX);






#endif // OGBP_H
