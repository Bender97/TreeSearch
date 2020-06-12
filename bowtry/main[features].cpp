#include <vector>
#include <iostream>
#include <string>

#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>

using namespace std;
using namespace cv;

int canny_threshold;

Mat src;
vector<KeyPoint> kp;
Mat desc;
int flag = 1;

void onChange(int, void* info) {
    Mat img = *(Mat *) info;
    blur( img, img, Size(3,3) );

    Mat hsv_channels[3];
    cvtColor(img, img, COLOR_RGB2HSV);
    cv::split( img, hsv_channels );
    img = hsv_channels[2];

    if (canny_threshold<1) canny_threshold = 1;
    /// Canny detector
    Canny(img, img, canny_threshold,
          canny_threshold * 3, 3 );

    Ptr<Feature2D> f2d;
    f2d = xfeatures2d::SIFT::create();
    vector<KeyPoint> keypoints;
    Mat descriptors;
    f2d->detectAndCompute(img, Mat(), keypoints, descriptors);


    Mat hsv2_channels[3];
    if (flag) {
        cvtColor(src, src, COLOR_RGB2HSV);
        flag=0;
    }
    cv::split( src, hsv2_channels );
    Mat source = hsv2_channels[2];

    Canny(source, source, canny_threshold,
          canny_threshold * 3, 3 );
    f2d->detectAndCompute(source, Mat(), kp, desc);

    BFMatcher matcher = BFMatcher(NORM_L2);

    vector<DMatch> matches;
    matcher.match(descriptors, desc, matches);

    vector<Point2f> obj, scene;
    for (int j = 0; j < matches.size(); j++) {
        obj.push_back(keypoints[matches[j].queryIdx].pt);
        scene.push_back(kp[matches[j].trainIdx].pt);
    }
    Mat mask;
    findHomography(obj, scene, RANSAC, 2, mask);

    vector<DMatch> good_matches;

    for (int j = 0; j < matches.size(); j++) {
        if ((unsigned int) mask.at<uchar>(j)) {
            good_matches.push_back(matches[j]);
        }
    }

    Mat out;
    drawMatches(img, keypoints, source, kp, good_matches, out);

    while(out.rows>1000 || out.cols>1500)
        resize(out, out, Size(out.cols / 2, out.rows / 2));

    imshow("temp", out);
}

int main(int argc, const char* argv[])
{
    src = imread("../Benchmark_step_1/Figure2.jpg");
    canny_threshold = 24;

    for (int i=1; i<10; i++) {
        Mat img;
        img = imread("../dataset/0" + to_string(i) + ".jpg");
        if (img.empty())
            img = imread("../dataset/0" + to_string(i) + ".jpeg");
        while(img.rows>1000 || img.cols>1500)
            resize(img, img, Size(img.cols / 2, img.rows / 2));


        namedWindow("temp", WINDOW_AUTOSIZE);

        createTrackbar("canny thr", "temp", &canny_threshold, 256, onChange, &img);

        onChange(0, &img);

        waitKey(0);
        destroyAllWindows();
    }
    return 0;
}