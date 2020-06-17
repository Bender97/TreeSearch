//
// Created by fusy on 14/06/20.
//

#include "../include/Detector.h"

void Detector::setVocabulary(Mat vocabulary) {
    this->vocabulary = vocabulary;
}

void Detector::setClassifier(Ptr<ml::SVM> &classifier) {
    this->classifier = classifier;
}

Mat Detector::detectTrees(Mat img) {
    //create a nearest neighbor matcher
    //Ptr<DescriptorMatcher> matcher(new FlannBasedMatcher);
    Ptr<DescriptorMatcher> matcher = BFMatcher::create(NORM_L2);
    //create Sift feature point detector
    Ptr<xfeatures2d::SIFT> detector = xfeatures2d::SIFT::create();
    //create BoF (or BoW) descriptor extractor
    BOWImgDescriptorExtractor bowDE(detector, matcher);
    //Set the dictionary with the vocabulary we created in the first step
    bowDE.setVocabulary(this->vocabulary);

    Mat result = img.clone();

    cvtColor(img, img, COLOR_BGR2GRAY);
    //To store the keypoints that will be extracted by SIFT
    vector<KeyPoint> keypoints;

    int min_size = min(img.rows, img.cols);
    int step = 10;

    /*** WINDOWING PARAMETERS ***/
    int rows = 2;
    int cols = 3;
    int exclude_big_image = 1; // 1 true, 0 false [include also full image histogram computation]
    /*** VOCABULARY PARAMETERS ***/
    int num_bins = vocabulary.rows;

    vector<Rect> windows;
    Mat histogram[rows*cols + 1];

    for (int scale = 1; scale < 5; scale++) {
        int w_size = min_size / scale;

        for (int x = 0; x <= img.cols - w_size; x += step) {
            for (int y = 0; y <= img.rows - w_size; y += step) {

                windows = getFrames(img, rows, cols, x, y, w_size);

                bool flag = false;

                for (int w=exclude_big_image; w<windows.size(); w++) {
                    //Detect SIFT keypoints (or feature points)
                    detector->detect(img(windows[w]), keypoints);

                    //extract BoW (or BoF) descriptor from given image
                    bowDE.compute(img(windows[w]), keypoints, histogram[w]);
                    if (histogram[w].empty()) {
                        histogram[w] = Mat::zeros(1, num_bins, CV_32F);
                        /*w = windows.size();
                        flag = true;*/
                    }
                }
                if (!flag) {
                    Mat tot_desc(1, num_bins*(windows.size()-exclude_big_image), CV_32F);

                    for (int w=exclude_big_image; w<windows.size(); w++) {
                        histogram[w].copyTo(tot_desc(Rect((w-exclude_big_image)*num_bins, 0, num_bins, 1)));
                    }
                    int response = (int) this->classifier->predict(tot_desc);

                    cout << x << "." << y << " result: " << response << endl;
                    if (response == 1) {
                        rectangle(result, windows[0], {255, 0, 255}, scale);
                        //circle(result, Point(x+w_size/2, y+w_size/2), 5, (255, 0, 0), -1);
                        cout << "Te go visto nassare" << endl;
                    }
                }
                cout << x << "." << y << " flag: " << flag << endl;

                Mat canvas = result.clone();

//                rectangle(canvas, window[0], { 0, 0, 255 }, 1);
                //circle(result, Point(x+w_size/2, y+w_size/2), 5, Scalar::all(-1));
                rectangle(canvas, windows[0], { 0, 0, 255 });

                while (canvas.cols > 1500 || canvas.rows > 1000)
                    resize(canvas, canvas, Size(canvas.cols / 2, canvas.rows / 2));

                imshow("Window", canvas);

                waitKey(1);
            }
        }


    }

    return img;
}
