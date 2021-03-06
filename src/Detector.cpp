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
    Ptr<DescriptorMatcher> matcher(new FlannBasedMatcher);
    //create Sift feature point detector
    Ptr<xfeatures2d::SIFT> detector = xfeatures2d::SIFT::create();
    //create BoF (or BoW) descriptor extractor
    BOWImgDescriptorExtractor bowDE(detector, matcher);
    //Set the dictionary with the vocabulary we created in the first step
    bowDE.setVocabulary(this->vocabulary);

    Mat out_img = img.clone();

    cvtColor(img, img, COLOR_BGR2GRAY);
    //To store the keypoints that will be extracted by SIFT
    vector<KeyPoint> keypoints;

    int min_size = min(img.rows, img.cols);
    int step = 100;

    for (int scale = 1; scale < 3; scale++) {
        int w_size = min_size / scale;

        for (int x = 0; x <= img.cols - w_size; x += step) {
            for (int y = 0; y <= img.rows - w_size; y += step) {
                Rect window(x, y, w_size, w_size);

                //Detect SIFT keypoints (or feature points)
                detector->detect(img(window), keypoints);
                //To store the BoW (or BoF) representation of the image
                Mat histogram;
                //extract BoW (or BoF) descriptor from given image


                bowDE.compute(img, keypoints, histogram);

                Mat canvas = img.clone();
                Mat canvas_keys = img(window).clone();

                rectangle(canvas, window, { 0, 0, 255 });

                while (canvas.cols > 800 || canvas.rows > 600)
                    resize(canvas, canvas, Size(canvas.cols / 2, canvas.rows / 2));

                //imshow("canvas", canvas);

                if (!histogram.empty()) {

                    int response = (int) this->classifier->predict(histogram);

                    if (response == 1) {

                        drawKeypoints(canvas_keys, keypoints, canvas_keys, { 255, 0, 0 });
                        imshow("canvas_keys", canvas_keys);
                        waitKey(0);

                        rectangle(out_img, window, {0, 0, 255}, scale);
                        cout << "Te go visto nassare" << endl;
                    }
                }
                waitKey(1);
            }
        }


    }

    return out_img;
}
