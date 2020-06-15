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

    Mat result = img.clone();

    cvtColor(img, img, COLOR_BGR2GRAY);
    //To store the keypoints that will be extracted by SIFT
    vector<KeyPoint> keypoints;

    int min_size = min(img.rows, img.cols);
    int step = 10;

    Rect window[7];

    for (int scale = 1; scale < 5; scale++) {
        int w_size = min_size / scale;

        for (int x = 0; x <= img.cols - w_size; x += step) {
            for (int y = 0; y <= img.rows - w_size; y += step) {

                int width = w_size/3;
                int height = w_size/2;

                window[0] = Rect(x, y, w_size, w_size);
                window[1] = Rect(0, 0, width, height);
                window[2] = Rect(width, 0, width, height);
                window[3] = Rect(width*2, 0, width, height);
                window[4] = Rect(0, height, width, height);
                window[5] = Rect(width, height, width, height);
                window[6] = Rect(width*2, height, width, height);

                //To store the BoW (or BoF) representation of the image
                Mat histogram[7];

                bool flag = false;

                for (int w=1; w<7; w++) {
                    //Detect SIFT keypoints (or feature points)
                    detector->detect(img(window[w]), keypoints);

                    //extract BoW (or BoF) descriptor from given image
                    bowDE.compute(img(window[w]), keypoints, histogram[w]);
                    if (histogram[w].empty()) {
                        histogram[w] = Mat::zeros(1, 200, CV_32F);
                        //flag = true;
                        //w = 20;
                    }
                }
                if (!flag) {
                    Mat tot_desc(1, 200*6, histogram[1].type());

                    for (int w=1; w<7; w++) {
                        histogram[w].copyTo(tot_desc(Rect((w-1)*200, 0, 200, 1)));
                    }
                    int response = (int) this->classifier->predict(tot_desc);

                    cout << x << "." << y << " result: " << response << endl;
                    if (response == 1) {
                        rectangle(result, window[0], {255, 0, 255}, scale);
                        //circle(result, Point(x+w_size/2, y+w_size/2), 5, (255, 0, 0), -1);
                        cout << "Te go visto nassare" << endl;
                    }
                }
                cout << x << "." << y << " result: " << flag << endl;

                Mat canvas = result.clone();

//                rectangle(canvas, window[0], { 0, 0, 255 }, 1);
                //circle(result, Point(x+w_size/2, y+w_size/2), 5, Scalar::all(-1));
                rectangle(canvas, window[0], { 0, 0, 255 });

                while (canvas.cols > 1500 || canvas.rows > 1000)
                    resize(canvas, canvas, Size(canvas.cols / 2, canvas.rows / 2));

                imshow("Window", canvas);

                waitKey(1);
            }
        }


    }

    return img;
}
