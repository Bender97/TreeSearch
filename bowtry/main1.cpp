#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <fstream>
#include <dirent.h>
#include <sys/stat.h>

using namespace std;
using namespace cv;

int main() {
    string dir = "../dataset/";
    string filepath;

    DIR *dp;
    dp = opendir(dir.c_str());

    struct dirent *dirp;
    struct stat filestat;

    //detecting keypoints
    Ptr<DescriptorExtractor> extractor = xfeatures2d::SIFT::create();
    vector<KeyPoint> keypoints;
    Mat descriptors;
    Mat training_descriptors(1, extractor->descriptorSize(), extractor->descriptorType());
    Mat img;

    cout << "------- build vocabulary ---------" << endl;
    cout << "extract descriptors.." << endl;

    int count = 0;

    while(count++ < 15 && (dirp=readdir(dp))) {
        filepath = dir + dirp->d_name;
        if (stat( filepath.c_str(), &filestat )) continue;
        if (S_ISDIR( filestat.st_mode )) continue;

        img = imread(filepath);
        extractor->detectAndCompute(img, Mat(), keypoints, descriptors);
        training_descriptors.push_back(descriptors);
        cout << ".";

        //drawKeypoints(img, keypoints, img, Scalar::all(-1));
        //imshow("temp", img);
        //waitKey(0);

    }

    cout << endl;
    closedir(dp);

    cout << "Total descriptors: " << training_descriptors.rows << endl;

    BOWKMeansTrainer bowtrainer(150);

    bowtrainer.add(training_descriptors);

    cout << "cluster BOW features" << endl;
    Mat vocabulary = bowtrainer.cluster();

    Ptr<DescriptorMatcher> matcher(new BFMatcher(NORM_L2));

    BOWImgDescriptorExtractor bowide(extractor, matcher);

    bowide.setVocabulary(vocabulary);
    //setup training data for classifiers
    map<string,Mat> classes_training_data; classes_training_data.clear();


    Mat response_hist;
    char buf[255];
    ifstream ifs("../training.txt");

    do {
        ifs.getline(buf, 255);
        string line(buf);
        istringstream iss(line);
        cout << line << endl;

        iss >> filepath;

        string class_;
        iss >> class_;
        cout << filepath << " " << class_ << endl;

        img = imread(filepath);
        char c__[] = {(char)atoi(class_.c_str()),'\0'};
        string c_(c__);

        bowide.compute(img, keypoints, response_hist);
        if(classes_training_data.count(c_) == 0) { //not yet created...
            classes_training_data[c_].create(0,response_hist.cols,response_hist.type());
        }
        classes_training_data[c_].push_back(response_hist);

    } while(!ifs.eof());

    //train 1-vs-all SVMs
    Ptr<ml::SVM> classes_classifiers = ml::SVM::create();

    for (int i=0;i<2;i++) {
        string class_ = to_string(i);
        cout << "training class: " << class_ << ".." << endl;

        Mat samples(0,response_hist.cols,response_hist.type());
        Mat labels(0,1,CV_32FC1);

        //copy class samples and label
        samples.push_back(classes_training_data[class_]);
        Mat class_label = Mat::ones(classes_training_data[class_].rows, 1, CV_32FC1);
        labels.push_back(class_label);

        //copy rest samples and label
        for (map<string,Mat>::iterator it1 = classes_training_data.begin(); it1 != classes_training_data.end(); ++it1) {
            string not_class_ = (*it1).first;
            if(not_class_.compare(class_)==0) continue;
            samples.push_back(classes_training_data[not_class_]);
            class_label = Mat::zeros(classes_training_data[not_class_].rows, 1, CV_32FC1);
            labels.push_back(class_label);
        }

        cout << "Train.." << endl;
        Mat samples_32f; samples.convertTo(samples_32f, CV_32F);
        Ptr<ml::SVM> classifier = ml::SVM::create();
        classifier->train(samples_32f, labels);


    }

    return 0;
}
