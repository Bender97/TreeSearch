#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/core/utils/filesystem.hpp>
#include <fstream>

#include "../include/FilesUtility.h"
#include "../include/ImagesUtility.h"
#include "../include/VocabularyUtility.h"

using namespace std;
using namespace cv;

vector<Mat> loadImages(cv::String images_path) {


    vector<String> filenames;   // vector of all the filenames found in the specified folder
    vector<string> allowedExtensions = { ".jpg", ".jpeg", ".png", ".bmp" };    // allowed extensions for images

    /// for each extension, search for images in the specified folder
    for (int i = 0; i < allowedExtensions.size(); i++) {
        vector<String> currentFilenames;    // filenames found with the i-th extension

        /// look for matches with the pattern
        cv::glob(images_path + "*" + allowedExtensions[i],
                 currentFilenames, true);

        /// insert found filenames in the general vector
        filenames.insert(
                filenames.end(),
                currentFilenames.begin(),
                currentFilenames.end());
    }

    if (filenames.empty()) cout << "No images found. Exiting." << endl;

    vector<Mat> images;

    /// load the images corresponding to each filename
    for (int i = 0; i < filenames.size(); i++)
    {
        Mat img;
        img = imread(filenames[i]);
        while (img.cols > 800 || img.rows > 600)
            resize(img, img, Size(img.cols / 2, img.rows / 2));
        images.push_back(img);
    }

    return images;
}

Mat createDictionary(String samples_path)
{
    //Step 1 - Obtain the set of bags of features.

    //to store the input file names

    vector<Mat> sample_imgs = loadImages(samples_path);

    //to store the current input image
    Mat input;
    //To store the keypoints that will be extracted by SIFT
    vector<KeyPoint> keypoints;
    //To store the SIFT descriptor of current image
    Mat descriptor;
    //To store all the descriptors that are extracted from all the images.
    Mat featuresUnclustered;
    //The SIFT feature extractor and descriptor
    Ptr<xfeatures2d::SIFT> detector = xfeatures2d::SIFT::create();

    //I select 20 (1000/50) images from 1000 images to extract
    //feature descriptors and build the vocabulary
    for (int f = 0; f < sample_imgs.size(); f++) {

        cvtColor(sample_imgs[f], input, COLOR_BGR2GRAY);
        //detect feature points
        detector->detect(input, keypoints);
        //compute the descriptors for each keypoint
        detector->compute(input, keypoints, descriptor);
        //put the all feature descriptors in a single Mat object
        featuresUnclustered.push_back(descriptor);
        //print the percentage
        printf("%i percent done\n", (int)((float)f / sample_imgs.size() * 100));
    }

    //Construct BOWKMeansTrainer
    //the number of bags
    int dictionarySize = 200;
    //define Term Criteria
    TermCriteria tc(TermCriteria::Type::MAX_ITER, 100, 0.001);
    //retries number
    int retries = 1;
    //necessary flags
    int flags = KMEANS_PP_CENTERS;
    //Create the BoW (or BoF) trainer
    BOWKMeansTrainer bowTrainer(dictionarySize, tc, retries, flags);
    //cluster the feature vectors
    Mat dictionary = bowTrainer.cluster(featuresUnclustered);

    return dictionary;
}

void detectTrees(String test_path, Mat dictionary)
{
    //Step 2 - Obtain the BoF descriptor for given image/video frame.

    //create a nearest neighbor matcher
    Ptr<DescriptorMatcher> matcher(new FlannBasedMatcher);
    //create Sift feature point detector
    Ptr<xfeatures2d::SIFT> detector = xfeatures2d::SIFT::create();
    //create BoF (or BoW) descriptor extractor
    BOWImgDescriptorExtractor bowDE(detector, matcher);
    //Set the dictionary with the vocabulary we created in the first step
    bowDE.setVocabulary(dictionary);

    vector<Mat> test_imgs = loadImages(test_path);

    ofstream file;
    file.open("dataset.csv");

    for (int idx=0; idx < test_imgs.size(); idx++) {
        //read the image
        Mat img;
        cvtColor(test_imgs[idx], img, COLOR_BGR2GRAY);
        //To store the keypoints that will be extracted by SIFT
        vector<KeyPoint> keypoints;


        int min_size = min(img.rows, img.cols);
        int step = 50;

        //FileStorage fs1("descriptor.yml", FileStorage::WRITE);
        //prepare the yml (some what similar to xml) file

        int cont = 0;

        for (int scale = 1; scale < 5; scale++) {
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

                    file << "img" << to_string(cont) << ",";

                    for (int word = 0; word < histogram.cols; word++) {
                        float value = histogram.at<float>(0, word);
                        file << value << ",";
                    }
                    file << endl;

                    cout << window << endl;

                    cont++;

                    Mat canvas = img.clone();

                    rectangle(canvas, window, { 0, 0, 255 });

                    imshow("Window", canvas);

                    waitKey(1);

                }
            }
        }
    }
    file.close();

}

int main(int argc, const char* argv[])
{
    /*
    String samples_path = "../dataset/";
    String test_path = "../Benchmark_step_1/";

    Mat dictionary = createDictionary(samples_path);
    detectTrees(test_path, dictionary);*/


    
    return 0;
}


