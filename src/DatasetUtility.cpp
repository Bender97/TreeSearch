//
// Created by cogny on 14/06/20.
//

#include "../include/DatasetUtility.h"

void buildTrainingSet(string input_images_path, Mat vocabulary, string output_CSVs_path, float proportion)
{

    // load images from tree and non_tree directories
	vector<string> tree_images_paths = loadImagesPaths(input_images_path + TREE_DIR);
    vector<string> non_tree_images_paths = loadImagesPaths(input_images_path + NON_TREE_DIR);

    cout << "tree images: " << tree_images_paths.size() << endl;
    cout << "non tree images: " << non_tree_images_paths.size() << endl;

    int n_tree = (int)tree_images_paths.size();
    int n_non_tree = (int)non_tree_images_paths.size();

    int n_images = n_tree + n_non_tree;

    vector<pair<string, int>> images_paths(n_images);

    // push all the images in the general vector together with their class
    for (int i = 0; i < n_tree; i++)
    {
        images_paths[i] = pair<string, int>(tree_images_paths[i], TREE_CLASS);
    }
    for (int i = 0; i < n_non_tree; i++)
    {
        images_paths[i + n_tree] = pair<string, int>(non_tree_images_paths[i], NON_TREE_CLASS);
    }

    auto rng = default_random_engine(random_device{}());
    std::shuffle(begin(images_paths), end(images_paths), rng);

    //create a nearest neighbor matcher
    //Ptr<DescriptorMatcher> matcher(new FlannBasedMatcher);
    Ptr<DescriptorMatcher> matcher = BFMatcher::create(NORM_L2);
    //create Sift feature point detector
    Ptr<xfeatures2d::SIFT> detector = xfeatures2d::SIFT::create();
    //create BoF (or BoW) descriptor extractor
    BOWImgDescriptorExtractor bowDE(detector, matcher);
    //Set the dictionary with the vocabulary we created in the first step
    bowDE.setVocabulary(vocabulary);

    ofstream train_set_f = prepareCSV(output_CSVs_path + TRAINING_SET_DIR);
    ofstream test_set_f  = prepareCSV(output_CSVs_path + TEST_SET_DIR);

    //To store the keypoints that will be extracted by SIFT
    vector<KeyPoint> keypoints;

    /*** WINDOWING PARAMETERS ***/
    int rows = 2;
    int cols = 3;
    int exclude_big_image = 1; // 1 true, 0 false [include also full image histogram computation]
    /*** VOCABULARY PARAMETERS ***/
    int num_bins = vocabulary.rows;

    vector<Rect> windows;
    Mat histogram[rows*cols + 1];

    for (int i = 0; i < n_images; i++)
    {
        Mat img = imread(images_paths[i].first);
        int img_class = images_paths[i].second;

        windows = getWindows(img, rows, cols);

        //To store the BoW (or BoF) representation of the image

        cvtColor(img, img, COLOR_BGR2GRAY);

        bool flag = false;

        for (int w=exclude_big_image; w<windows.size(); w++) {
            //Detect SIFT keypoints (or feature points)
            detector->detect(img(windows[w]), keypoints);

            //extract BoW (or BoF) descriptor from given image
            bowDE.compute(img(windows[w]), keypoints, histogram[w]);
            if (histogram[w].empty()) {
                //histogram[w] = Mat::zeros(1, 200, CV_32F);
                w = windows.size();
                flag = true;
            }
        }
        if (!flag) {
            Mat tot_desc(1, num_bins*(windows.size()-exclude_big_image), CV_32F);

            for (int w=exclude_big_image; w<windows.size(); w++) {
                histogram[w].copyTo(tot_desc(Rect((w-exclude_big_image)*num_bins, 0, num_bins, 1)));
            }

            // insert the first n_images * proportion images into the train set, the rest into the test set
            if (i < n_images * proportion) {
                addRowCSV(train_set_f, tot_desc, img_class);
            } else {
                addRowCSV(test_set_f, tot_desc, img_class);
            }
        }

        cout << to_string((int)((float)(i + 1) / n_images * 100)) << "%" << endl;
    }
    
    train_set_f.close();
    test_set_f.close();
}

pair<Mat, Mat> loadDataset(string dataset_path)
{

    vector<pair<Mat, int>> dataset = readCSV(dataset_path);

    int n_samples = (int)dataset.size();
    int n_bags = dataset[0].first.cols;

    Mat samples(n_samples, n_bags, dataset[0].first.type());
    Mat labels(n_samples, 1, CV_32SC1);

    for (int i = 0; i < n_samples; i++)
    {
        // select a row of the matrix and copy it to the samples matrix
        Rect row(0, i, n_bags, 1);
        dataset[i].first.copyTo(samples(row));
        // copy the class to the labels matrix
        labels.at<int>(0, i) = dataset[i].second;
        
    }

	return pair<Mat, Mat>(samples, labels);
}