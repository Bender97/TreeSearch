//
// Created by cogny on 14/06/20.
//

#include "../include/DatasetUtility.h"

void buildTrainingSet(string input_images_path, Mat vocabulary, string output_CSVs_path, float proportion)
{

    cout << "Loading images filename from root: " << input_images_path << endl;
    // load images from tree and non_tree directories
    vector<string> tree_images_paths;
    vector<string> maybe_tree_images_paths;
    vector<string> non_tree_images_paths;

    try {
        tree_images_paths = loadImagesPaths(input_images_path + TREE_DIR);
        maybe_tree_images_paths = loadImagesPaths(input_images_path + MAYBE_TREE_DIR);
        non_tree_images_paths = loadImagesPaths(input_images_path + NON_TREE_DIR);
    }
	catch (cv::Exception& e) {
	    cout << "Error loading images! Make sure the folder substructure is:" << endl
	        << "+ " << input_images_path << endl
	        << "--- + " << TREE_DIR << endl
	        << "--- + " << MAYBE_TREE_DIR << endl
	        << "--- + " << NON_TREE_DIR << endl;
	    return;
	}

	cout << "Images filenames correcly loaded! I've found:" << endl;

    int n_tree = (int)tree_images_paths.size();
    int n_maybe_tree = (int)maybe_tree_images_paths.size();
    int n_non_tree = (int)non_tree_images_paths.size();

    cout << "tree images: " << n_tree << endl;
    cout << "maybe tree images: " << n_maybe_tree << endl;
    cout << "non tree images: " << n_non_tree << endl << endl;

    cout << "Starting building the training and test sets" << endl;

    int n_images = n_tree + n_maybe_tree + n_non_tree;

    vector<pair<string, int>> images_paths(n_images);

    // push all the images in the general vector together with their class
    
    int j = 0;
    for (int i = 0; i < n_tree; i++, j++)
    {
        images_paths[j] = pair<string, int>(tree_images_paths[i], TREE_CLASS);
    }
    for (int i = 0; i < n_maybe_tree; i++, j++)
    {
        images_paths[j] = pair<string, int>(maybe_tree_images_paths[i], MAYBE_TREE_CLASS);
    }
    for (int i = 0; i < n_non_tree; i++, j++)
    {
        images_paths[j] = pair<string, int>(non_tree_images_paths[i], NON_TREE_CLASS);
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
    int rows = 3;
    int cols = 3;
    /*** VOCABULARY PARAMETER ***/
    int num_bins = vocabulary.rows;

    vector<Rect> windows;
    vector<Mat> histogram(rows*cols + 1);

    for (int i = 0; i < n_images; i++)
    {
        Mat img = imread(images_paths[i].first);
        int img_class = images_paths[i].second;

        cvtColor(img, img, COLOR_BGR2GRAY);

        int min_size = min(img.rows, img.cols);
        int step = min_size/3;

        if (img_class == NON_TREE_CLASS) {
            int w_size = min_size;

            for (int x = 0; x <= img.cols - w_size; x += step) {
                for (int y = 0; y <= img.rows - w_size; y += step) {

                    windows = getFrames(rows, cols, x, y, w_size);

                    for (int w=0; w<windows.size(); w++) {
                        //Detect SIFT keypoints (or feature points)
                        detector->detect(img(windows[w]), keypoints);

                        //extract BoW (or BoF) descriptor from given image
                        bowDE.compute(img(windows[w]), keypoints, histogram[w]);
                        if (histogram[w].empty())
                            histogram[w] = Mat::zeros(1, num_bins, CV_32F);
                    }

                    Mat tot_desc(1, num_bins*windows.size(), CV_32F);

                    for (int w=0; w<windows.size(); w++)
                        histogram[w].copyTo(tot_desc(Rect(w*num_bins, 0, num_bins, 1)));

                    // insert the first n_images * proportion images into the train set, the rest into the test set
                    if (i < n_images * proportion) {
                        addRowCSV(train_set_f, tot_desc, img_class);
                    } else {
                        addRowCSV(test_set_f, tot_desc, img_class);
                    }
                }

            }
        }
        else {

            windows = getWindows(img, rows, cols);

            for (int w = 0; w < windows.size(); w++) {
                //Detect SIFT keypoints (or feature points)
                detector->detect(img(windows[w]), keypoints);

                //extract BoW (or BoF) descriptor from given image
                bowDE.compute(img(windows[w]), keypoints, histogram[w]);
                if (histogram[w].empty())
                    histogram[w] = Mat::zeros(1, num_bins, CV_32F);

                Mat tot_desc(1, num_bins * windows.size(), CV_32F);

                for (int w = 0; w < windows.size(); w++)
                    histogram[w].copyTo(tot_desc(Rect(w * num_bins, 0, num_bins, 1)));

                // insert the first n_images * proportion images into the train set, the rest into the test set
                if (i < n_images * proportion) {
                    addRowCSV(train_set_f, tot_desc, img_class);
                } else {
                    addRowCSV(test_set_f, tot_desc, img_class);
                }
            }
        }

        cout << to_string((int)((float)(i + 1) / n_images * 100)) << "% completed" << endl;
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