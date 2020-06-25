//
// Created by cogny on 14/06/20.
//

#include "../include/ClassifierUtility.h"

double trainModel(const string &train_dataset_path, Ptr<ml::ANN_MLP> &ann)
{
    pair<Mat, Mat> dataset = loadDataset(train_dataset_path);

    int nfeatures = dataset.first.cols;
    ann = ml::ANN_MLP::create();
    Mat_<int> layers(4,1);
    layers(0) = nfeatures;     // input
    layers(1) = 3 * 512;  // hidden
    layers(2) = 3 * 128;  // hidden
    layers(3) = 3;      // output, 1 pin per class.
    ann->setLayerSizes(layers);
    ann->setActivationFunction(ml::ANN_MLP::SIGMOID_SYM,0,0);
    ann->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER+TermCriteria::EPS, 300, 0.0001));
    ann->setTrainMethod(ml::ANN_MLP::BACKPROP, 0.0001);

    Mat train_classes = Mat::zeros(dataset.first.rows, 3, CV_32FC1);
    for(int i=0; i<train_classes.rows; i++)
    {
        train_classes.at<float>(i, dataset.second.at<int>(i)) = 1.f;
    }
    cerr << dataset.first.size() << " " << train_classes.size() << endl;

    ann->train(dataset.first, ml::ROW_SAMPLE, train_classes);

    /** COMPUTE TRAIN ERROR **/
    int error = 0;

    for (int r=0; r<dataset.first.rows; r++) {
        Mat hist = dataset.first.row(r);

        int response = (int) ann->predict(hist);
        int result = round(response);
        if (result!=dataset.second.at<int>(r))
            error++;
    }

	return (double) error/dataset.second.rows;
}

double testModel(const string &test_dataset_path, Ptr<ml::SVM> svm)
{
    cout << "Loading Test dataset: " << test_dataset_path << endl;
    pair<Mat, Mat> dataset = loadDataset(test_dataset_path);
    cout << "Test dataset correctly loaded" << endl << endl;

    /** COMPUTE TEST ERROR **/
    int error = 0;

    for (int r=0; r<dataset.first.rows; r++) {
        Mat hist = dataset.first.row(r);

        float response = svm->predict(hist);
        int result = (int) round(response);
        if (result!=dataset.second.at<int>(r))
            error++;
    }

    return (double) error/dataset.second.rows;

}