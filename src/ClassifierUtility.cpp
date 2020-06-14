//
// Created by cogny on 14/06/20.
//

#include "../include/ClassifierUtility.h"

double trainModel(string train_dataset_path, Ptr<ml::SVM> svm)
{
    pair<Mat, Mat> dataset = loadDataset(train_dataset_path);

    Mat trainingDataMat(dataset.first);
    Mat labelsMat(dataset.second);

    svm = ml::SVM::create();
    svm->setType(ml::SVM::C_SVC);
    svm->setKernel(ml::SVM::LINEAR);
    svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 100, 1e-6));
    svm->train(trainingDataMat, ml::ROW_SAMPLE, labelsMat);

    int error = 0;

    for (int r=0; r<trainingDataMat.rows; r++) {
        Mat img = trainingDataMat.row(r);
        int result = round(svm->predict(img));
        if (result!=labelsMat.at<uchar>(r))
            error++;
    }

	return (double) error/labelsMat.rows;
}

double testModel(string test_dataset_path, Ptr<ml::SVM> svm)
{
    pair<Mat, Mat> dataset = loadDataset(test_dataset_path);

    Mat testDataMat(dataset.first);
    Mat labelsMat(dataset.second);
    int error = 0;

    for (int r=0; r<testDataMat.rows; r++) {
        Mat img = testDataMat.row(r);
        int result = round(svm->predict(img));
        if (result!=labelsMat.at<uchar>(r))
            error++;
    }

    return (double) error/labelsMat.rows;

}