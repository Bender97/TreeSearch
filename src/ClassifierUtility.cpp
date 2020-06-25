
//
// Created by cogny on 14/06/20.
//

#include "../include/ClassifierUtility.h"

double trainModel(const string &train_dataset_path, Ptr<ml::SVM> &svm)
{
    pair<Mat, Mat> dataset = loadDataset(train_dataset_path);

    Ptr<ml::TrainData> td = ml::TrainData::create(dataset.first, ml::ROW_SAMPLE, dataset.second);

    svm = ml::SVM::create();
    svm->setType(ml::SVM::NU_SVC);
    svm->setNu(0.05);
    svm->setKernel(ml::SVM::CHI2);
    svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 100000, 1e-6));
    svm->trainAuto(td);

    /** COMPUTE TRAIN ERROR **/
    int error = 0;

    for (int r=0; r<dataset.first.rows; r++) {
        Mat hist = dataset.first.row(r);

        int response = (int) svm->predict(hist);
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