//
// Created by cogny on 14/06/20.
//

#include "../include/ClassifierUtility.h"

double trainModel(string train_dataset_path, Ptr<ml::SVM> &svm)
{
    pair<Mat, Mat> dataset = loadDataset(train_dataset_path);

    Ptr<ml::TrainData> td = ml::TrainData::create(dataset.first, ml::ROW_SAMPLE, dataset.second);

    svm = ml::SVM::create();
    svm->setType(ml::SVM::NU_SVC);
    svm->setNu(0.95);
    svm->setKernel(ml::SVM::RBF);

    svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 100000, 1e-6));
//    svm->train(dataset.first, ml::ROW_SAMPLE, dataset.second);
    svm->trainAuto(td);
    int error = 0;

    for (int r=0; r<dataset.first.rows; r++) {
        Mat hist = dataset.first.row(r);

        int response = svm->predict(hist);
        int result = round(response);
//        cout << "predicted: " << response << " actual: " << (int)dataset.second.at<int>(r) << endl;
        if (result!=dataset.second.at<int>(r))
            error++;
    }
    Mat resp;
    cout << "calcError: " << svm->calcError(td, 0, resp) << endl;
    cout << "error: " << error << endl;
	return (double) error/dataset.second.rows;
}

double testModel(string test_dataset_path, Ptr<ml::SVM> svm)
{
    pair<Mat, Mat> dataset = loadDataset(test_dataset_path);

    int error = 0;

    for (int r=0; r<dataset.first.rows; r++) {
        Mat hist = dataset.first.row(r);

        float response = svm->predict(hist);
        int result = round(response);
//        cout << "predicted: " << response << " actual: " << (int)dataset.second.at<int>(r) << endl;
        if (result!=dataset.second.at<int>(r))
            error++;
    }

    return (double) error/dataset.second.rows;

}