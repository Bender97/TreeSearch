//
// Created by fusy on 13/06/20.
//

#include "../include/ImagesUtility.h"

vector<string> loadImagesPaths(string path) {

    vector<string> filenames;   // vector of all the filenames found in the specified folder
    vector<string> allowedExtensions = { ".jpg", ".jpeg", ".JPG", ".png", ".bmp" };    // allowed extensions for images

    /// for each extension, search for images in the specified folder
    for (int i = 0; i < allowedExtensions.size(); i++) {
        vector<string> currentFilenames;    // filenames found with the i-th extension

        /// look for matches with the pattern
        cv::glob(path + "*" + allowedExtensions[i],
                 currentFilenames, true);

        /// insert found filenames in the general vector
        filenames.insert(
                filenames.end(),
                currentFilenames.begin(),
                currentFilenames.end());
    }

    if (filenames.empty()) cout << "No images found. Exiting." << endl;

    return filenames;
}

Mat loadImage(string path) {
    return imread(path);
}

bool storeImage(string path, Mat img) {
    return imwrite(path, img);
}

vector<Rect> getWindows(Mat img, int rows, int cols){

    vector<Rect> windows = vector<Rect>(rows*cols+1);
    windows[0] = Rect(0, 0, img.cols, img.rows);

    int width = img.cols/cols;
    int height = img.rows/rows;

    for (int r=0; r<rows; r++)
        for (int c=0; c<cols; c++)
            windows[r*cols + c +1] = Rect(c*width, r*height, width, height);

    return windows;
}

vector<Rect> getFrames(Mat img, int rows, int cols, int x, int y, int w_size){

    vector<Rect> windows = vector<Rect>(rows*cols+1);
    windows[0] = Rect(x, y, w_size, w_size);

    int width = w_size/cols;
    int height = w_size/rows;

    for (int r=0; r<rows; r++)
        for (int c=0; c<cols; c++)
            windows[r*cols + c +1] = Rect(x + c*width, y + r*height, width, height);

    return windows;
}