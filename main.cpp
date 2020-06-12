#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>


using namespace std;
using namespace cv;

int H, S, V, Hrange, Srange, Vrange;

int canny_threshold;

int morph_size;
int blur_size, sigma_range, sigma_space;

int canny_low, canny_high;

void onChange(int, void * info) {
    Mat img = (*(Mat *) info).clone();

    for (int r=0; r<img.rows; r++) {
        for(int c=0; c<img.cols; c++) {
            Vec3b & color = img.at<Vec3b>(r, c);
            if (color[0]<H-Hrange || color[0]>H+Hrange) color[0]=0;
            if (color[1]<S-Srange || color[1]>S+Srange) color[1]=0;
            if (color[2]<V-Vrange || color[2]>V+Vrange) color[2]=0;
        }
    }
    while(img.cols>800 || img.rows>600)
        resize(img, img, Size(img.cols/2, img.rows/2));

    Mat hsv_channels[3];
    cv::split( img, hsv_channels );
    img = hsv_channels[2];

    Mat temp = img.clone();

    bilateralFilter( temp, img, blur_size*2+1, sigma_range, sigma_space);
    //if (canny_threshold<1) canny_threshold = 1;
    if (canny_low<1) canny_low = 1;
    if (canny_high<1) canny_high = 1;
    /// Canny detector
    Canny(img, img, canny_low,
          canny_high, 3 );
    /*Mat kernel1 = Mat::ones(4, 4, CV_8U);
    dilate(img, img, kernel1);


    //Mat element = getStructuringElement( 2, Size( 2*morph_size + 1, 2*morph_size+1 ), Point( morph_size, morph_size ) );
    Mat element = getStructuringElement( 0, Size( 2*morph_size + 1, 2*morph_size+1 ), Point( 1, 1 ) );
    morphologyEx( img, img, 2, element );*/

    /*** develop mask ***/
    int temp_H = 13;
    int temp_Hrange = 35;
    int temp_S = 24;
    int temp_Srange = 256;
    int temp_V = 15;
    int temp_Vrange = 256;
    int temp_canny_low = 173;
    int temp_canny_high = 943;
    int temp_blur_size = 15;
    int temp_sigma_range = 24;
    int temp_sigma_space = 25;

    Mat mask = (*(Mat *) info).clone();
    for (int r=0; r<mask.rows; r++) {
        for(int c=0; c<mask.cols; c++) {
            Vec3b & color = mask.at<Vec3b>(r, c);
            if (color[0]<temp_H-temp_Hrange || color[0]>temp_H+temp_Hrange) color[0]=0;
            if (color[1]<temp_S-temp_Srange || color[1]>temp_S+temp_Srange) color[1]=0;
            if (color[2]<temp_V-temp_Vrange || color[2]>temp_V+temp_Vrange) color[2]=0;
        }
    }
    while(mask.cols>800 || mask.rows>600)
        resize(mask, mask, Size(mask.cols/2, mask.rows/2));

    cv::split( mask, hsv_channels );
    mask = hsv_channels[2];

    temp = mask.clone();

    bilateralFilter( temp, mask, temp_blur_size*2+1, temp_sigma_range, temp_sigma_space);

    /// Canny detector
    Canny(mask, mask, temp_canny_low,
          temp_canny_high, 3 );

    

    imshow("temp", img);

}

int main(int args, char ** argv) {

    H = 13;
    Hrange = 35;
    S = 24;
    Srange = 256;
    V = 15;
    Vrange = 256;
    canny_threshold = 98;

    morph_size = 3;
    blur_size  =1;

    //Mat img = imread("../train/17.jpg");
    for (int i=1; i<10; i++) {
        Mat img = imread("../Benchmark_step_1/Figure " + to_string(i) + ".jpg");
        //Mat img = imread("../dataset/01.jpg");
        //Mat img = imread(argv[1]);

        //cvtColor(img, img, COLOR_BGR2YCrCb);
        cvtColor(img, img, COLOR_BGR2HSV);

        namedWindow("temp", WINDOW_AUTOSIZE);

        /*createTrackbar("H threshold", "temp", &H, 256, onChange, &img);
        createTrackbar("Hrange", "temp", &Hrange, 256, onChange, &img);*/

        createTrackbar("S threshold", "temp", &S, 256, onChange, &img);
        createTrackbar("Srange", "temp", &Vrange, 256, onChange, &img);

        /*createTrackbar("V threshold", "temp", &V, 256, onChange, &img);
        createTrackbar("Vrange", "temp", &Srange, 256, onChange, &img);*/

        //createTrackbar("Canny_thr", "temp", &canny_threshold, 256, onChange, &img);
        createTrackbar("Canny_low", "temp", &canny_low, 2000, onChange, &img);
        createTrackbar("Canny_high", "temp", &canny_high, 2000, onChange, &img);

        //createTrackbar("Morph size", "temp", &morph_size, 6, onChange, &img);
        createTrackbar("blur size", "temp", &blur_size, 50, onChange, &img);
        createTrackbar("sigma range", "temp", &sigma_range, 100, onChange, &img);
        createTrackbar("sigma space", "temp", &sigma_space, 100, onChange, &img);

        onChange(0, &img);

        //imshow("temp", img);
        waitKey(0);
        destroyAllWindows();
    }






    return 0;
}
