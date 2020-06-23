//
// Created by fusy on 14/06/20.
//

#include "../include/Detector.h"

void Detector::setVocabulary(Mat vocabulary) {
    this->vocabulary = vocabulary;
}

void Detector::setClassifier(Ptr<ml::SVM> &classifier) {
    this->classifier = classifier;
}

Mat Detector::detectTrees(Mat img, bool verbose) {
    //create a nearest neighbor matcher
    //Ptr<DescriptorMatcher> matcher(new FlannBasedMatcher);
    Ptr<DescriptorMatcher> matcher = BFMatcher::create(NORM_L2);
    //create Sift feature point detector
    Ptr<xfeatures2d::SIFT> detector = xfeatures2d::SIFT::create();
    //create BoF (or BoW) descriptor extractor
    BOWImgDescriptorExtractor bowDE(detector, matcher);
    //Set the dictionary with the vocabulary we created in the first step
    bowDE.setVocabulary(this->vocabulary);

    Mat canv_result = img.clone();
    Mat result = img.clone();


    cvtColor(img, img, COLOR_BGR2GRAY);

    //To store the keypoints that will be extracted by SIFT
    vector<KeyPoint> keypoints;

    int min_size = min(img.rows, img.cols);
    int step = min_size / 12;

    /*** WINDOWING PARAMETERS ***/
    int rows = 3;
    int cols = 3;
    /*** VOCABULARY PARAMETER ***/
    int num_bins = vocabulary.rows;

    vector<Rect> windows;
    vector<Mat> histogram(rows*cols + 1);

    vector<Rect> regions;
    vector<int> classes;
    vector<float> scales;

    for (int scale = 1; scale < 4; scale++) {
        int w_size = min_size / scale;

        for (int x = 0; x <= img.cols - w_size; x += step) {
            for (int y = 0; y <= img.rows - w_size; y += step) {

                windows = getFrames(rows, cols, x, y, w_size);

                bool flag = false;

                for (int w=0; w<windows.size(); w++) {
                    //Detect SIFT keypoints (or feature points)
                    detector->detect(img(windows[w]), keypoints);

                    //extract BoW (or BoF) descriptor from given image
                    bowDE.compute(img(windows[w]), keypoints, histogram[w]);
                    if (histogram[w].empty())
                        histogram[w] = Mat::zeros(1, num_bins, CV_32F);

                    Mat tot_desc(1, num_bins*windows.size(), CV_32F);

                    for (int w=0; w<windows.size(); w++) {
                        histogram[w].copyTo(tot_desc(Rect(w*num_bins, 0, num_bins, 1)));
                    }
                    int response = (int) this->classifier->predict(tot_desc);


                    if (response == TREE_CLASS) {
                        rectangle(canv_result, windows[0], {255, 0, 255}, scale);

                        regions.push_back(windows[0]);
                        classes.push_back(TREE_CLASS);
                        scales.push_back(scale);
                    }
                    else if (response == MAYBE_TREE_CLASS) {
                        rectangle(canv_result, windows[0], { 0, 255, 0 }, scale);

                        regions.push_back(windows[0]);
                        classes.push_back(MAYBE_TREE_CLASS);
                        scales.push_back(scale);
                    }

                if(verbose) {
                    cout << "analyzing: " << x << "." << y << endl;

                    Mat canvas = canv_result.clone();

                    rectangle(canvas, windows[0], {0, 0, 255});

                    while (canvas.cols > 1500 || canvas.rows > 1000)
                        resize(canvas, canvas, Size(canvas.cols / 2, canvas.rows / 2));

                    imshow("Window", canvas);

                    waitKey(1);
                }
            }
        }
    }

    vector<Rect> unified_regs = unifyRegions(regions, classes, 1.8, 0.6);

    for (auto region : unified_regs) {

        rectangle(result, region, { 0, 0, 255 });

    }

    return result;
}

vector<Rect> Detector::unifyRegions(vector<Rect> regions, vector<int> classes, float max_span, float score_threshold) {

    vector<Point2i> sums_tl_regions;
    vector<Point2i> sums_br_regions;
    vector<float> n_regs_unified;

    // for each region
    for (int i = 0; i < regions.size(); i++) {

        bool found = false;
        Point2i reg_tl = regions[i].tl();
        Point2i reg_br = regions[i].br();

        // count trees 4 times and maybe_trees 1 time
        float mult = classes[i] == TREE_CLASS ? 100 : 1;

        // for each output region until found (if found!)
        for (int j = 0; j < sums_tl_regions.size() && !found; j++) {

            // get the tl point of the out region and its max_span rectangle
            Point2i out_reg_tl = sums_tl_regions[j] / n_regs_unified[j];
            Point2i out_reg_br = sums_br_regions[j] / n_regs_unified[j];

            // get the out region size scaled by max_span
            int span_size = (out_reg_br.x - out_reg_tl.x) * max_span;

            Rect out_span (out_reg_tl - Point2i(span_size / 2, span_size / 2), Size(span_size, span_size));
            Rect out_reg (out_reg_tl, out_reg_br);

            // if the current region tl is inside the max span of the out region
            // or the current region is inside the out region
            // then count it in
            if (isInsideRect(reg_tl, out_span) || isInsideRect(regions[i], out_reg)) {

                if (isInsideRect(reg_tl, out_span)) cout << "inside span" << endl;
                else cout << "inside region" << endl;

                found = true;
                // weight also by scale wrt the output region
                cout << "region is of size = " << regions[i].size() << endl;
                cout << "out region is of size = " << out_reg.size() << endl;
                float scale = ((float)regions[i].size().height / out_reg.size().height);
                mult *= pow(scale, 5);

                cout << "region number: " << j << endl;

                cout << "scale is = " << pow(scale, 5) << endl;
                cout << "mult is = " << mult << endl;

                // update the output region
                sums_tl_regions[j] += reg_tl * mult;
                sums_br_regions[j] += reg_br * mult;
                n_regs_unified[j] += mult;


            }

        }

        // if no output region was found to unify
        if (!found) {

            cout << "new region!" << endl;

            // create a new output region
            sums_tl_regions.push_back(reg_tl * mult);
            sums_br_regions.push_back(reg_br * mult);
            n_regs_unified.push_back(mult);

        }
    }

    float max_mult = 0;
    // max score computation (the region with max multiplicity will have score 1)
    for (int i = 0; i < n_regs_unified.size(); i++) {

        if (n_regs_unified[i] > max_mult) {
            max_mult = n_regs_unified[i];
        }
    }

    vector<Rect> out_regions(sums_tl_regions.size());

    for (int i = 0; i < sums_tl_regions.size(); i++) {

        float score = n_regs_unified[i] / max_mult;

        cout << "score: " << score << endl;

        // if the region passes the non maxima suppression add it to the final regions
        if (score >= score_threshold)
            out_regions[i] = Rect(sums_tl_regions[i] / n_regs_unified[i], sums_br_regions[i] / n_regs_unified[i]);

    }

    return out_regions;

}
/*
vector<Rect> Detector::unifyRegionsClustering(vector<Rect> regions, vector<int> classes, float dist_threshold, float score_threshold) {

    int reg_num = regions.size();

    vector<Point2i> reg_pts(reg_num);
    vector<Point2i> multiplicity(reg_num);

    int maybe_tree_w = 1;
    int tree_w = 10;

    // compute regions centers
    for (int i = 0; i < reg_num; i++) {

        Size reg_size = regions[i].size();
        Point2i center = regions[i].tl() + Point2i(reg_size.width / 2, reg_size.height / 2);

        reg_pts.push_back(center);

    }

    bool term_cond = false;
    vector<Point2i> out_centers;
    vector<float> n_regs_unified;

    // for each region
    for (int i = 0; i < regions.size(); i++) {

        bool found = false;
        Point2i reg_center = reg_pts[i];

        // count trees 4 times and maybe_trees 1 time
        float mult = classes[i] == TREE_CLASS ? tree_w : maybe_tree_w;

        // nearest center
        float min_dist = 0;
        int min_idx = -1;

        // for each output region until found (if found!)
        for (int j = 0; j < out_centers.size(); j++) {

            // get the out region size scaled by max_span
            Point2i out_center = out_centers[j];

            float delta_x = reg_center.x - out_center.x;
            float delta_y = reg_center.y - out_center.y;

            float dist = delta_x*delta_x + delta_y*delta_y;

            if (dist < 5) {



            }

        }

        // if no output region was found to unify
        if (!found) {

            cout << "new region!" << endl;

            out_centers.push_back(reg_center);

        }
    }


    float max_mult = 0;
    // max score computation (the region with max multiplicity will have score 1)
    for (int i = 0; i < n_regs_unified.size(); i++) {

        if (n_regs_unified[i] > max_mult) {
            max_mult = n_regs_unified[i];
        }
    }

    vector<Rect> out_regions(out_centers.size());

    for (int i = 0; i < out_centers.size(); i++) {

        float score = n_regs_unified[i] / max_mult;

        cout << "score: " << score << endl;

        // if the region passes the non maxima suppression add it to the final regions
        if (score >= score_threshold)
            out_regions[i] = Rect();

    }

    return out_regions;

}
*/
bool Detector::isInsideRect(Point2i pt, Rect rect) {

    Point2i tl = rect.tl();
    Point2i br = rect.br();
    
    return (tl.x <= pt.x && pt.x <= br.x) && (tl.y <= pt.y && pt.y <= br.y);

}

bool Detector::isInsideRect(Rect inside, Rect rect) {

    Point2i tl_inside = inside.tl();
    Point2i br_inside = inside.br();

    return isInsideRect(tl_inside, rect) && isInsideRect(br_inside, rect);

}