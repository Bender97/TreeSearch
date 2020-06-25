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

    for (int scale = 1; scale < 5; scale++) {
        int w_size = min_size / scale;

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

                Mat tot_desc(1, num_bins * windows.size(), CV_32F);

                for (int w = 0; w < windows.size(); w++) {
                    histogram[w].copyTo(tot_desc(Rect(w * num_bins, 0, num_bins, 1)));
                }
                int response = (int) this->classifier->predict(tot_desc);


                if (response == TREE_CLASS) {
                    rectangle(canv_result, windows[0], {255, 0, 255}, scale);

                    regions.push_back(windows[0]);
                    classes.push_back(TREE_CLASS);
                    scales.push_back(scale);
                } else if (response == MAYBE_TREE_CLASS) {
                    rectangle(canv_result, windows[0], {0, 255, 0}, scale);

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

    //vector<Rect> unified_regs = unifyRegions(regions, classes, 1.8, 0.6);
    vector<Rect> unified_regs = unifyRegionsClustering(regions, classes, 0.01, 100, 2, 0.02);

    for (auto region : unified_regs) {

        rectangle(result, region, { 0, 0, 255 }, 2);

    }

    return result;
}

vector<Rect> Detector::unifyRegions(vector<Rect> regions, vector<int> classes, float max_span, float score_threshold)
{

    vector<Point2i> sums_tl_regions;
    vector<Point2i> sums_br_regions;
    vector<float> n_regs_unified;

    // for each region
    for (int i = 0; i < regions.size(); i++)
    {

        bool found = false;
        Point2i reg_tl = regions[i].tl();
        Point2i reg_br = regions[i].br();

        // count trees 4 times and maybe_trees 1 time
        float mult = classes[i] == TREE_CLASS ? 100 : 1;

        // for each output region until found (if found!)
        for (int j = 0; j < sums_tl_regions.size() && !found; j++)
        {

            // get the tl point of the out region and its max_span rectangle
            Point2i out_reg_tl = sums_tl_regions[j] / n_regs_unified[j];
            Point2i out_reg_br = sums_br_regions[j] / n_regs_unified[j];

            // get the out region size scaled by max_span
            int span_size = (out_reg_br.x - out_reg_tl.x) * max_span;

            Rect out_span(out_reg_tl - Point2i(span_size / 2, span_size / 2), Size(span_size, span_size));
            Rect out_reg(out_reg_tl, out_reg_br);

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
        if (!found)
        {

            cout << "new region!" << endl;

            // create a new output region
            sums_tl_regions.push_back(reg_tl * mult);
            sums_br_regions.push_back(reg_br * mult);
            n_regs_unified.push_back(mult);

        }
    }

    float max_mult = 0;
    // max score computation (the region with max multiplicity will have score 1)
    for (int i = 0; i < n_regs_unified.size(); i++)
    {

        if (n_regs_unified[i] > max_mult)
        {
            max_mult = n_regs_unified[i];
        }
    }

    vector<Rect> out_regions(sums_tl_regions.size());

    for (int i = 0; i < sums_tl_regions.size(); i++)
    {

        float score = n_regs_unified[i] / max_mult;

        cout << "score: " << score << endl;

        // if the region passes the non maxima suppression add it to the final regions
        if (score >= score_threshold)
            out_regions[i] = Rect(sums_tl_regions[i] / n_regs_unified[i], sums_br_regions[i] / n_regs_unified[i]);

    }

    return out_regions;

}

vector<Rect> Detector::unifyRegionsClustering(vector<Rect> regions, vector<int> classes, float c, float win_size, float shift_threshold, float score_threshold) {

    int reg_num = regions.size();

    // step 1: compute the centers of the regions together with their scale (size of the region)
    // and the multiplicity (if they are maybe_tree they are evaluated as 1 point, trees are
    // evaluated as 10 points)
    vector<Point2i> reg_pts(reg_num);
    vector<int> scales(reg_num);
    vector<float> multiplicity(reg_num);

    int maybe_tree_weight = 1;
    int tree_weight = 10;

    // compute regions centers
    for (int i = 0; i < reg_num; i++)
    {

        Size reg_size = regions[i].size();
        Point2i center = regions[i].tl() + Point2i(reg_size.width / 2, reg_size.height / 2);

        reg_pts[i] = center;
        scales[i] = regions[i].height;
        multiplicity[i] = classes[i] == TREE_CLASS ? tree_weight : maybe_tree_weight;

    }

    // step 2: run the mean shift algorithm to find the output regions (basins of attraction)
    // together with their score (number of regions unified),
    // radius of region by x and y, height and width of the region
    vector<Point2i> out_centers;
    vector<float> n_regs_unified;
    vector<float> radius_x;
    vector<float> radius_y;
    vector<int> heights;
    vector<int> widths;

    // for each region (each point int the set)
    for (int i = 0; i < reg_num; i++)
    {

        float p_x = reg_pts[i].x;
        float p_y = reg_pts[i].y;

        // shift
        float shift_x;
        float shift_y;
        // differencies of the original point and shift (movement)
        float diff_x;
        float diff_y;
        // window size to consider
        float win_size = scales[i] / 2;

        do
        {
            float scale_factor = 0;
            shift_x = 0;
            shift_y = 0;

            for (int j = 0; j < reg_num; j++)
            {

                // numerator
                float dx = p_x - reg_pts[j].x;
                float dy = p_y - reg_pts[j].y;

                // apply normal kernel on distance (depending on window size)
                float dist = dx * dx + dy * dy;
                float weight = c * exp(-1.0 / 2.0 * dist / (win_size * win_size));

                shift_x += reg_pts[j].x * weight * multiplicity[j];
                shift_y += reg_pts[j].y * weight * multiplicity[j];

                // denominator
                scale_factor += weight * multiplicity[j];

            }

            shift_x = shift_x / scale_factor;
            shift_y = shift_y / scale_factor;

            diff_x = p_x - shift_x;
            diff_y = p_y - shift_y;

            p_x = shift_x;
            p_y = shift_y;



        } while (abs(diff_x) >= shift_threshold || abs(diff_y) >= shift_threshold);


        // now find the peak where the point stopped
        bool found_peak = false;
        for (int j = 0; j < out_centers.size() && !found_peak; j++)
        {

            float dx = p_x - out_centers[j].x;
            float dy = p_y - out_centers[j].y;
            float dist_peak = sqrt(dx * dx + dy * dy);

            // if the distance from the peak is small, then the point is in the peak j
            if (dist_peak <= win_size / 2)
            {
                found_peak = true;
                n_regs_unified[j] += multiplicity[i];

                float dx_radius = abs(reg_pts[i].x - out_centers[j].x);
                float dy_radius = abs(reg_pts[i].y - out_centers[j].y);

                if (dx_radius > radius_x[j]) {
                    radius_x[j] = dx_radius;
                    widths[j] = dx_radius + scales[i] / 2;
                }
                if (dy_radius > radius_y[j]) {
                    radius_y[j] = dy_radius;
                    heights[j] = dy_radius + scales[i] / 2;
                }

            }

        }
        // if a new peak was found, add it
        if (!found_peak)
        {
            out_centers.push_back(Point2i(p_x, p_y));
            n_regs_unified.push_back(multiplicity[i]);
            heights.push_back(scales[i]);
            widths.push_back(scales[i]);
            radius_x.push_back(0);
            radius_y.push_back(0);
        }

    }

    float max_mult = 0;
    // max score computation (the region with max multiplicity will have score 1)
    for (int i = 0; i < n_regs_unified.size(); i++)
    {

        if (n_regs_unified[i] > max_mult)
        {
            max_mult = n_regs_unified[i];
        }
    }

    // compute the final regions
    vector<Rect> out_regions(out_centers.size());

    for (int i = 0; i < out_centers.size(); i++)
    {

        float score = n_regs_unified[i] / max_mult;

        cout << out_centers[i] <<  " score: " << score << endl;

        // if the region passes the non maxima suppression add it to the final regions
        if (score >= score_threshold)
        {
            Point2i tl(out_centers[i].x - widths[i] / 2, out_centers[i].y - heights[i] / 2);
            Size size(widths[i], heights[i]);

            out_regions[i] = Rect(tl, size);
        }

    }

    return out_regions;

}


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