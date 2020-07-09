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

    // post processing of the classified regions
    vector<Rect> unified_regs = unifyRegionsClustering(regions, classes, 0.01, 2, 0.02);
    // draw the output regions on the resulting image
    for (auto region : unified_regs) {

        rectangle(result, region, { 0, 0, 255 }, 2);

    }

    return result;
}


vector<Rect> Detector::unifyRegionsClustering(vector<Rect> regions, vector<int> classes, float c, float shift_threshold, float score_threshold) {

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
    vector<float> tree_rank;
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
        float win_size = scales[i] / 4;

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

                if (tree_rank[j] <= multiplicity[i])
                {

                    float dx_radius = abs(reg_pts[i].x - out_centers[j].x);
                    float dy_radius = abs(reg_pts[i].y - out_centers[j].y);

                    if (tree_rank[j] < multiplicity[i])
                    {
                        // if the tree rank is upgraded, force the update
                        tree_rank[j] = multiplicity[i];
                        radius_x[j] = dx_radius;
                        widths[j] = (dx_radius + scales[i] / 2) * 2;
                        radius_y[j] = dy_radius;
                        heights[j] = (dy_radius + scales[i] / 2) * 2;
                    }
                    else
                    {
                        if (dx_radius > radius_x[j]) {
                            radius_x[j] = dx_radius;
                            widths[j] = (dx_radius + scales[i] / 2) * 2;
                        }
                        if (dy_radius > radius_y[j]) {
                            radius_y[j] = dy_radius;
                            heights[j] = (dy_radius + scales[i] / 2) * 2;
                        }
                    }
                }
            }
        }
        // if a new peak was found, add it
        if (!found_peak)
        {
            out_centers.push_back(Point2i(p_x, p_y));
            n_regs_unified.push_back(multiplicity[i]);
            tree_rank.push_back(multiplicity[i]);
            heights.push_back(scales[i]);
            widths.push_back(scales[i]);
            radius_x.push_back(0);
            radius_y.push_back(0);
        }

    }
    
    // step 3: merge output regions that belongs to the same tree but have different centers
    // move the number of regions unified from an output region to the other when merged, and
    // never consider it again
    for (int i = 0; i < n_regs_unified.size(); i++)
    {
        // don't consider merged regions
        if (n_regs_unified[i] != 0)
        {
            // compute the range from the size of the output region
            int min_size_i = min(heights[i], widths[i]);
            int range = min_size_i / 4;

            for (int j = 0; j < n_regs_unified.size(); j++) {

                if (i != j && n_regs_unified[j] != 0)
                {

                    int dx_radius = abs(out_centers[i].x - out_centers[j].x);
                    int dy_radius = abs(out_centers[i].y - out_centers[j].y);

                    if (dx_radius <= range && dy_radius <= range)
                    {
                        // take the weighted average of the sizes
                        widths[i] = (n_regs_unified[i] * widths[i] + n_regs_unified[j] * widths[j]) / (n_regs_unified[i] + n_regs_unified[j]);
                        heights[i] = (n_regs_unified[i] * heights[i] + n_regs_unified[j] * heights[j]) / (n_regs_unified[i] + n_regs_unified[j]);
                        // move the weight to the receiver region
                        n_regs_unified[i] += n_regs_unified[j];
                        n_regs_unified[j] = 0;
                    }
                }
            }
        }
    }

    // step 4: merge output regions that are contained in one another
    for (int i = 0; i < n_regs_unified.size(); i++)
    {
        if (n_regs_unified[i] != 0)
        {

            int min_size_i = min(heights[i], widths[i]);
            int range = min_size_i / 4;

            for (int j = 0; j < n_regs_unified.size(); j++) {

                if (i != j && n_regs_unified[j] != 0)
                {
                    // compute boundaries
                    int left_x_i = out_centers[i].x - widths[i] / 2;
                    int right_x_i = out_centers[i].x + widths[i] / 2;
                    int upper_y_i = out_centers[i].y - heights[i] / 2;
                    int lower_y_i = out_centers[i].y + heights[i] / 2;

                    int left_x_j = out_centers[j].x - widths[j] / 2;
                    int right_x_j = out_centers[j].x + widths[j] / 2;
                    int upper_y_j = out_centers[j].y - heights[j] / 2;
                    int lower_y_j = out_centers[j].y + heights[j] / 2;

                    bool is_inside =    (left_x_i <= left_x_j && right_x_j <= right_x_i &&
                                        upper_y_i <= upper_y_j && lower_y_j <= lower_y_i) ||
                                        (left_x_j <= left_x_i && right_x_i <= right_x_j &&
                                        upper_y_j <= upper_y_i && lower_y_i <= lower_y_j);
                    
                    if (is_inside)
                    {
                        // take the max size because the bigger region will prevail on the smaller
                        widths[i] = max(widths[i], widths[j]);
                        heights[i] = max(heights[i], heights[j]);
                        // move the weight to the receiver region
                        n_regs_unified[i] += n_regs_unified[j];
                        n_regs_unified[j] = 0;

                    }
                }
            }
        }
    }

    // step 6: take the maximum weight and compute the score of each output region
    // for non-maxima suppression

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