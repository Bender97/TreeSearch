string dir = "../dataset/";
    string filepath;

    DIR *dp;
    dp = opendir(dir.c_str());

    struct dirent *dirp;
    struct stat filestat;

    //detecting keypoints
    Ptr<DescriptorExtractor> extractor = xfeatures2d::SIFT::create();
    vector<KeyPoint> keypoints;
    Mat descriptors;
    Mat training_descriptors(1, extractor->descriptorSize(), extractor->descriptorType());
    Mat img;

    cout << "------- build vocabulary ---------" << endl;
    cout << "extract descriptors.." << endl;

    int count = 0;

    while(count++ < 15 && (dirp=readdir(dp))) {
        filepath = dir + dirp->d_name;
        if (stat( filepath.c_str(), &filestat )) continue;
        if (S_ISDIR( filestat.st_mode )) continue;

        img = imread(filepath);
        extractor->detectAndCompute(img, Mat(), keypoints, descriptors);
        training_descriptors.push_back(descriptors);
        cout << ".";

        drawKeypoints(img, keypoints, img, Scalar::all(-1));
        imshow("temp", img);
        waitKey(0);

    }

    cout << endl;
    closedir(dp);

    cout << "Total descriptors: " << training_descriptors.rows << endl;

    BOWKMeansTrainer bowtrainer(150);

    bowtrainer.add(training_descriptors);

    cout << "cluster BOW features" << endl;
    Mat vocabulary = bowtrainer.cluster();

    Ptr<DescriptorMatcher> matcher(new BFMatcher(NORM_L2));

    BOWImgDescriptorExtractor bowide(extractor, matcher);

    bowide.setVocabulary(vocabulary);
*/
    //setup training data for classifiers
    string filepath;
    char buf[255];
    ifstream ifs("../training.txt");
    do {
        ifs.getline(buf, 255);
        string line(buf);
        istringstream iss(line);
        cout << line << endl;

        iss >> filepath;
        Rect r; char delim;

        iss >> r.x >> delim;
        iss >> r.y >> delim;
        iss >> r.width >> delim;
        iss >> r.height;
        string class_;
        iss >> class_;
        cout << r << " " << class_ << endl;
    } while(!ifs.eof());