cmake_minimum_required(VERSION 2.8)
project("TreeSearch")
find_package( OpenCV REQUIRED )

include_directories(${OpenCV_INCLUDE_DIRS})

set (UTILITIES_H include/ImagesUtility.h include/FilesUtility.h include/VocabularyUtility.h include/DatasetUtility.h include/ClassifierUtility.h)
set (UTILITIES_CPP src/ImagesUtility.cpp src/FilesUtility.cpp src/VocabularyUtility.cpp src/DatasetUtility.cpp src/ClassifierUtility.cpp)
set (DETECTOR_SRC include/Detector.h src/Detector.cpp)

add_executable(VocabularyCreation ${UTILITIES_H} ${UTILITIES_CPP} src/mains/VocabularyCreation.cpp)
target_link_libraries(VocabularyCreation ${OpenCV_LIBS})

add_executable(TrainingSetCreation ${UTILITIES_H} ${UTILITIES_CPP} src/mains/TrainingSetCreation.cpp)
target_link_libraries(TrainingSetCreation ${OpenCV_LIBS})

add_executable(ClassifierTraining ${UTILITIES_H} ${UTILITIES_CPP} src/mains/ClassifierTraining.cpp)
target_link_libraries(ClassifierTraining ${OpenCV_LIBS})

add_executable(ClassifierTesting ${UTILITIES_H} ${UTILITIES_CPP} src/mains/ClassifierTesting.cpp)
target_link_libraries(ClassifierTesting ${OpenCV_LIBS})

add_executable(TreeDetector ${UTILITIES_H} ${UTILITIES_CPP} ${DETECTOR_SRC} src/mains/TreeDetector.cpp)
target_link_libraries(TreeDetector ${OpenCV_LIBS})
