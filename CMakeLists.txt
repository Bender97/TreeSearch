cmake_minimum_required(VERSION 2.8)
project("TreeSearch")
find_package( OpenCV REQUIRED )

include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(VocabularyCreation src/mains/VocabularyCreation.cpp)
target_link_libraries(VocabularyCreation ${OpenCV_LIBS})

add_executable(TrainingSetCreation src/mains/TrainingSetCreation.cpp)
target_link_libraries(TrainingSetCreation ${OpenCV_LIBS})

add_executable(ClassifierTraining src/mains/ClassifierTraining.cpp)
target_link_libraries(ClassifierTraining ${OpenCV_LIBS})

add_executable(ClassifierTesting src/mains/ClassifierTesting.cpp)
target_link_libraries(ClassifierTesting ${OpenCV_LIBS})

add_executable(TreeDetector src/mains/TreeDetector.cpp)
target_link_libraries(TreeDetector ${OpenCV_LIBS})
