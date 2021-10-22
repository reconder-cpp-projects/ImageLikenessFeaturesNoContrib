
#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/features2d.hpp"
#include <opencv2/core/utils/logger.hpp>
using namespace cv;
#include <iostream>
#include <string>


struct Image
{
    String filename;
    long keypoints_count;
    Mat descriptors;
    //Используется ORB(Oriented FAST and Rotated BRIEF) алгоритv c кол-вом features = 400.
    Image(String fn, std::shared_ptr<ORB>& detector) : filename(fn) 
    {
        auto img = imread(samples::findFile(fn), IMREAD_GRAYSCALE);
        std::vector<KeyPoint> keypoints;
        //Вычисляем keypoints и descriptors для каждого изображения при помощи ORB алгоритма
        detector->detectAndCompute(img, noArray(), keypoints, descriptors);
        keypoints_count = keypoints.size();
        // descriptors.convertTo(descriptors, CV_32F); // Если вдруг захочется использовать с FLANN, хотя это глупо
    };
};



void compare_images(Image img1, Image img2, short threshold)
{
    

    /*Создаем матчер, который будет искать соответствия брутфорсом, тк ORB выдает бинарные дескрипторы*/
    std::shared_ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(DescriptorMatcher::BRUTEFORCE_L1);
    std::vector< std::vector<DMatch> > matches;
    matcher->knnMatch(img1.descriptors, img2.descriptors, matches, 2);
    /*Фильтруем совпадения при помощи теста Лове. Он сравнивает дистанции совпадений и оставляет только те, у который отношение самой маленькой дистанции ко второй
      самой маленькой высоко.*/
    const float ratio_threshold = 0.7f;
    long good_matches = 0;
    for (size_t i = 0; i < matches.size(); i++)
    {
        if (matches[i][0].distance < ratio_threshold * matches[i][1].distance)
        {
            good_matches++;
        }
    }
    //Вычисляем схожесть изображений как отношение хороших совпадений к количеству минимальному количеству углов(features).
    short likeness = (short)(100 * good_matches / std::min(img1.keypoints_count, img2.keypoints_count));

    if (likeness >= threshold) std::cout << img1.filename << " " << img2.filename << " " << likeness << "%\n";
}






std::vector<String> parse_input()
{
    std::cout << "Input images' filenames:" << std::endl;
    std::vector<String> inputs;
    inputs.reserve(10);
    String line;
    std::getline(std::cin, line);
    line = "00";
    while (std::getline(std::cin, line) && !line.empty())
    {
        inputs.push_back(line);
    }
    return inputs;
}





void main()
{
    cv::utils::logging::setLogLevel(utils::logging::LogLevel::LOG_LEVEL_SILENT);
    int threshold;
    std::cout << "Enter the threshold:" << std::endl;
    std::cin >> threshold;
    threshold = 0;
    auto inputs = parse_input();
    std::shared_ptr<ORB> detector = ORB::create(400);
    std::vector<Image> images;
    images.reserve(inputs.size());
    String blank;
    for (auto& input : inputs)
    {
        images.push_back(Image(inputs.back(), detector));
        inputs.pop_back();
    }

    for (int i = 0; i < images.size(); i++)
        for (int j = i + 1; j < images.size(); j++)
            compare_images(images[i], images[j], threshold);

}