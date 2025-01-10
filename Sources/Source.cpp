#include"opencv2/opencv.hpp"
#include"opencv2/highgui/highgui.hpp"
#include<iostream>
#include<math.h>
using namespace cv;
using namespace std;

void rgbToGray(const string& inputPath, const string& outputPath) {
    Mat image = imread(inputPath);
    if (image.empty()) {
        cout << "Could not open or find the image: " << inputPath << endl;
        return;
    }
    Mat grayImage(image.rows, image.cols, CV_8UC1);
    for (int y = 0; y < image.rows; ++y) {
        for (int x = 0; x < image.cols; ++x) {
            Vec3b intensity = image.at<Vec3b>(y, x);
            int grayValue = 0.299 * intensity[2] + 0.587 * intensity[1] + 0.114 * intensity[0];
            grayImage.at<uchar>(y, x) = grayValue;
        }
    }

    imshow("Grayscale Image", grayImage);
    waitKey(0);

    imwrite(outputPath, grayImage);
    cout << "Grayscale image saved as: " << outputPath << endl;
}

void adjustBrightness(const string& inputPath, const string& outputPath, double brightnessFactor) {
    Mat image = imread(inputPath);
    if (image.empty()) {
        cout << "Could not open or find the image: " << inputPath << endl;
        return;
    }

    Mat brightenedImage = Mat::zeros(image.size(), image.type());
    if (image.channels() == 3) {
        for (int y = 0; y < image.rows; ++y) {
            for (int x = 0; x < image.cols; ++x) {
                for (int c = 0; c < image.channels(); ++c) {
                    int newValue = image.at<Vec3b>(y, x)[c] + brightnessFactor;
                    brightenedImage.at<Vec3b>(y, x)[c] = saturate_cast<uchar>(newValue);
                }
            }
        }
    }
    else if (image.channels() == 1) {
        for (int y = 0; y < image.rows; ++y) {
            for (int x = 0; x < image.cols; ++x) {
                int newValue = image.at<uchar>(y, x) + brightnessFactor;
                brightenedImage.at<uchar>(y, x) = saturate_cast<uchar>(newValue);
            }
        }
    }

    imshow("Brightness Adjusted Image", brightenedImage);
    waitKey(0);

    imwrite(outputPath, brightenedImage);
    cout << "Brightness adjusted image saved as: " << outputPath << endl;
}

void adjustContrast(const string& inputPath, const string& outputPath, double contrastFactor) {
    Mat image = imread(inputPath);
    if (image.empty()) {
        cout << "Could not open or find the image: " << inputPath << endl;
        return;
    }

    Mat adjustedImage = Mat::zeros(image.size(), image.type());
    if (image.channels() == 3) {
        for (int y = 0; y < image.rows; ++y) {
            for (int x = 0; x < image.cols; ++x) {
                for (int c = 0; c < image.channels(); ++c) {
                    int newValue = image.at<Vec3b>(y, x)[c] * contrastFactor;
                    adjustedImage.at<Vec3b>(y, x)[c] = saturate_cast<uchar>(newValue);
                }
            }
        }
    }
    else if (image.channels() == 1) {
        for (int y = 0; y < image.rows; ++y) {
            for (int x = 0; x < image.cols; ++x) {
                int newValue = image.at<uchar>(y, x) * contrastFactor;
                adjustedImage.at<uchar>(y, x) = saturate_cast<uchar>(newValue);
            }
        }
    }

    imshow("Contrast Adjusted Image", adjustedImage);
    waitKey(0);

    imwrite(outputPath, adjustedImage);
    cout << "Contrast adjusted image saved as: " << outputPath << endl;
}

void applyAverageFilter(const string& inputPath, const string& outputPath, int kernelSize) {

    Mat image = imread(inputPath);
    if (image.empty()) {
        cout << "Could not open or find the image: " << inputPath << endl;
        return;
    }

    Mat filteredImage = Mat::zeros(image.size(), image.type());

    if (image.channels() == 3) {
        for (int y = 0; y < image.rows; ++y) {
            for (int x = 0; x < image.cols; ++x) {

                int startY = max(0, y - kernelSize / 2);
                int endY = min(image.rows - 1, y + kernelSize / 2);
                int startX = max(0, x - kernelSize / 2);
                int endX = min(image.cols - 1, x + kernelSize / 2);

                Vec3f sum(0.0f, 0.0f, 0.0f);
                for (int j = startY; j <= endY; ++j) {
                    for (int i = startX; i <= endX; ++i) {
                        sum += image.at<Vec3b>(j, i);
                    }
                }
                Vec3b average = sum / ((endY - startY + 1) * (endX - startX + 1));

                filteredImage.at<Vec3b>(y, x) = average;
            }
        }
    }
    else if (image.channels() == 1) {
        for (int y = 0; y < image.rows; ++y) {
            for (int x = 0; x < image.cols; ++x) {

                int startY = max(0, y - kernelSize / 2);
                int endY = min(image.rows - 1, y + kernelSize / 2);
                int startX = max(0, x - kernelSize / 2);
                int endX = min(image.cols - 1, x + kernelSize / 2);

                uchar sum = 0;
                for (int j = startY; j <= endY; ++j) {
                    for (int i = startX; i <= endX; ++i) {
                        sum += image.at<uchar>(j, i);
                    }
                }
                uchar average = sum / ((endY - startY + 1) * (endX - startX + 1));

                filteredImage.at<uchar>(y, x) = average;
            }
        }
    }
    
    imshow("Avg Filtered Image", filteredImage);
    waitKey(0);

    imwrite(outputPath, filteredImage);
    cout << "Avg Filtered image saved as: " << outputPath << endl;
}

void applyMedianFilter(const string& inputPath, const string& outputPath, int kernelSize) {
    Mat image = imread(inputPath);
    if (image.empty()) {
        cout << "Could not open or find the image: " << inputPath << endl;
        return;
    }

    Mat filteredImage = Mat::zeros(image.size(), image.type());

    if (image.channels() == 3) {
        for (int y = 0; y < image.rows; ++y) {
            for (int x = 0; x < image.cols; ++x) {

                int startY = max(0, y - kernelSize / 2);
                int endY = min(image.rows - 1, y + kernelSize / 2);
                int startX = max(0, x - kernelSize / 2);
                int endX = min(image.cols - 1, x + kernelSize / 2);

                vector<uchar> values0;
                vector<uchar> values1;
                vector<uchar> values2;
                for (int j = startY; j <= endY; ++j) {
                    for (int i = startX; i <= endX; ++i) {
                        values0.push_back(image.at<Vec3b>(j, i)[0]);
                        values1.push_back(image.at<Vec3b>(j, i)[1]);
                        values2.push_back(image.at<Vec3b>(j, i)[2]);
                    }
                }

                sort(values0.begin(), values0.end());
                sort(values1.begin(), values1.end());
                sort(values2.begin(), values2.end());

                int medianIndex = values0.size() / 2;
                uchar medianValue0 = values0[medianIndex];
                uchar medianValue1 = values1[medianIndex];
                uchar medianValue2 = values2[medianIndex];

                filteredImage.at<Vec3b>(y, x)[0] = medianValue0;
                filteredImage.at<Vec3b>(y, x)[1] = medianValue1;
                filteredImage.at<Vec3b>(y, x)[2] = medianValue2;
            }
        }
    }
    else if (image.channels() == 1) {
        for (int y = 0; y < image.rows; ++y) {
            for (int x = 0; x < image.cols; ++x) {
                int startY = max(0, y - kernelSize / 2);
                int endY = min(image.rows - 1, y + kernelSize / 2);
                int startX = max(0, x - kernelSize / 2);
                int endX = min(image.cols - 1, x + kernelSize / 2);

                vector<uchar> values;
                for (int j = startY; j <= endY; ++j) {
                    for (int i = startX; i <= endX; ++i) {
                        values.push_back(image.at<uchar>(j, i));
                    }
                }

                sort(values.begin(), values.end());

                int medianIndex = values.size() / 2;
                uchar medianValue = values[medianIndex];

                filteredImage.at<uchar>(y, x) = medianValue;
            }
        }
    }

    imshow("Median Filtered Image", filteredImage);
    waitKey(0);

    imwrite(outputPath, filteredImage);
    cout << "Median Filtered image saved as: " << outputPath << endl;
}

void applyGaussianFilter(const string& inputPath, const string& outputPath, int kernelSize) {

    Mat image = imread(inputPath);
    if (image.empty()) {
        cout << "Could not open or find the image: " << inputPath << endl;
        return;
    }

    Mat filteredImage = Mat::zeros(image.size(), image.type());

    double sigma = 0.3 * ((kernelSize - 1) * 0.5 - 1) + 0.8;
    Mat kernel = Mat::zeros(Size(kernelSize, kernelSize), CV_32F);
    int center = kernelSize / 2;
    double total = 0.0;
    for (int i = 0; i < kernelSize; ++i) {
        for (int j = 0; j < kernelSize; ++j) {
            int x = i - center;
            int y = j - center;
            kernel.at<float>(i, j) = exp(-(x * x + y * y) / (2 * sigma * sigma));
            total += kernel.at<float>(i, j);
        }
    }

    for (int i = 0; i < kernelSize; ++i) {
        for (int j = 0; j < kernelSize; ++j) {
            kernel.at<float>(i, j) = kernel.at<float>(i, j) / total;
        }
    }

    if (image.channels() == 3) {

        for (int y = 0; y < image.rows; ++y) {
            for (int x = 0; x < image.cols; ++x) {

                int startY = max(0, y - kernelSize / 2);
                int endY = min(image.rows - 1, y + kernelSize / 2);
                int startX = max(0, x - kernelSize / 2);
                int endX = min(image.cols - 1, x + kernelSize / 2);

                Vec3f sum(0.0f, 0.0f, 0.0f);
                for (int j = startY; j <= endY; ++j) {
                    for (int i = startX; i <= endX; ++i) {
                        sum += image.at<Vec3b>(j, i) * kernel.at<float>(j - y + center, i - x + center);
                    }
                }

                Vec3b sum3b = sum;
                filteredImage.at<Vec3b>(y, x) = sum3b;

            }
        }
    }
    else if (image.channels() == 1) {

        for (int y = 0; y < image.rows; ++y) {
            for (int x = 0; x < image.cols; ++x) {

                int startY = max(0, y - kernelSize / 2);
                int endY = min(image.rows - 1, y + kernelSize / 2);
                int startX = max(0, x - kernelSize / 2);
                int endX = min(image.cols - 1, x + kernelSize / 2);

                float sum = 0;
                for (int j = startY; j <= endY; ++j) {
                    for (int i = startX; i <= endX; ++i) {
                        sum += image.at<uchar>(j, i) * kernel.at<float>(j - y + center, i - x + center);
                    }
                }

                uchar sumR = sum;
                filteredImage.at<uchar>(y, x) = sumR;

            }
        }
    }

    imshow("Gaussian Filtered Image", filteredImage);
    waitKey(0);

    imwrite(outputPath, filteredImage);
    cout << "Gaussian Filtered image saved as: " << outputPath << endl;
}

void applySobelEdgeDetection(const string& inputPath, const string& outputPath) {

    Mat image_temp = imread(inputPath);
    if (image_temp.empty()) {
        cout << "Could not open or find the image: " << inputPath << endl;
        return;
    }
    Mat image_gr(image_temp.rows, image_temp.cols, CV_8UC1);
    if (image_temp.channels() == 3) {
        for (int y = 0; y < image_temp.rows; ++y) {
            for (int x = 0; x < image_temp.cols; ++x) {
                Vec3b intensity = image_temp.at<Vec3b>(y, x);
                int grayValue = 0.299 * intensity[2] + 0.587 * intensity[1] + 0.114 * intensity[0];
                image_gr.at<uchar>(y, x) = grayValue;
            }
        }
    }
    Mat image = Mat::zeros(image_gr.size(), image_gr.type());

    int kernelSize = 3;
    double sigma = 0.3 * ((kernelSize - 1) * 0.5 - 1) + 0.8;
    Mat kernel = Mat::zeros(Size(kernelSize, kernelSize), CV_32F);
    int center = kernelSize / 2;
    double total = 0.0;
    for (int i = 0; i < kernelSize; ++i) {
        for (int j = 0; j < kernelSize; ++j) {
            int x = i - center;
            int y = j - center;
            kernel.at<float>(i, j) = exp(-(x * x + y * y) / (2 * sigma * sigma));
            total += kernel.at<float>(i, j);
        }
    }

    for (int i = 0; i < kernelSize; ++i) {
        for (int j = 0; j < kernelSize; ++j) {
            kernel.at<float>(i, j) = kernel.at<float>(i, j) / total;
        }
    }

    for (int y = 0; y < image_gr.rows; ++y) {
        for (int x = 0; x < image_gr.cols; ++x) {

            int startY = max(0, y - kernelSize / 2);
            int endY = min(image_gr.rows - 1, y + kernelSize / 2);
            int startX = max(0, x - kernelSize / 2);
            int endX = min(image_gr.cols - 1, x + kernelSize / 2);

            float sum = 0;
            for (int j = startY; j <= endY; ++j) {
                for (int i = startX; i <= endX; ++i) {
                    sum += image_gr.at<uchar>(j, i) * kernel.at<float>(j - y + center, i - x + center);
                }
            }

            uchar sumR = sum;
            image.at<uchar>(y, x) = sumR;

        }
    }

    int sobelX[3][3] = { {-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1} };
    int sobelY[3][3] = { {-1, -2, -1}, {0, 0, 0}, {1, 2, 1} };

    Mat gradientMagnitude = Mat::zeros(image.size(), CV_8UC1);

    for (int y = 1; y < image.rows - 1; ++y) {
        for (int x = 1; x < image.cols - 1; ++x) {
            int gx = 0;
            for (int j = 0; j < kernelSize; ++j) {
                for (int i = 0; i < kernelSize; ++i) {
                    gx += sobelX[j][i] * image.at<uchar>(y + j - 1, x + i - 1);
                }
            }
            int gy = 0;
            for (int j = 0; j < kernelSize; ++j) {
                for (int i = 0; i < kernelSize; ++i) {
                    gy += sobelY[j][i] * image.at<uchar>(y + j - 1, x + i - 1);
                }
            }
            gradientMagnitude.at<uchar>(y, x) = sqrt(gx * gx + gy * gy);
        }
    }
    double minValue, maxValue;
    Point minLoc, maxLoc;
    minMaxLoc(gradientMagnitude, &minValue, &maxValue, &minLoc, &maxLoc);
    int nguong = maxValue * 0.2;
    for (int y = 0; y < gradientMagnitude.rows; ++y) {
        for (int x = 0; x < gradientMagnitude.cols; ++x) {
            uchar temp_n = gradientMagnitude.at<uchar>(y, x);
            if (temp_n < nguong) {
                gradientMagnitude.at<uchar>(y, x) = 0;
            }
            else {
                gradientMagnitude.at<uchar>(y, x) = 255;
            }
        }
    }

    imshow("Edge Detected Image", gradientMagnitude);
    waitKey(0);

    imwrite(outputPath, gradientMagnitude);
    cout << "Edge Detected image saved as: " << outputPath << endl;
}

void applyLaplaceEdgeDetection(const string& inputPath, const string& outputPath) {

    Mat image_temp = imread(inputPath);
    if (image_temp.empty()) {
        cout << "Could not open or find the image: " << inputPath << endl;
        return;
    }
    Mat image_gr(image_temp.rows, image_temp.cols, CV_8UC1);
    if (image_temp.channels() == 3) {
        for (int y = 0; y < image_temp.rows; ++y) {
            for (int x = 0; x < image_temp.cols; ++x) {
                Vec3b intensity = image_temp.at<Vec3b>(y, x);
                int grayValue = 0.299 * intensity[2] + 0.587 * intensity[1] + 0.114 * intensity[0];
                image_gr.at<uchar>(y, x) = grayValue;
            }
        }
    }
    Mat image = Mat::zeros(image_gr.size(), image_gr.type());

    int kernelSize = 3;
    double sigma = 0.3 * ((kernelSize - 1) * 0.5 - 1) + 0.8;
    Mat kernel = Mat::zeros(Size(kernelSize, kernelSize), CV_32F);
    int center = kernelSize / 2;
    double total = 0.0;
    for (int i = 0; i < kernelSize; ++i) {
        for (int j = 0; j < kernelSize; ++j) {
            int x = i - center;
            int y = j - center;
            kernel.at<float>(i, j) = exp(-(x * x + y * y) / (2 * sigma * sigma));
            total += kernel.at<float>(i, j);
        }
    }

    for (int i = 0; i < kernelSize; ++i) {
        for (int j = 0; j < kernelSize; ++j) {
            kernel.at<float>(i, j) = kernel.at<float>(i, j) / total;
        }
    }

    for (int y = 0; y < image_gr.rows; ++y) {
        for (int x = 0; x < image_gr.cols; ++x) {

            int startY = max(0, y - kernelSize / 2);
            int endY = min(image_gr.rows - 1, y + kernelSize / 2);
            int startX = max(0, x - kernelSize / 2);
            int endX = min(image_gr.cols - 1, x + kernelSize / 2);

            float sum = 0;
            for (int j = startY; j <= endY; ++j) {
                for (int i = startX; i <= endX; ++i) {
                    sum += image_gr.at<uchar>(j, i) * kernel.at<float>(j - y + center, i - x + center);
                }
            }

            uchar sumR = sum;
            image.at<uchar>(y, x) = sumR;

        }
    }

    int laplaceY[3][3] = { {-1, -1, -1}, {-1, 8, -1}, {-1, -1, -1} };

    Mat gradientMagnitude = Mat::zeros(image.size(), CV_32FC1);

    for (int y = 1; y < image.rows - 1; ++y) {
        for (int x = 1; x < image.cols - 1; ++x) {
            int gy = 0;
            for (int j = 0; j < kernelSize; ++j) {
                for (int i = 0; i < kernelSize; ++i) {
                    gy += laplaceY[j][i] * image.at<uchar>(y + j - 1, x + i - 1);
                }
            }
            gradientMagnitude.at<float>(y, x) = gy;
        }
    }

   
    double minValue, maxValue;
    Point minLoc, maxLoc;
    minMaxLoc(gradientMagnitude, &minValue, &maxValue, &minLoc, &maxLoc);
    int nguong = maxValue * 0.08;
    Mat gradientMagnitude_temp = Mat::zeros(gradientMagnitude.size(), gradientMagnitude.type());
    for (int y = 0; y < gradientMagnitude.rows; ++y) {
        for (int x = 0; x < gradientMagnitude.cols; ++x) {
            float temp_n = gradientMagnitude.at<float>(y, x);
            if (temp_n < nguong) {
                gradientMagnitude_temp.at<float>(y, x) = 0;
            }
            else {
                gradientMagnitude_temp.at<float>(y, x) = 255;
            }
        }
    }

    Mat gradientMagnitude_xuat = Mat::zeros(gradientMagnitude.size(), CV_8UC1);
    for (int y = 1; y < gradientMagnitude.rows - 1; ++y) {
        for (int x = 1; x < gradientMagnitude.cols - 1; ++x) {
            float center_temp = gradientMagnitude_temp.at<float>(y, x);
            bool check = true;
            int count = 0;
            for (int dy = -1; dy <= 1; ++dy) {
                for (int dx = -1; dx <= 1; ++dx) {
                    float neighbor_temp = gradientMagnitude_temp.at<float>(y + dy, x + dx);
                    if (center_temp * neighbor_temp == 0) {
                        count++;
                    }
                }
            }
            if (count > 6) {
                check = false;
            }
            float center = gradientMagnitude.at<float>(y, x);
            bool isEdge = false;
            if (center_temp == 255 && check == true) {
                for (int dy = -1; dy <= 1; ++dy) {
                    for (int dx = -1; dx <= 1; ++dx) {
                        if (dx == 0 && dy == 0) continue;
                        float neighbor = gradientMagnitude.at<float>(y + dy, x + dx);
                        if (center * neighbor < 0) {
                            isEdge = true;
                            break;
                        }
                    }
                    if (isEdge) break;
                }

                if (isEdge) {
                    gradientMagnitude_xuat.at<uchar>(y, x) = 255;
                }
            }   
        }
    }
    
    imshow("Edge Detected Image", gradientMagnitude_xuat);
    waitKey(0);

    imwrite(outputPath, gradientMagnitude_xuat);
    cout << "Edge Detected image saved as: " << outputPath << endl;
}

int main(int argc, char* argv[]) {

    string operation = argv[1];
    string inputPath = argv[2];
    string outputPath = argv[3];

    if (operation == "-rgb2gray") {
        rgbToGray(inputPath, outputPath);
    }
    else if (operation == "-brightness") {
        double brightnessFactor = stod(argv[4]);
        adjustBrightness(inputPath, outputPath, brightnessFactor);
    }
    else if (operation == "-constrast") {
        double contrastFactor = stod(argv[4]);
        adjustContrast(inputPath, outputPath, contrastFactor);
    }
    else if (operation == "-avg") {
        int kernelSize = stoi(argv[4]);
        applyAverageFilter(inputPath, outputPath, kernelSize);
    }
    else if (operation == "-med") {
        int kernelSize = stoi(argv[4]);
        applyMedianFilter(inputPath, outputPath, kernelSize);
    }
    else if (operation == "-gau") {
        int kernelSize = stoi(argv[4]);
        applyGaussianFilter(inputPath, outputPath, kernelSize);
    }
    else if (operation == "-sobel") {
        applySobelEdgeDetection(inputPath, outputPath);
    }
    else if (operation == "-laplace") {
        applyLaplaceEdgeDetection(inputPath, outputPath);
    }
    else {
        cout << "Invalid operation specified!" << endl;
        return -1;
    }
    return 0;
}