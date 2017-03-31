// ACE.cpp : 定义控制台应用程序的入口点。
//

#include "stdafx.h"
#include <opencv2\opencv.hpp>
#include <iostream>
using namespace std;
using namespace cv;

//计算积分图像(单通道)
Mat Intergral(const Mat& image)
{
	Mat result(image.size(), CV_64FC1);
	int nr = image.rows;
	int nc = image.cols;
	for (int i = 0; i < nr; i++)
	{
		const double* inData = image.ptr<double>(i);
		double* outData = result.ptr<double>(i);
		if (i != 0)
		{
			const double* outData_up = result.ptr<double>(i - 1);
			for (int j = 0; j < nc; j++)
			{
				if (j != 0)
				{
					outData[j] = inData[j] + outData_up[j] + outData[j - 1] - outData_up[j - 1];
				}
				else
				{
					outData[j] = inData[j] + outData_up[j];
				}
			}
		}
		else
		{
			for (int j = 0; j < nc; j++)
			{

				if (j != 0)
				{
					outData[j] = inData[j] + outData[j - 1];
				}
				else
				{
					outData[j] = inData[j];
				}
			}
		}
	}
	return result;
}
//计算积分图像(单通道OR三通道)
Mat Intergral_2(const Mat& image)
{
	Mat result;
	Mat image_2;
	if (image.channels() == 1)
	{
		image.convertTo(image_2, CV_64FC1);
		result.create(image.size(), CV_64FC1);

	}
	else if (image.channels() == 3)
	{
		image.convertTo(image_2, CV_64FC3);
		result.create(image.size(), CV_64FC3);
	}
	//cout << image_2 << endl;
	int c = image_2.channels();
	int nr = image_2.rows;
	int nc = image_2.cols*c;
	for (int i = 0; i < nr; i++)
	{
		const double* inData = image_2.ptr<double>(i);
		double* outData = result.ptr<double>(i);
		if (i != 0)
		{
			const double* outData_up = result.ptr<double>(i - 1);
			for (int j = 0; j < nc; j++)
			{
				if (j >= c)
				{
					outData[j] = inData[j] + outData_up[j] + outData[j - c] - outData_up[j - c];
				}
				else
				{
					outData[j] = inData[j] + outData_up[j];
				}
			}
		}
		else
		{
			for (int j = 0; j < nc; j++)
			{

				if (j >= c)
				{
					outData[j] = inData[j] + outData[j - c];
				}
				else
				{
					outData[j] = inData[j];
				}
			}
		}
	}
	return result;
}
//利用积分图像计算局部标准差(单通道)
Mat Localstd_fast(const Mat& image, int d)
{

	Mat result(image.size(), CV_64FC1);
	//边界填充
	Mat image_big;
	copyMakeBorder(image, image_big, d + 1, d + 1, d + 1, d + 1, BORDER_REFLECT_101);
	image_big.convertTo(image_big, CV_64FC1);
	Mat image_big_2 = image_big.mul(image_big);
	Mat Intergral_image1 = Intergral(image_big);
	Mat Intergral_image2 = Intergral(image_big_2);

	int N = (2 * d + 1)*(2 * d + 1);
	int nr = image.rows;
	int nc = image.cols;
	for (int i = 0; i < nr; i++)
	{
		double* outData = result.ptr<double>(i);
		double* inDataUp1 = Intergral_image1.ptr<double>(i);
		double* inDataUp2 = Intergral_image2.ptr<double>(i);
		double* inDataDown1 = Intergral_image1.ptr<double>(i + 2 * d + 1);
		double* inDataDown2 = Intergral_image2.ptr<double>(i + 2 * d + 1);
		for (int j = 0; j < nc; j++)
		{
			double sumi1 = inDataDown1[j + (2 * d + 1)] + inDataUp1[j] - inDataUp1[j + (2 * d + 1)] - inDataDown1[j];
			double sumi2 = inDataDown2[j + (2 * d + 1)] + inDataUp2[j] - inDataUp2[j + (2 * d + 1)] - inDataDown2[j];
			outData[j] = (sumi2 - (sumi1*sumi1) / N) / N;
		}
	}
	cv::sqrt(result, result);
	return result;
}
//利用积分图像计算局部标准差(单通道OR三通道)
Mat Localstd_fast_2(const Mat& image, int d)
{
	Mat result;
	if (image.channels() == 1)
	{
		result.create(image.size(), CV_64FC1);

	}
	else if (image.channels() == 3)
	{
		result.create(image.size(), CV_64FC3);
	}
	//边界填充
	Mat image_big;
	copyMakeBorder(image, image_big, d + 1, d + 1, d + 1, d + 1, BORDER_REFLECT_101);
	image_big.convertTo(image_big, result.type());
	Mat image_big_2 = image_big.mul(image_big);
	Mat Intergral_image1 = Intergral_2(image_big);
	Mat Intergral_image2 = Intergral_2(image_big_2);

	int N = (2 * d + 1)*(2 * d + 1);
	int c = image.channels();
	int nr = image.rows;
	int nc = image.cols*c;

	//cout << Intergral_image1 << endl;
	//cout << Intergral_image2 << endl;

	for (int i = 0; i < nr; i++)
	{
		double* outData = result.ptr<double>(i);
		double* inDataUp1 = Intergral_image1.ptr<double>(i);
		double* inDataUp2 = Intergral_image2.ptr<double>(i);
		double* inDataDown1 = Intergral_image1.ptr<double>(i + 2 * d + 1);
		double* inDataDown2 = Intergral_image2.ptr<double>(i + 2 * d + 1);
		for (int j = 0; j < nc; j++)
		{
			//cout << inDataUp2[j] << " " << inDataUp2[j + (2 * d + 1)*c] << endl;
			//cout << inDataDown2[j] << " " << inDataDown2[j + (2 * d + 1)*c] << endl;
			double sumi1 = inDataDown1[j + (2 * d + 1)*c] + inDataUp1[j] - inDataUp1[j + (2 * d + 1)*c] - inDataDown1[j];
			double sumi2 = inDataDown2[j + (2 * d + 1)*c] + inDataUp2[j] - inDataUp2[j + (2 * d + 1)*c] - inDataDown2[j];
			outData[j] = (sumi2 - (sumi1*sumi1) / N) / N;
		}
	}
	cv::sqrt(result, result);
	return result;
}
//利用积分图像计算局部平均值与局部标准差（单通道OR三通道）
int Local_MeanStd(const Mat &_image, Mat &_mean, Mat &_std, int d)
{
	if (_image.channels() == 1)
	{
		_mean.create(_image.size(), CV_64FC1);
		_std.create(_image.size(), CV_64FC1);

	}
	else if (_image.channels() == 3)
	{
		_mean.create(_image.size(), CV_64FC3);
		_std.create(_image.size(), CV_64FC3);
	}
	//边界填充
	Mat image_big;
	copyMakeBorder(_image, image_big, d + 1, d + 1, d + 1, d + 1, BORDER_REFLECT_101);
	image_big.convertTo(image_big, _mean.type());
	Mat image_big_2 = image_big.mul(image_big);
	Mat Intergral_image1 = Intergral_2(image_big);
	Mat Intergral_image2 = Intergral_2(image_big_2);

	int N = (2 * d + 1)*(2 * d + 1);
	int c = _image.channels();
	int nr = _image.rows;
	int nc = _image.cols*c;

	//cout << Intergral_image1 << endl;
	//cout << Intergral_image2 << endl;

	for (int i = 0; i < nr; i++)
	{
		double* outData1 = _mean.ptr<double>(i);
		double* outData2 = _std.ptr<double>(i);
		double* inDataUp1 = Intergral_image1.ptr<double>(i);
		double* inDataUp2 = Intergral_image2.ptr<double>(i);
		double* inDataDown1 = Intergral_image1.ptr<double>(i + 2 * d + 1);
		double* inDataDown2 = Intergral_image2.ptr<double>(i + 2 * d + 1);
		for (int j = 0; j < nc; j++)
		{
			
			double sumi1 = inDataDown1[j + (2 * d + 1)*c] + inDataUp1[j] - inDataUp1[j + (2 * d + 1)*c] - inDataDown1[j];
			double sumi2 = inDataDown2[j + (2 * d + 1)*c] + inDataUp2[j] - inDataUp2[j + (2 * d + 1)*c] - inDataDown2[j];
			outData1[j] = sumi1 / N;
			outData2[j] = (sumi2 - sumi1*outData1[j]) / N;
		}
	}
	cv::sqrt(_std, _std);
	return 0;
}
//ACE
int ACE(const Mat &_image, Mat &_result, int _d, int _Scale, int _MaxCG)
{
	Mat localmean, localstd;
	Local_MeanStd(_image, localmean, localstd, _d);
	if (_image.channels() == 1)
	{
		_result.create(_image.size(), CV_64FC1);
	}
	else if (_image.channels() == 3)
	{
		_result.create(_image.size(), CV_64FC3);	
	}
	Mat mean_m, std_m;
	meanStdDev(_image, mean_m, std_m); 
	double std[3];
	std[0] = std_m.at<double>(0, 0);
	std[1] = std_m.at<double>(1, 0);
	std[2] = std_m.at<double>(2, 0);

	int c = _image.channels();
	int nr = _image.rows;
	int nc = _image.cols*c;
	double CG;
	for (int i = 0; i < nr; i++)
	{
		double* meanData = localmean.ptr<double>(i);
		double* stdData = localstd.ptr<double>(i);
		const uchar* imageData = _image.ptr<uchar>(i);
		double* outData = _result.ptr<double>(i);
		for (int j = 0; j < nc; j++)
		{
			CG = std[j % 3] / stdData[j];
			if (CG > _MaxCG)
				CG = _MaxCG;
			outData[j] = imageData[j] + _Scale*CG*(int(imageData[j]) - meanData[j]);
		}
	}
	_result.convertTo(_result, CV_8UC3);
	return 0;
}
//Intergral与Localstd_fast测试
int main1()
{
	Mat image = imread("image\\1.jpg");
	Mat gray;
	cvtColor(image, gray, CV_BGR2GRAY);
	double time;
	time = (double)getTickCount();
	Mat result = Localstd_fast(gray, 10);
	time = 1000 * ((double)getTickCount() - time) / getTickFrequency();
	cout << time << endl;
	result.convertTo(result, CV_8UC1);
	imshow("result", result);
	waitKey(0);
	system("pause");
	return 0;
}
//Intergral_2与Localstd_fast_2测试
int main2()
{
	//Mat mat(5, 5, CV_8UC3, Scalar(1,2,3));
	//cout << mat << endl;
	//cout << Localstd_fast_2(mat,1) << endl;
	Mat image = imread("image\\1.jpg");
	//Mat gray;
	//cvtColor(image, gray, CV_BGR2GRAY);
	double time;
	time = (double)getTickCount();
	Mat result = Localstd_fast_2(image, 10);
	//Mat result = Intergral_2(image);
	time = 1000 * ((double)getTickCount() - time) / getTickFrequency();
	cout << time << endl;
	result.convertTo(result, image.type());
	imshow("result", result);
	waitKey(0);
	system("pause");
	return 0;
	return 0;
}
//Local_MeanStd测试
int main3()
{
	Mat image = imread("image\\1.jpg");
	double time;
	Mat Localmean, Localstd;
	time = (double)getTickCount();
	Local_MeanStd(image, Localmean, Localstd, 5);
	//blur(image, Localmean, Size(11, 11));
	time = 1000 * ((double)getTickCount() - time) / getTickFrequency();
	cout << time << endl;
	Localmean.convertTo(Localmean, image.type());
	Localstd.convertTo(Localstd, image.type());
	imshow("Localmean", Localmean);
	imshow("Localstd", Localstd);
	waitKey(0);
	system("pause");
	return 0;
}
int main()
{
	Mat image = imread("image\\4.png");
	Mat result;
	double time = (double)getTickCount();
	ACE(image, result, 20, 1, 3);
	time = 1000 * ((double)getTickCount() - time) / getTickFrequency();
	cout << time << endl;
	imshow("result",result);
	waitKey(0);
	system("pause");
	return 0;
}