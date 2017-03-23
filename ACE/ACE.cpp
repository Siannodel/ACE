// ACE.cpp : �������̨Ӧ�ó������ڵ㡣
//

#include "stdafx.h"
#include <opencv2\opencv.hpp>
#include <iostream>
using namespace std;
using namespace cv;

//�������ͼ��(��ͨ��)
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
//�������ͼ��(��ͨ��OR��ͨ��)
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
//���û���ͼ�����ֲ���׼��(��ͨ��)
Mat Localstd_fast(const Mat& image, int d)
{

	Mat result(image.size(), CV_64FC1);
	//�߽����
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
//���û���ͼ�����ֲ���׼��(��ͨ��OR��ͨ��)
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
	//�߽����
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
//Intergral��Localstd_fast����
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
//Intergral_2��Localstd_fast_2����
int main()
{
	//Mat mat(5, 5, CV_8UC3, Scalar(1,2,3));
	//cout << mat << endl;
	//cout << Localstd_fast_2(mat,1) << endl;
	Mat image = imread("image\\1.jpg");
	//Mat gray;
	//cvtColor(image, gray, CV_BGR2GRAY);
	double time;
	time = (double)getTickCount();
	//Mat result = Localstd_fast_2(image, 10);
	Mat result = Intergral_2(image);
	time = 1000 * ((double)getTickCount() - time) / getTickFrequency();
	cout << time << endl;
	result.convertTo(result, image.type());
	imshow("result", result);
	waitKey(0);
	system("pause");
	return 0;
	return 0;
}