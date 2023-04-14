// SHIJUEMEITI(3).cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
//

//导入工具包
#include<opencv2/opencv.hpp>
#include<iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include<stdlib.h>


using namespace cv;
using namespace std;


//直接手撸一个均值滤波函数

void My_MeanFilter(const Mat& My_image, Mat& Target, int kernel_size)  // My_image 为待处理的图像，Target为目标图像，kernel_size为卷积核的尺寸，卷积核的尺寸必须为奇数，不然无法进行卷积操作
{
	int* kernel = new int[kernel_size * kernel_size];           // 设置卷积核的大小

	for (int i = 0; i < kernel_size * kernel_size; i++)         // 首先以均值滤波为例，初始化其参数均为1
		kernel[i] = 1;


	Mat Container;	//存放像素数据
	int padding_len = kernel_size / 2;

	//类似于深度学习中的Padding操作，为卷积后的像素数据暂存矩阵添加行与列
	Container.create(Size(My_image.cols + 2 * padding_len, My_image.rows + 2 * padding_len), My_image.type());
	Target.create(Size(My_image.cols, My_image.rows), My_image.type());

	
	//由于为彩色图片，所以图片有三个维度，自然需要获取RGB三通道，然后分别进行处理
	int channel = My_image.channels();
	//定义两个指针，指向待处理图像和暂存矩阵
	uchar* pM = My_image.data;
	uchar* pC = Container.data;


	//行与列都进行Padding
	for (int row = 0; row < Container.rows; row++) //遍历每一行
	{
		for (int col = 0; col < Container.cols; col++) //遍历每一列
		{
			for (int num_channel = 0; num_channel < channel; num_channel++)	//遍历每一个通道
			{
				if (row >= padding_len && row < Container.rows - padding_len && col >= padding_len && col < Container.cols - padding_len)
					pC[(Container.cols * row + col) * channel + num_channel] = pM[(My_image.cols * (row - padding_len) + col - padding_len) * channel + num_channel]; //如果均满足，则可按照My_image的指针对container中的像素数据进行处理
				else  
					pC[(Container.cols * row + col) * channel + num_channel] = 0;//否则就进行补0操作
			}
		}
	}


	//定义指针，指向目标图像
	uchar* pd = Target.data;
	pC = Container.data;
	for (int row = padding_len; row < Container.rows - padding_len; row++)//乘积求和
	{
		for (int col = padding_len; col < Container.cols - padding_len; col++)
		{
			for (int c = 0; c < channel; c++)
			{
				short t = 0;
				for (int x = - padding_len; x <= padding_len; x++)
				{
					for (int y = -padding_len; y <= padding_len; y++)
					{
						t += kernel[(padding_len + x) * kernel_size + y + padding_len] * pC[((row + x) * Container.cols + col + y) * channel + c]; //从暂存矩阵（已经Padding）中使用均值滤波，得到的结果赋值为t
					}
				}
				//此处注意要防止数据溢出(int是32位数据，ushort是16位数据)
				pd[(Target.cols * (row - padding_len) + col - padding_len) * channel + c] = saturate_cast<ushort> (t / (kernel_size * kernel_size)); //将t的结果首先做kernel_size的平均，再通过指针放在目标图像矩阵当中
			}
		}
	}

	delete[] kernel;       // 将均值滤波定义的kernel删除，节约内存
}





int main(int argc, char** argv)
{

	//选择OpenCV的训练结果路径
	string xmlPath = "C:/Users/lenovo/Desktop/C++ PROJECT/opencv/build/etc/haarcascades/haarcascade_frontalface_default.xml";


	//载入待处理图像
	Mat My_image = imread("./Shelby.png");
	resize(My_image, My_image, Size(1146, 715));
	imshow("输入图像", My_image);

	//载入检测器
	CascadeClassifier detector;
	detector.load(xmlPath);


	if (!detector.load(xmlPath))   //加载训练文件  
	{
		cout << "不能加载指定的xml文件" << endl;

		return -1;

	}

	//指定检测人脸
	vector<Rect> faces;

	detector.detectMultiScale(My_image, faces, 1.1, 3, 0, Size(30, 30));

	for (size_t t = 0; t < faces.size(); t++)
	{

		rectangle(My_image, faces[t], Scalar(0, 0, 255), 2, 8); //使用矩形框描绘出检测区域
	}

	imshow("结果", My_image);

	waitKey(0);

	return 0;
	






	
}

