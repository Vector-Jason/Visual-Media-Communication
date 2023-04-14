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




	//读取待处理图像
	Mat My_image = imread("./Shelby.png"); //利用相对路径进行读取
	namedWindow("待处理图像", WINDOW_AUTOSIZE);
	imshow("待处理图像", My_image);

	// 获取图像处理位置
	Rect rect(390, 135, 370, 350);//四个参数分别代表窗口中心的X、Y以及窗口本身的宽与高
	
	Mat roi = My_image(rect);
	imshow("ROI", roi);


	//对人脸进行滤波处理
	int kernel = 15; //设置卷积核kernel的大小，此时效果最佳
	//特别地，通过实际操作发现：卷积核越小，模糊效果越差，反之，将从模糊效果逐渐变为提取边缘特征
	Mat image = roi.clone();
	My_MeanFilter(image, image, kernel); //对roi区域图像数据进行均值滤波
	namedWindow("处理后", WINDOW_AUTOSIZE);
	imshow("处理后", image);


	//将处理后的图像数据覆盖在ROI上
	image.copyTo(roi);
	namedWindow("结果");
	imshow("结果", My_image);


	waitKey(0);//一直等待按键。
	return 0;


}

