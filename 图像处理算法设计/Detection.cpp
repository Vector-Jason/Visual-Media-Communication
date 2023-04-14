// SHIJUEMEITI(3).cpp : ���ļ����� "main" ����������ִ�н��ڴ˴���ʼ��������
//

//���빤�߰�
#include<opencv2/opencv.hpp>
#include<iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include<stdlib.h>


using namespace cv;
using namespace std;


//ֱ����ߣһ����ֵ�˲�����

void My_MeanFilter(const Mat& My_image, Mat& Target, int kernel_size)  // My_image Ϊ�������ͼ��TargetΪĿ��ͼ��kernel_sizeΪ����˵ĳߴ磬����˵ĳߴ����Ϊ��������Ȼ�޷����о������
{
	int* kernel = new int[kernel_size * kernel_size];           // ���þ���˵Ĵ�С

	for (int i = 0; i < kernel_size * kernel_size; i++)         // �����Ծ�ֵ�˲�Ϊ������ʼ���������Ϊ1
		kernel[i] = 1;


	Mat Container;	//�����������
	int padding_len = kernel_size / 2;

	//���������ѧϰ�е�Padding������Ϊ���������������ݴ�������������
	Container.create(Size(My_image.cols + 2 * padding_len, My_image.rows + 2 * padding_len), My_image.type());
	Target.create(Size(My_image.cols, My_image.rows), My_image.type());

	
	//����Ϊ��ɫͼƬ������ͼƬ������ά�ȣ���Ȼ��Ҫ��ȡRGB��ͨ����Ȼ��ֱ���д���
	int channel = My_image.channels();
	//��������ָ�룬ָ�������ͼ����ݴ����
	uchar* pM = My_image.data;
	uchar* pC = Container.data;


	//�����ж�����Padding
	for (int row = 0; row < Container.rows; row++) //����ÿһ��
	{
		for (int col = 0; col < Container.cols; col++) //����ÿһ��
		{
			for (int num_channel = 0; num_channel < channel; num_channel++)	//����ÿһ��ͨ��
			{
				if (row >= padding_len && row < Container.rows - padding_len && col >= padding_len && col < Container.cols - padding_len)
					pC[(Container.cols * row + col) * channel + num_channel] = pM[(My_image.cols * (row - padding_len) + col - padding_len) * channel + num_channel]; //��������㣬��ɰ���My_image��ָ���container�е��������ݽ��д���
				else  
					pC[(Container.cols * row + col) * channel + num_channel] = 0;//����ͽ��в�0����
			}
		}
	}


	//����ָ�룬ָ��Ŀ��ͼ��
	uchar* pd = Target.data;
	pC = Container.data;
	for (int row = padding_len; row < Container.rows - padding_len; row++)//�˻����
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
						t += kernel[(padding_len + x) * kernel_size + y + padding_len] * pC[((row + x) * Container.cols + col + y) * channel + c]; //���ݴ�����Ѿ�Padding����ʹ�þ�ֵ�˲����õ��Ľ����ֵΪt
					}
				}
				//�˴�ע��Ҫ��ֹ�������(int��32λ���ݣ�ushort��16λ����)
				pd[(Target.cols * (row - padding_len) + col - padding_len) * channel + c] = saturate_cast<ushort> (t / (kernel_size * kernel_size)); //��t�Ľ��������kernel_size��ƽ������ͨ��ָ�����Ŀ��ͼ�������
			}
		}
	}

	delete[] kernel;       // ����ֵ�˲������kernelɾ������Լ�ڴ�
}





int main(int argc, char** argv)
{

	//ѡ��OpenCV��ѵ�����·��
	string xmlPath = "C:/Users/lenovo/Desktop/C++ PROJECT/opencv/build/etc/haarcascades/haarcascade_frontalface_default.xml";


	//���������ͼ��
	Mat My_image = imread("./Shelby.png");
	resize(My_image, My_image, Size(1146, 715));
	imshow("����ͼ��", My_image);

	//��������
	CascadeClassifier detector;
	detector.load(xmlPath);


	if (!detector.load(xmlPath))   //����ѵ���ļ�  
	{
		cout << "���ܼ���ָ����xml�ļ�" << endl;

		return -1;

	}

	//ָ���������
	vector<Rect> faces;

	detector.detectMultiScale(My_image, faces, 1.1, 3, 0, Size(30, 30));

	for (size_t t = 0; t < faces.size(); t++)
	{

		rectangle(My_image, faces[t], Scalar(0, 0, 255), 2, 8); //ʹ�þ��ο������������
	}

	imshow("���", My_image);

	waitKey(0);

	return 0;
	






	
}

