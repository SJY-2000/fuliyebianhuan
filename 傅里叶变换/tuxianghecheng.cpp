#include <iostream>
#include <opencv2/opencv.hpp>
using namespace cv;
using namespace std;

int removeFrequnce();
//int dftDemo();
//int ifftDemo();
//int mouseROI();

void on_mouse(int EVENT, int x, int y, int flags, void* userdata);
int selectPolygon(cv::Mat srcMat, cv::Mat &dstMat);
int calcVisibalMag(cv::Mat srcMat, cv::Mat & dstMat);
int calcVisbalDft(cv::Mat srcMat, cv::Mat & magMat, cv::Mat & ph, double & normVal);
int calcDft2Image(Mat srcMat, cv::Mat magMat, cv::Mat ph, double normVal, cv::Mat &dstMat);


std::vector<Point>  mousePoints;//�洢����points����������
Point points;//�洢�����������

int main()
{
	//��ʼ��ʱ
	double start = static_cast<double>(getTickCount());

	//dftDemo();
	//mouseROI();
	removeFrequnce();
	//������ʱ
	double time = ((double)getTickCount() - start) / getTickFrequency();
	//��ʾʱ��
	cout << "processing time:" << time / 1000 << "ms" << endl;

	//�ȴ�������Ӧ�����������������
	system("pause");
	return 0;
}


/*******************************************
1.����һ��ͼƬ����������ӻ��ķ�ֵ��
2.�ٷ�ֵ��ͼ�ϣ�ͨ�����ѡ����Ҫȥ����Ƶ��
3.ȥ����ѡ����źţ�Ȼ��ԭͼ��
4.����n����ֵ����maskMat
**********************************************/
int removeFrequnce()
{
	cv::Mat dstMat, dstMat_1, dstMat_2;
	cv::Mat src_1 = imread("G:\\picture\\yu.png", 0);
	cv::Mat src_2 = imread("G:\\picture\\sagan.png", 0);

	imshow("src_1", src_1);
	imshow("src_2", src_2);

	cv::Mat magMat_1, magMat_2;
	cv::Mat phMat, phMat_1, phMat_2;
	cv::Mat maskMat_1, maskMat_2;
	magMat_1 = src_1.clone();
	phMat_1 = src_1.clone();
	magMat_2 = src_2.clone();
	phMat_2 = src_2.clone();
	double normVal_1, normVal_2;


	if (src_1.empty()) {
		std::cout << "failed to read image!:" << std::endl;
		return -1;
	}
	if (src_2.empty()) {
		std::cout << "failed to read image!:" << std::endl;
		return -1;
	}

	//������ӻ���mag���Լ���λ�ף��Լ���һ��ϵ��
	calcVisbalDft(src_1, magMat_1, phMat_1, normVal_1);
	calcVisbalDft(src_2, magMat_2, phMat_2, normVal_2);
	imshow("magMat_1", magMat_1);
	imshow("magMat_2", magMat_2);
	waitKey(0);
	//�ڷ�ֵ���ϣ�ͨ�����ѡ����Ҫȥ����Ƶ��
	selectPolygon(magMat_1, maskMat_1);

	Mat proceMag_1, proceMag_2;

	magMat_1 = magMat_1.mul(maskMat_1);//mul�������������Ӧλ�ĳ˻�

	maskMat_2 = 1 - maskMat_1;//ȡ����Ĥ
	magMat_2 = magMat_2.mul(maskMat_2);

	proceMag_1 = magMat_1 * 255;
	proceMag_2 = magMat_2 * 255;
	imwrite("�����Ƶ��1.jpg", proceMag_1);
	imwrite("�����Ƶ��2.jpg", proceMag_2);

	//��任
	calcDft2Image(src_1, magMat_1, phMat_1, normVal_1, dstMat_1);
	calcDft2Image(src_2, magMat_2, phMat_2, normVal_2, dstMat_2);

	addWeighted(dstMat_1, 0.5, dstMat_2, 0.5, 3, dstMat);
	imshow("dstMat", dstMat);
	imshow("dstMat_1", dstMat_1);
	imshow("dstMat_2", dstMat_2);
	waitKey(0);
	return 0;
}


/***************************************�����Ӧ����*******************************************/
void on_mouse(int EVENT, int x, int y, int flags, void* userdata)
{

	Mat hh;
	hh = *(Mat*)userdata;//userdataָ��ͼ��ָ��
	Point p(x, y);
	switch (EVENT)
	{
	case EVENT_LBUTTONDOWN:
	{
		points.x = x;     //pointsһ��ʼ�����ȫ�ֱ���
		points.y = y;
		mousePoints.push_back(points);
		circle(hh, points, 4, Scalar(255, 255, 255), -1);
		imshow("mouseCallback", hh);
	}
	break;
	}

}


/*ѡ��Ƶ��ȥ������*/
int selectPolygon(cv::Mat srcMat, cv::Mat &dstMat)
{

	vector<vector<Point>> contours;
	cv::Mat selectMat;

	cv::Mat m = cv::Mat::ones(srcMat.size(), CV_32F);

	if (!srcMat.empty()) {
		srcMat.copyTo(selectMat);
		srcMat.copyTo(dstMat);
	}
	else {
		std::cout << "failed to read image!:" << std::endl;
		return -1;
	}

	namedWindow("mouseCallback");
	imshow("mouseCallback", selectMat);
	setMouseCallback("mouseCallback", on_mouse, &selectMat);
	waitKey(0);
	destroyAllWindows();//�����ͼ����
						//����roi
	contours.push_back(mousePoints);
	if (contours[0].size() < 3) {
		std::cout << "failed to read image!:" << std::endl;
		return -1;
	}

	drawContours(m, contours, 0, Scalar(0), -1);

	m.copyTo(dstMat);

	return 0;
}


//����һ��ͼƬ������丵��Ҷ�任��Ŀ��ӻ��ķ�ֵ��
//ͬʱ�����λ�ף��ͻ�ԭ��һ��ʱ��ϵ���������ֵ
int calcVisbalDft(cv::Mat srcMat, cv::Mat & magMat, cv::Mat & ph, double & normVal)
{
	int m = getOptimalDFTSize(srcMat.rows); //2,3,5�ı����и���Ч�ʵĸ���Ҷ�任
	int n = getOptimalDFTSize(srcMat.cols);
	Mat padded;
	//�ѻҶ�ͼ��������Ͻ�,���ұߺ��±���չͼ��,��չ�������Ϊ0;
	copyMakeBorder(srcMat, padded, 0, m - srcMat.rows, 0, n - srcMat.cols, BORDER_CONSTANT, Scalar::all(0));
	//planes[0]Ϊdft�任��ʵ����planes[1]Ϊ�鲿��phΪ��λ�� plane_true=magΪ��ֵ
	Mat planes[] = { Mat_<float>(padded), Mat::zeros(padded.size(), CV_32F) };
	Mat planes_true = Mat_<float>(padded);

	//������λ��Mat_����ȷ�����������ͣ�����Ԫ��ʱ����Ҫ��ָ��Ԫ�����ͣ�
	ph = Mat_<float>(padded);

	Mat complexImg;
	//��ͨ��complexImg����ʵ�������鲿

	merge(planes, 2, complexImg);
	//���ϱߺϳɵ�mat���и���Ҷ�任,***֧��ԭ�ز���***,����Ҷ�任���Ϊ����.ͨ��1�����ʵ��,ͨ����������鲿
	dft(complexImg, complexImg);
	//�ѱ任��Ľ���ָ����mat,һ��ʵ��,һ���鲿,�����������
	split(complexImg, planes);

	//---------------�˲���Ŀ��Ϊ���õ���ʾ��ֵ---�����ָ�ԭͼʱ�����ٴ���һ��-------------------------
	magnitude(planes[0], planes[1], planes_true);//������mag
	phase(planes[0], planes[1], ph);//��λ��ph
	Mat A = planes[0];
	Mat B = planes[1];
	magMat = planes_true;

	magMat += Scalar::all(1);//�Է�ֵ��1
							 //������ķ�ֵһ��ܴ󣬴ﵽ10^4,ͨ��û�а취��ͼ������ʾ��������Ҫ�������log��⡣
	log(magMat, magMat);

	//ȡ�����е����ֵ�����ں�����ԭʱȥ��һ��
	minMaxLoc(magMat, 0, &normVal, 0, 0);

	//�޼�Ƶ��,���ͼ����л������������Ļ�,����Ƶ���ǲ��ԳƵ�,���Ҫ�޼�
	magMat = magMat(Rect(0, 0, magMat.cols & -2, magMat.rows & -2));
	ph = ph(Rect(0, 0, magMat.cols & -2, magMat.rows & -2));
	Mat _magI = magMat.clone();
	//�����ȹ�һ��������ʾ��Χ��
	normalize(_magI, _magI, 0, 1, NORM_MINMAX);
	//imshow("before rearrange", _magI);

	//��ʾ����Ƶ��ͼ
	int cx = magMat.cols / 2;
	int cy = magMat.rows / 2;

	//������������Ϊ��׼����magͼ��ֳ��Ĳ���
	Mat tmp;
	Mat q0(magMat, Rect(0, 0, cx, cy));
	Mat q1(magMat, Rect(cx, 0, cx, cy));
	Mat q2(magMat, Rect(0, cy, cx, cy));
	Mat q3(magMat, Rect(cx, cy, cx, cy));
	q0.copyTo(tmp);
	q3.copyTo(q0);
	tmp.copyTo(q3);
	q1.copyTo(tmp);
	q2.copyTo(q1);
	tmp.copyTo(q2);
	//��һ��
	normalize(magMat, magMat, 0, 1, NORM_MINMAX);
	imwrite("ԭƵ��.png", magMat*255);
	return 0;
}

/*IDFT��任*/
int calcDft2Image(Mat srcMat, cv::Mat magMat, cv::Mat ph, double normVal, cv::Mat & dstMat)
{
	Mat dst;
	Mat mag = magMat.clone();

	int m = getOptimalDFTSize(srcMat.rows); //2,3,5�ı����и���Ч�ʵĸ���Ҷ�任
	int n = getOptimalDFTSize(srcMat.cols);
	Mat padded;
	//�ѻҶ�ͼ��������Ͻ�,���ұߺ��±���չͼ��,��չ�������Ϊ0;
	copyMakeBorder(srcMat, padded, 0, m - srcMat.rows, 0, n - srcMat.cols, BORDER_CONSTANT, Scalar::all(0));
	//planes[0]Ϊdft�任��ʵ����planes[1]Ϊ�鲿��phΪ��λ�� plane_true=magΪ��ֵ
	Mat planes[] = { Mat_<float>(padded), Mat::zeros(padded.size(), CV_32F) };
	Mat planes_true = Mat_<float>(padded);

	Mat complexImg;
	//��ͨ��complexImg����ʵ�������鲿

	int cx = mag.cols / 2;
	int cy = mag.rows / 2;
	//ǰ�����跴����һ�飬Ŀ����Ϊ����任��ԭͼ
	Mat q00(mag, Rect(0, 0, cx, cy));
	Mat q10(mag, Rect(cx, 0, cx, cy));
	Mat q20(mag, Rect(0, cy, cx, cy));
	Mat q30(mag, Rect(cx, cy, cx, cy));

	Mat tmp;
	//��������
	q00.copyTo(tmp);
	q30.copyTo(q00);
	tmp.copyTo(q30);
	q10.copyTo(tmp);
	q20.copyTo(q10);
	tmp.copyTo(q20);

	mag = mag * normVal;//����һ���ľ���ԭ 
	exp(mag, mag);		//��Ӧ��ǰ��ȥ����
	mag = mag - Scalar::all(1);//��Ӧǰ��+1
	polarToCart(mag, ph, planes[0], planes[1]);//�ɷ�����mag����λ��ph�ָ�ʵ��planes[0]���鲿planes[1]
	merge(planes, 2, complexImg);//��ʵ���鲿�ϲ�


								 //-----------------------����Ҷ����任-----------------------------------
	Mat ifft(Size(mag.cols, mag.rows), CV_8UC1);
	//����Ҷ��任
	idft(complexImg, ifft, DFT_REAL_OUTPUT);
	normalize(ifft, ifft, 0, 1, NORM_MINMAX);

	Rect rect(0, 0, mag.cols, mag.rows); //������Ҷ�任���������ͼ������СΪԭ��ͼ��Ĵ�С
	dst = ifft(rect);
	dst = dst * 255;

	dst.convertTo(dstMat, CV_8UC1);

	return 0;
}
