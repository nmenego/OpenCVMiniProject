#include <vector>
#include <opencv2/core/core.hpp>   // basic OpenCV structures
#include <opencv2/highgui/highgui.hpp>  // OpenCV window I/O
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/video/video.hpp>
#include <stdio.h>

using namespace cv;

bool isWithinDrawBox(int x, int y) {

	if (x >= 255 && x < 605 && y >= 15 && y < 475) {
		return true;
	} else {
		return false;
	}
}

void setColor(Scalar &sca, int x, int y) {

	if (x >= 25 && x < 190) {
		if (y >= 20 && y < 108) {
			sca = Scalar(0, 0, 255);
		} else if (y >= 120 && y < 208) {
			sca = Scalar(0, 255, 0);
		} else if (y >= 220 && y < 308) {
			sca = Scalar(255, 0, 0);
		} else if (y >= 320 && y < 408) {
			sca = Scalar(255, 255, 255);
		} else {
			// do nothing
		}
	}
}

void drawBackground(Mat &matrix) {
	rectangle(matrix, Point(6, 10), Point(206, 470), Scalar(0, 0, 0), 5, 8, 0);
	rectangle(matrix, Point(250, 10), Point(600, 470), Scalar(0, 0, 0), 5, 8,
			0);
	// colors
	rectangle(matrix, Point(25, 20), Point(190, 108), Scalar(0, 0, 255),
	CV_FILLED, 8, 0);
	rectangle(matrix, Point(25, 120), Point(190, 208), Scalar(0, 255, 0),
	CV_FILLED, 8, 0);
	rectangle(matrix, Point(25, 220), Point(190, 308), Scalar(255, 0, 0),
	CV_FILLED, 8, 0);
	rectangle(matrix, Point(25, 320), Point(190, 408), Scalar(0, 0, 0),
	CV_FILLED, 8, 0);
}

int main(int argc, char** argv) {

	bool draw_mode = false;
	// open the default camera
	VideoCapture cap(0);
	// check if we succeeded
	if (!cap.isOpened()) {
		return -1;
	}

	string orig_vid = "Original Video";
	string paint = "Paint";
	namedWindow(orig_vid, CV_WINDOW_AUTOSIZE);

	Mat frame, subFrame, imageHSV;
	int frameCount;
	cap >> frame; // get a new frame from camera
	int width = frame.cols;
	int height = frame.rows;
	printf("Frame Dimensions (w x h):  %d x %d\n", width, height);
	int xCenter = width / 2;
	int yCenter = height / 2;
	int widthObject = width / 16;
	int halfObject = widthObject / 2;
	int x, y;
	float sum, sumX, sumY;
	float histValue;
	int hueBin, satBin;

	// These declarations are used in computing the color histogram
	int hueBins = 60;  // quantize hue (0..179) to levels
	int hueBinSlice = 180 / hueBins;  // size of a hue slice
	int satBins = 32;  // quantize saturation (black&white 0..255) to levels
	int satBinSlice = 256 / satBins;  // size of a sat slice
	int histSize[] = { hueBins, satBins };
	float hueRanges[] = { 0, 180 };
	float satRanges[] = { 0, 256 };
	const float * ranges[] = { hueRanges, satRanges };
	MatND hist;
	int channels[] = { 0, 1 }; // compute the histogram from 0-th and 1st channels
	bool uniform = true;  // histogram is uniform
	bool accumulate = false;

	// wait for a few seconds until the user settles the object inside the green box
	for (frameCount = 0; frameCount < 80; frameCount++) {
		cap >> frame; // get a new frame from the camera
		flip(frame, frame, 1);
		drawBackground(frame);
		rectangle(frame,
				Rect(xCenter - halfObject, yCenter - halfObject, widthObject,
						widthObject), Scalar(0, 255, 0), 3, 8);
		imshow(orig_vid, frame);
		if (waitKey(20) >= 0)
			break;
	}

	// get the object in the green box and learn its color
	cap >> frame; // get a new frame from the camera
	flip(frame, frame, 1);
	subFrame = frame(
			Rect(xCenter - halfObject, yCenter - halfObject, widthObject,
					widthObject));
	cvtColor(subFrame, imageHSV, CV_BGR2HSV);

	// Calculate the object's color histogram
	// http://docs.opencv.org/2.4/doc/tutorials/imgproc/histograms/histogram_calculation/histogram_calculation.html
	// http://docs.opencv.org/2.4/modules/imgproc/doc/histograms.html?highlight=calchist#calchist
	calcHist(&imageHSV, 1, channels, Mat(), // no mask for selecting pixels
			hist, 2, histSize, ranges, uniform, accumulate);

	// create a matrix to store user's drawn image
	Mat drawing, background;
	background.create(frame.size(), frame.type());
	drawing.create(frame.size(), frame.type());
	drawing = Scalar(255, 255, 255);
	background = Scalar(255, 255, 255);
	Scalar selected_color = Scalar(255, 255, 255);

	// follow the object based on its color composition
	for (frameCount = 0;; frameCount++) {
		cap >> frame; // get a new frame from the camera
		flip(frame, frame, 1);  // flip the image horizontally
		cvtColor(frame, imageHSV, CV_BGR2HSV); // convert to Hue-Saturation-Value color space
		drawBackground(frame);

		sum = 0; // these sums are used for computing the x and y center of mass
		sumX = 0;
		sumY = 0;
		for (y = 0; y < height; y++) {
			for (x = 0; x < width; x++) {
				if (y >= 0 && y < height && x >= 0 && x < width) {
					hueBin = imageHSV.at<Vec3b>(y, x)[0] / hueBinSlice;
					satBin = imageHSV.at<Vec3b>(y, x)[1] / satBinSlice;
					histValue = hist.at<float>(hueBin, satBin); // fetch the pixel's frequency from the histogram
					sum += histValue; // use the pixel's frequency as a weight for the center of mass
					sumX += x * histValue;
					sumY += y * histValue;
				}
			}
		}

		printf("\n");
		if (sum > 0.001) {
			xCenter = (int) (sumX / sum);   // compute the center of mass
			yCenter = (int) (sumY / sum);
		}
		if (frameCount % 10 == 0) {
			printf("xCenter,yCenter = %d %d\n", xCenter, yCenter);
		}

		rectangle(frame,
				Rect(xCenter - halfObject, yCenter - halfObject, widthObject,
						widthObject), Scalar(0, 0, 255), 2, 8);

		draw_mode = isWithinDrawBox(xCenter, yCenter);
		// put a red box around the object
		if (draw_mode) {
			// TODO draw using tool selected.
			circle(drawing, Point(xCenter, yCenter), 5, selected_color, 10, 4,
					0);

		} else {
			// stop drawing...
			setColor(selected_color, xCenter, yCenter);
		}
		addWeighted(frame, 0.5, drawing, 0.5, 0.0, frame);
		imshow(orig_vid, frame);
		//imshow(paint, drawing);

		if (waitKey(20) >= 0)
			break;
	}

	return 0;
}

