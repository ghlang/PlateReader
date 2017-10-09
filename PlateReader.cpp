/*
 */
#include "opencv2/highgui/highgui.hpp"
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <alpr.h>
#include <iostream>
#include <fstream>
#include <istream>
#include <sstream>
#include <locale>
#include <iomanip>
#include <vector>
#include <map>
#include <chrono>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <ctime>
#include <stdio.h>
#include <direct.h>
#include "MediaInfoDLL/MediaInfoDLL.h" //Dynamicly-loaded library (.dll or .so)
#include "Lane.h"
#include "Plate.h"

#define MediaInfoNameSpace MediaInfoDLL;

using namespace cv;
using namespace std;
using namespace alpr;
using namespace MediaInfoNameSpace;

float mthreshold = 30; // pixel may differ only up to "threshold" to count as being "similar"




#define _debugwin

class PlateRecognizer
{
public:
	enum WindowType { WTNone = 0, WTColor, WTMask };
	PlateRecognizer()
	{
		regionsOfInterest.clear();
	}
	PlateRecognizer(string folder, int i)
	{
		id = i;
		mAlpr = new Alpr("eu", "d:\\openalpr_2.2.0\\openalpr.conf");
		plateFolder = folder;
		if (!folder.empty()) {
			mkdir(folder.c_str());
			txtfile.open(plateFolder + "Plates.txt");
			txtfile1.open(plateFolder + "Dir1.txt");
			txtfile2.open(plateFolder + "Dir2.txt");
			checkForTrucks = false;
		}
		else
		{
			checkForTrucks = true;
		}
	
#ifdef _debugwin
		namedWindow(plateFolder, CV_WINDOW_KEEPRATIO); //resizable window;
		resizeWindow(plateFolder, 100, 100);
#endif
	}
	~PlateRecognizer() {
	}
	int AnalzyeFrame(Mat src, __time64_t time, bool refframe);
	void CheckForInactivePlates();
	Rect GetROI() { return lane.GetROI(); }
	void drawLane(Mat img) {
		lane.draw(img, checkForTrucks);
	}
	int CheckForLaneHit(Point mouse)
	{
		return lane.CheckForHit(mouse);
	}
	void MoveSelectedLanePoint(Point mouse)
	{
		return lane.MoveSelectedPoint(mouse);
	}
	bool CheckForTrucks() { return checkForTrucks; };
	bool IsTruckInLane() { return truckInLane; };
	void RecalcLane() { lane.Recalc(); }
	Lane *GetLane() { return &lane; }
	void SetWindowType(WindowType type) {
		wtDisplay = type;
	};
	void NextWindowType()
	{
		switch (wtDisplay)
		{
		case WTNone:
			wtDisplay = WTColor;
			break;
		case WTColor:
			wtDisplay = WTMask;
			break;
		case WTMask:
		default:
			wtDisplay = WTNone;
			break;
		}
	};


private:
	Mat frame;
	Mat lastFrame;
	Lane lane;
	int id;
	Alpr* mAlpr;
	std::vector < alpr::AlprRegionOfInterest > regionsOfInterest;
	string plateFolder;
	ofstream txtfile;
	ofstream txtfile1;
	ofstream txtfile2;

	bool checkForTrucks = false;
	bool truckInLane = false;
	int processFrames = 0;
	int motionFrames = 0;
	int noMotionFrames = 0;
	int lastVehicleX = 0;
	int motionVehicle = 0;
	std::vector <Plate*> vecPlates;
	Mat plateFrame;
	Rect roi;
	bool reco = false;
	WindowType wtDisplay = WTNone;

};






void PlateRecognizer::CheckForInactivePlates()
{
	auto it = std::begin(vecPlates);
	while (it != std::end(vecPlates)) {
		// skip if no plate was found

		if (!(*it)->IsPlateActive())
		{
			if ((*it)->Write(txtfile, txtfile1, txtfile2,id))
			{
				motionFrames = 0;
				noMotionFrames = 0;
			}
			delete (*it);
			vecPlates.erase(it);
		}
		else
			++it;
	}
}

int PlateRecognizer::AnalzyeFrame(Mat src, __time64_t time, bool refframe)
{
	roi = lane.GetROI();
	if (frame.cols != roi.width || frame.rows != roi.height)
	{
		resize(src, frame, Size(roi.width, roi.height));
		resize(src, lastFrame, Size(roi.width, roi.height));
		resizeWindow(plateFolder, roi.width/2, roi.height/2);
	}
	src(roi).copyTo(frame);
	double motion = 0.;
	Mat diff;
	Mat diff1Channel;
	Mat mask;
	int leftcol = 0;
	int rightcol = 0;
	lane.MaskOut(frame);


#if 1
	processFrames--;
	if (processFrames > 0)
		reco = true;
	else
		reco = false;

	if (!reco || true)
	{
		absdiff(frame, lastFrame, diff);

#if 0
		Mat yuvFrame;
		Mat yuvLastFrame;
		cvtColor(frame, yuvFrame, CV_BGR2YCrCb);
		cvtColor(lastFrame, yuvLastFrame, CV_BGR2YCrCb);
		uchar* pFrame = frame.data;
		uchar* pyuvFrame = yuvFrame.data;
		uchar* pyuvLastFrame = yuvLastFrame.data;
		uchar* pdiff = diff.data;

		for (int i = 0; i < frame.cols*frame.rows; i++)
		{
/*			pyuvFrame++;
			pyuvLastFrame++;
			float uvdiff = fabs((int)*pyuvFrame - (int)*pyuvLastFrame);
			pyuvFrame++;
			pyuvLastFrame++;
			uvdiff += fabs((int)*pyuvFrame - (int)*pyuvLastFrame);
			pyuvFrame++;
			pyuvLastFrame++;
			for (int j = 0; j < 3; j++)
			{
				if (uvdiff < mthreshold)
					*pdiff = 0;
				else
					*pdiff = *pFrame;
				pdiff++;
				pFrame++;
			}*/

			*pdiff = *pyuvFrame;
			pdiff++;
			*pdiff = *pyuvFrame;
			pdiff++;
			*pdiff = *pyuvFrame;
			pdiff++;
			pyuvFrame++;
			pyuvFrame++;
			pyuvFrame++;

		}
#endif


		// WARNING: this will weight channels differently! - instead you might want some different metric here. e.g. (R+B+G)/3 or MAX(R,G,B)
		cvtColor(diff, diff1Channel, CV_BGR2GRAY);

		mask = diff1Channel >  mthreshold;

		bool actitiy = lane.CheckForActivity(mask);

		for (int i = 0; i < 2; i++)
		{
			if (lane.IsVehiclePresent(i))
			{
				std::cout << "Lane " << id+1 << put_time(localtime(&time), " %H:%M:%S") << " Dir " << i + 1 << " -----" << endl; 
				txtfile << put_time(localtime(&time), "%H:%M:%S") << ";-----" << endl;  //put_time(localtime(&time), "%c");
				if (i == 0)
					txtfile1 << put_time(localtime(&time), "%H:%M:%S") << ";-----" << endl;  //put_time(localtime(&time), "%c");
				else
					txtfile2 << put_time(localtime(&time), "%H:%M:%S") << ";-----" << endl;  //put_time(localtime(&time), "%c");
			}
		}

		lane.drawDriveDirections(mask);

#if 0
		uchar* data = mask.data;
		unsigned int *colhits = new unsigned int[mask.cols];
		int hits = 0;
		int maxhits = 0;
		int maxhitscol = 0;
//		for (int i = 0; i < mask.rows && !reco; i++)
		for (int i = 0; i < mask.cols; i++)
			colhits[i] = 0;
		for (int i = 0; i < mask.rows; i++)
		{
			for (int j = 0; j < mask.cols; j++)
			{
				if ((int)*data > 0)
				{
					colhits[j] ++;
					if (colhits[j] > maxhits)
					{
						maxhits = colhits[j];
						maxhitscol = j;
					}
					hits++;
				}
				data++;
			}
		}
		leftcol = maxhitscol;
		rightcol = maxhitscol;
		for (int i = maxhitscol; i > 0; i--)
		{
			if (colhits[i] < 5)
				break;
			leftcol = i;
		}
		for (int i = maxhitscol; i < mask.cols; i++)
		{
			if (colhits[i] < 5)
				break;
			rightcol = i;
		}

		if ((double)hits / (double)(mask.cols*mask.rows) > 0.01)
		{
			reco = true;
			processFrames = 50;
			motionFrames++;
			noMotionFrames = 0;
			int VehicleX = (leftcol + rightcol) / 2.;
			//if (lastVehicleX > 0) {
			//	motionVehicle += (lastVehicleX - VehicleX);
			//	std::cout << lastVehicleX - VehicleX << endl;
			//}
			lastVehicleX = VehicleX;

		}
		else
		{
			/*			noMotionFrames++;
			if (noMotionFrames == 20)
			{
				if (motionFrames > 40)
				{
					txtfile << put_time(localtime(&time), "%H:%M:%S") << ";-----" << endl;  //put_time(localtime(&time), "%c");
					if (motionVehicle < 0)
						std::cout << put_time(localtime(&time), "%H:%M:%S") << "Dir 1;-----" << endl;  //put_time(localtime(&time), "%c");
					else
						std::cout << put_time(localtime(&time), "%H:%M:%S") << "Dir 2;-----" << endl;  //put_time(localtime(&time), "%c");
					motionFrames++;
				}
				motionFrames = 0;
				lastVehicleX = 0;
				motionVehicle = 0;
			}*/
		}
#endif
		frame.copyTo(lastFrame);
	}
#ifdef _debugwin
	if (wtDisplay == WTMask)
		imshow(plateFolder, mask);
#endif
	reco = true;
#else
	reco = true;
#endif
#if 0
	if (!reco)
	{
		for (auto it = std::begin(vecPlates); it < std::end(vecPlates); it++)
		{
			(*it)->SkipFrame();
		}
		CheckForInactivePlates();

		return 0;
	}
#endif

	regionsOfInterest.clear();
	AlprResults results = mAlpr->recognize(frame.data, 3, frame.cols, frame.rows, regionsOfInterest);

	for (int i = 0; i < results.plates.size(); i++)
	{
		std::stringstream textstream;
		int fontFace = FONT_HERSHEY_SIMPLEX;
		double fontScale = 1;
		int thickness = 2;

		alpr::AlprPlateResult plate = results.plates[i];
		bool found = false;
		if (plate.topNPlates.size() < 1)
			continue;
		alpr::AlprPlate candidate = plate.topNPlates[0];
		int count = 0;
		for (int i = 0; i < candidate.characters.size(); i++)
			if (candidate.characters[i] == 'I' || candidate.characters[i] == '1') count++;
		if (count > 2)
			continue;
		textstream << candidate.characters;

#if 0
		{
			cout << "Plate " << i;
			alpr::AlprPlateResult plate = results.plates[i];
			for (int k = 0; k < plate.topNPlates.size() && k < 5; k++)
			{
				alpr::AlprPlate candidate = plate.topNPlates[k];
				cout << " " << candidate.characters << " " << candidate.overall_confidence;
				textstream << candidate.characters;
			}
			cout << endl;
		}
#endif


		int left = plate.plate_points[0].x;
		int right = plate.plate_points[0].x;
		int top = plate.plate_points[0].y;
		int bottom = plate.plate_points[0].y;


		for (int p = 1; p < 4; p++)
		{
			left = min(left, plate.plate_points[p].x);
			right = max(right, plate.plate_points[p].x);
			top = max(top, plate.plate_points[p].y);
			bottom = min(bottom, plate.plate_points[p].y);
		}
		Point center((left + right) / 2, (top + bottom) / 2);
		Rect plrect(left, bottom, right - left, top - bottom);
		rectangle(frame, plrect, Scalar(0xff, 0x00, 0x00), 2);
		putText(frame, textstream.str(), Point(left, top + 25), fontFace, fontScale,
			Scalar::all(255), thickness, 8);


		auto itfound = std::begin(vecPlates);
		for (auto it = std::begin(vecPlates); it < std::end(vecPlates); it++)
		{
			if ((*it)->CheckForPlate(center, plate))
			{
				found = true;
				itfound = it;
			}
		}
		int ins = 0;
		if (!found)
		{
			Plate *newPlate = new Plate(center, roi, time, plateFolder, lane.GetDirection(0));
			ins = newPlate->InsertCandidates(plate, frame);
			vecPlates.push_back(newPlate);
		}
		else
		{
			ins = (*itfound)->InsertCandidates(plate, frame);
		}
		motionFrames -= ins;
		if (motionFrames < 0)
			motionFrames = 0;

	}

	if (results.plates.size() > 0)
	{
		reco = true;
		processFrames = 50;
	}
	else
	{
		for (auto it = std::begin(vecPlates); it < std::end(vecPlates); it++)
		{
			(*it)->SkipFrame();
		}
	}

	CheckForInactivePlates();

#ifdef _debugwin
	if (wtDisplay == WTColor)
		imshow(plateFolder, frame);
#endif

	return results.plates.size();
}

int roiIndex = 1;
std::vector < PlateRecognizer*> plateRecognizer;


void onMouse(int event, int x, int y, int flags, void* param) {
	static bool mousedown = false;
	switch (event)
	{
	case CV_EVENT_LBUTTONDOWN:
		mousedown = false;
		roiIndex = 1;
		for (PlateRecognizer* recognizer : plateRecognizer)
		{
			if (recognizer->CheckForLaneHit(Point(x, y)))
			{
				mousedown = true;
				return;
			}
			roiIndex++;
		}
		if (!mousedown)
			roiIndex = 0;
		break;
	case CV_EVENT_MOUSEMOVE:
		if (mousedown && roiIndex > 0) {
			plateRecognizer[roiIndex - 1]->MoveSelectedLanePoint(Point(x, y));
		}
		break;
	case CV_EVENT_LBUTTONUP:
		if (mousedown && roiIndex > 0) {
			plateRecognizer[roiIndex - 1]->MoveSelectedLanePoint(Point(x, y));
			plateRecognizer[roiIndex - 1]->RecalcLane();
		}
		mousedown = false;
		break;
	}
}



int main(int ac, char** av)
{

	if (ac < 2)
		return 0;
  //  if (ac != 2)
  //  {
		//cout << "\nThis program looks for *.mp4 in the specified folder\n"
		//	"Usage:\n./" << av[0] << " <folder>\n" << "q,Q,esc -- quit\n"
		//	<< "-r start the recoginition immediately \n\n"
		//	<< "\tThis is a starter sample, to get you up and going in a copy pasta fashion\n"
		//	<< "\tThe program captures frames from a camera connected to your computer.\n"
		//	<< "\tTo find the video device number, try ls /dev/video* \n"
		//	<< "\tYou may also pass a video file, like my_vide.avi instead of a device number"
		//	<< "\n"
		//	<< "DATA:\n"
		//	<< "Generate a datamatrix from  from http://datamatrix.kaywa.com/  \n"
		//	<< "  NOTE: This only handles strings of len 3 or less\n"
		//	<< "  Resize the screen to be large enough for your camera to see, and it should find an read it.\n\n"
		//	<< endl;

		//return 1;
  //  }

	Alpr openalpr("eu", "d:\\openalpr_2.2.0\\openalpr.conf");

	// Optionally specify the top N possible plates to return (with confidences).  Default is 10
	openalpr.setTopN(3);

	// Make sure the library loaded before continuing.
	// For example, it could fail if the config/runtime_data is not found
	if (openalpr.isLoaded() == false)
	{
		std::cerr << "Error loading OpenALPR" << std::endl;
		return 0;
	}


	std::chrono::system_clock::time_point EncDate;
	ofstream outfile;
	bool play = true;
	int speed = 1;
	bool reco = false;
	bool slomo = false;
	float contrast = 1.0;
	bool headless = false;
	if (ac == 3) reco = true;
	if (ac == 3) headless = true;



	Mat newFrame;
	Mat frame;
	//Mat diff;
	//Mat diff1Channel;
	Mat mask;

	string window_name = "video";
	__time64_t frameTime;

//	string szPath = "D:\\KZVerfolgung\\Ibk_Ost_vonBregenz\\";
	string szPath = av[1];
	szPath += "\\";
	//outfile.open(szPath +"Plates.txt");




	WIN32_FIND_DATA FindFileData;
	string szFile = szPath + "*.mp4";
	HANDLE hFind = FindFirstFile(szFile.c_str(), &FindFileData);
	if (hFind == INVALID_HANDLE_VALUE) {
		return 0;
	}


	ifstream conf;
	conf.open(szPath + "Lanes.txt");
	int nPR;
	if (conf.is_open())
	{
		conf >> nPR;
		for (int i = 0; i < nPR; i++)
		{
			plateRecognizer.push_back(new PlateRecognizer(szPath + "Lane" + to_string(i+1) + "\\", i));
			plateRecognizer[i]->GetLane()->deserialize(conf);
		}
		conf >> contrast;
		if (!conf.eof())
			conf >> mthreshold;
		conf.close();
	}
	else
	{
		plateRecognizer.push_back(new PlateRecognizer(szPath + "Lane1\\",0)); //&direction1);
	}


//	regionsOfInterest.clear();
	roiIndex = 1;


	bool create_window = true;
	int showImage = 0;
	bool refframe = false;

	do {
		string szFoundFile = szPath + FindFileData.cFileName;
//		szFoundFile = "D:\\KZVerfolgung\\Ibk\\.mp4";
//		szFoundFile = "D:\\BVR\\StJohann\\Vormittag\\V4\\GP050429.mp4";
		cout << szFoundFile << endl;
		MediaInfo MI;
		MI.Open(__T(szFoundFile));
		MI.Option(__T("Inform"), __T("General; %Encoded_Date%"));
		auto inform = MI.Inform();
		std::tm t = {};
		std::istringstream ss(inform);
		ss >> std::get_time(&t, "UTC %Y-%m-%d %H:%M:%S");
		// correct for wrong daylight saving information.
		int isdst = t.tm_isdst;
		std::mktime(&t);
		t.tm_hour += isdst - t.tm_isdst;
		std::cout << std::put_time(&t, "%c") << '\n';
		// change to chrono for ease of use
		EncDate = std::chrono::system_clock::from_time_t(std::mktime(&t));
		MI.Close();


		VideoCapture capture(szFoundFile); //try to open string, this will attempt to open it as a video file
		if (!capture.isOpened())
		{
			cerr << "Failed to open a video file!\n" << endl;
			return 1;
		}
		double fps = capture.get(CV_CAP_PROP_FPS);
//		capture.set(CV_CAP_PROP_FPS, fps*2.0);

		int width = (int)capture.get(CV_CAP_PROP_FRAME_WIDTH);
		int height = (int)capture.get(CV_CAP_PROP_FRAME_HEIGHT);
		int frame_count = (int)capture.get(CV_CAP_PROP_FRAME_COUNT);
		int capturedFrames = 0;
		if (create_window && !headless)
		{
			namedWindow(window_name, CV_WINDOW_KEEPRATIO); //resizable window;
			resizeWindow(window_name, width / 2, height / 2);
			setMouseCallback(window_name, onMouse, nullptr);
			create_window = false;
			//namedWindow("mask", CV_WINDOW_KEEPRATIO); //resizable window;
			//resizeWindow("mask", width / 2, height / 2);
		}

		capture >> frame;

		int imgFound = 0;
		bool captured = true;
		auto start = std::chrono::high_resolution_clock::now();
		for (;;)
		{
			if (play)
			{
				int tcframe = (int)capture.get(CAP_PROP_POS_FRAMES);
				capture >> newFrame;
				int tnframe = (int)capture.get(CAP_PROP_POS_FRAMES);
				int addframes = 0;
				while (tnframe == tcframe && tcframe + addframes < frame_count)
				{
					addframes++;
					if (addframes > 200)
						addframes = 60 * 60 * 30; // 60 min * 60 sec * 30 frames
//					capture.set(CAP_PROP_POS_FRAMES, tcframe + 1);
					capture >> newFrame;
					tnframe = (int)capture.get(CAP_PROP_POS_FRAMES);
				}

				captured = true;
				//for (int c = 0; c < speed - 1 && !imgFound && !reco; c++)
				//{
				//	capture >> newFrame;
				//}
				auto CurrTime = EncDate + std::chrono::milliseconds((long)capture.get(CV_CAP_PROP_POS_MSEC));
				frameTime = std::chrono::system_clock::to_time_t(CurrTime);
				if ((int)capture.get(CV_CAP_PROP_POS_FRAMES) + addframes > frame_count - 5)
					break;
				if (slomo)
					Sleep(200);
			}
			if (!newFrame.empty())
			{
				//Mat diff;
				//Mat diff1Channel;
				//absdiff(frame, newFrame, diff);
				//	// WARNING: this will weight channels differently! - instead you might want some different metric here. e.g. (R+B+G)/3 or MAX(R,G,B)
				//cvtColor(diff, diff1Channel, CV_BGR2GRAY);

				//float threshold = 30; // pixel may differ only up to "threshold" to count as being "similar"

				//mask = diff1Channel > threshold;

				//imshow("mask", mask);
//				newFrame.copyTo(frame);
				newFrame.convertTo(frame, -1, contrast, 0);
				//				frame = frame + Scalar::all(100);
			}

			if (!frame.empty() && captured)
			{
				captured = false;
				if (reco)
				{
					imgFound = 0;
					for (auto it = std::begin(plateRecognizer); it < std::end(plateRecognizer); it++)
						//auto it = std::begin(plateRecognizer);
					{
						imgFound += (*it)->AnalzyeFrame(frame, frameTime, refframe);
					}
				}
				else
				{
					imgFound = 0;
				}

			}
			showImage++;
			//if (true) // imgFound || showImage > 5)
			if (showImage >= speed && !headless)
			{
				// now the result is in `buffer.str()`.				string text;
				std::stringstream textstream;
				int fontFace = FONT_HERSHEY_SIMPLEX;
				double fontScale = 1;
				int thickness = 2;

				int i = 1;
				for (PlateRecognizer* recognizer : plateRecognizer)
				{
					Rect roi = recognizer->GetROI();
					recognizer->drawLane(frame);
					//					rectangle(frame, roi, Scalar(0xff, 0xff, 0xff), 2);
					textstream.str("");
					textstream.clear();
					textstream << i;
					putText(frame, textstream.str(), Point(roi.x + 10, roi.y + 25), fontFace, fontScale,
						Scalar::all(255), thickness, 8);
					i++;
				}

				textstream.str("");
				textstream.clear();
				textstream << put_time(localtime(&frameTime), "%H:%M:%S speed ") << speed << " contrast " << contrast;
				if (reco)
					textstream << " R";
				// then put the text itself
				putText(frame, textstream.str(), Point(10, 25), fontFace, fontScale,
					Scalar(0, 0, 200), thickness, 8);

				//int pw = plateFrame.cols;
				//int ph = plateFrame.rows;
				//plateFrame.copyTo(lastFrame.rowRange(50, 50 + ph).colRange(0, pw));
				imshow(window_name, frame);
				if (showImage >= speed)
				{
					showImage = 0;
				}

				int iKey = waitKey(1);
				char key = (char)iKey; //delay N millis, usually long enough to display and capture input
				if (iKey > -1 && iKey < 255) {

					switch (key)
					{
					case 'q':
					case 'Q':
					case 27: //escape key
						return 0;
					case ' ': //Save an image
							  //sprintf(filename, "filename%.3d.jpg", n++);
							  //imwrite(filename, frame);
							  //cout << "Saved " << filename << endl;
						play = !play;
						speed = 1;
						break;
					case 'c':
						newFrame.copyTo(frame);
						break;
					case 'd':
						slomo = !slomo;
						break;
					case '+':
						if (speed < 20)
							speed++;
						break;
					case '-':
						if (speed > 1)
							speed--;
						break;
					case 'r':
						capture.set(CAP_PROP_POS_FRAMES, 0);
						break;
					case 'e':
					{
						for (auto it = std::begin(plateRecognizer); it < std::end(plateRecognizer); it++)
							//auto it = std::begin(plateRecognizer);
						{
							imgFound += (*it)->AnalzyeFrame(frame, frameTime, refframe);
						}

						//std::vector < alpr::AlprRegionOfInterest > regionsOfInterest;
						//regionsOfInterest.clear();
						//AlprResults results = openalpr.recognize(frame.data, 3, frame.cols, frame.rows,  regionsOfInterest);

						//for (int i = 0; i < results.plates.size(); i++)
						//{
						//	cout << "Plate " << i << endl;
						//	alpr::AlprPlateResult plate = results.plates[i];
						//	for (int k = 0; k < plate.topNPlates.size() && k < 3; k++)
						//	{
						//		alpr::AlprPlate candidate = plate.topNPlates[k];
						//		cout << candidate.characters << " " << candidate.overall_confidence << endl;
						//	}
						//}
					}
					break;
					case 'w':
						for (auto it = std::begin(plateRecognizer); it < std::end(plateRecognizer); it++)
						{
							(*it)->NextWindowType();
						}
						break;
					case 'l':
						reco = !reco;
						if (reco) play = true;
						speed = 1;
						break;
					case 'f':
						refframe = !refframe;
						break;
					case '1':
						roiIndex = 1;
						break;
					case '2':
						if (plateRecognizer.size() < 3)
							plateRecognizer.push_back(new PlateRecognizer(szPath + "Lane2\\", 1)); //&direction1);
						roiIndex = 2;
						break;
					case '3':
						if (plateRecognizer.size() < 3)
							plateRecognizer.push_back(new PlateRecognizer(szPath + "Lane3\\", 2)); //&direction1);
						roiIndex = 3;
					case '4':
						if (plateRecognizer.size() < 4)
							plateRecognizer.push_back(new PlateRecognizer(szPath + "Lane4\\", 3)); //&direction1);
						roiIndex = 4;
						break;
					case 's':
					{
						ofstream conf;
						conf.open(szPath + "Lanes.txt");
						conf << plateRecognizer.size() << endl;
						for (PlateRecognizer* recognizer : plateRecognizer)
						{
							recognizer->GetLane()->serialize(conf);
						}
						conf << contrast << endl;
						conf << mthreshold;
						conf.close();
						cv::imwrite(szPath + "Location.jpg", frame);

						break;
					}
					case 't':
						mthreshold += 5;
						break;
					case 'z':
						mthreshold -= 5;
						break;
					default:
						break;
						//case
						//	capture.set(CAP_PROP_POS_FRAMES, CAP);

					}
				}
				if (iKey > 255) {
					switch (iKey)
					{
					case 2555904: // arrow right
						play = false;
						captured = true;
						capture >> newFrame;
						if (!newFrame.empty())
						{
							capturedFrames++;
						}
						break;
					case 2424832: // arrow left
						play = false;
						capturedFrames = (int)capture.get(CAP_PROP_POS_FRAMES);
						capturedFrames -= 5;
						capture.set(CAP_PROP_POS_FRAMES, capturedFrames);
						capture >> newFrame;
						captured = true;
						if (!newFrame.empty())
						{
							capturedFrames -= 5;
						}
						break;
					case 2228224: // page down
						contrast -= 0.01;
						contrast = max(1.0, contrast);
						break;
					case 2162688: // page up
						contrast += 0.01;
						contrast = min(2.0, contrast);
						break;
					}
				}
			}
		}
		auto end = std::chrono::high_resolution_clock::now();
		std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() / 1000 << std::endl;

//		process(capture);
	} while (FindNextFile(hFind, &FindFileData));
	FindClose(hFind);


}
