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
#define MediaInfoNameSpace MediaInfoDLL;

using namespace cv;
using namespace std;
using namespace alpr;
using namespace MediaInfoNameSpace;

float mthreshold = 30; // pixel may differ only up to "threshold" to count as being "similar"


// this class defines a regions by four line centers and line directions. this regions is the search area in which we look for license plates travelling on this lane.
class Lane
{
public:
	Lane() {
		// corner points of the lane
		// initial shape is a square of 500 pixel side length and upper left corner at 100/100
		corners[0] = Point(100., 100.);
		corners[1] = Point(600., 100.);
		corners[2] = Point(600., 600.);
		corners[3] = Point(100., 600.);
		// line direction of the region. first line along the x axis
		direction[0] = Point(100., 0.);
		direction[1] = Point(0., 100.);
		direction[2] = Point(100., 0.);
		direction[3] = Point(0., 100.);
		for (int i = 0; i < 4; i++)
		{
			// center distance is 100 pixel from the corner point in the direction of the line
			dcenter[i] = 100.;
			ldir[i] = sqrt(direction[i].x*direction[i].x + direction[i].y*direction[i].y);
		}
		RecalcCenters();
		maxFrame.height = 1080;
		maxFrame.width  = 1920;
	};
	~Lane() {};
	void serialize(std::ofstream& stream)
	{
		for (int i = 0; i < 4; i++)
			stream << center[i].x << " " << center[i].y  << " " << direction[i].x << " " << direction[i].y << endl;
	};

	void deserialize(std::ifstream& stream)
	{
		for (int i = 0; i < 4; i++)
			stream >> center[i].x >> center[i].y >> direction[i].x >> direction[i].y;
		Recalc();
	};

	void draw(Mat img, bool truck);
	void MaskOut(Mat img);
	int CheckForHit(Point mouse);
	// stndard intersection calculation between two lines defined as vectors
	bool intersection(Point o1, Point d1, Point o2, Point d2,
		Point &r)
	{
		Point x = o2 - o1;
		//Point d1 = p1 - o1;
		//Point d2 = p2 - o2;

		double cross = d1.x*d2.y - d1.y*d2.x;
		if (abs(cross) < /*EPS*/1e-8)
			return false;

		double t1 = (x.x * d2.y - x.y * d2.x) / cross;
		r = o1 + d1 * t1;
		return true;
	};
	// depending on whether we have selected the line center or the line direction the coordinates are treated differently
	void MoveSelectedPoint(Point mouse)
	{
		if (hit > -1 && hit < 4)
		{
			center[hit] = mouse;
			int ic = hit;
			if (hit == 2) ic = 3;
			if (hit == 3) ic = 0;
			Point d = center[hit] - corners[ic];
			dcenter[hit] = sqrt(d.x*d.x + d.y*d.y);
		}
		if (hit > 9 && hit < 14)
		{
			int i = hit - 10;
			direction[i] = mouse - center[i];
			ldir[i] = sqrt(direction[i].x*direction[i].x + direction[i].y*direction[i].y);
		}
		Recalc();
	}
	void Recalc();
	void RecalcCenters();
	Rect GetROI() { return roi; }
private:
	Point corners[4];			// corner points defining the region
	Point center[4];			// a point along a line defining the origin of the line
	double dcenter[4];			// distance from the first corner
	Point direction[4];			// a second point along the line used to define the direction
	double ldir[4];				// distance for the direction point
	Rect roi;					// a rectangular region of interest used for copying
	int hit;
	Size maxFrame;				// the maximum size in pixels the region can have
};

void Lane::Recalc()
{
	// calculate the corner points from the center and direction points
	for (int i = 0; i < 4; i++)
	{
		int i1 = i + 1;
		if (i1 > 3) i1 = 0;
		intersection(center[i], direction[i], center[i1], direction[i1], corners[i1]);
	}
	// recalc the centers again, so the lie within the line
	RecalcCenters();
	// find the min and max values for the rectangle defining the region of interest
	Point bl, tr;
	bl = tr = corners[0];
	for (int i = 1; i < 4; i++)
	{
		bl.x = min(bl.x, corners[i].x);
		bl.y = min(bl.y, corners[i].y);
		tr.x = max(tr.x, corners[i].x);
		tr.y = max(tr.y, corners[i].y);
	}
	bl.x = max(bl.x, 0);
	bl.y = max(bl.y, 0);
	tr.x = min(maxFrame.width - 1, tr.x);
	tr.y = min(maxFrame.height - 1, tr.y);
	roi = Rect(bl, tr);
}

void Lane::RecalcCenters()
{
	for (int i = 0; i < 4; i++)
	{
		int ic = i;
		if (i == 2) ic = 3;
		if (i == 3) ic = 0;
		center[i] = corners[ic] + direction[i] * dcenter[i]/ldir[i];
	}
}

// mask out the areas which are outside the shape defined by the corner points
// this will prevent looking for license plates outside the lane definition
void Lane::MaskOut(Mat img)
{
	Mat mask(img.rows, img.cols, img.type());

	mask = Scalar(0xff, 0xff, 0xff);
	Point msk[4];
	for (int i = 0; i < 4; i++)
		msk[i] = corners[i] - Point(roi.x,roi.y);
	fillConvexPoly(mask, msk, 4, Scalar(0,0,0));

	subtract(img, mask, img);

}

// draw the lane region
void Lane::draw(Mat img, bool truck)
{
	intersection(center[0], direction[0], center[1], direction[1], corners[1]);
	intersection(center[1], direction[1], center[2], direction[2], corners[2]);
	intersection(center[2], direction[2], center[3], direction[3], corners[3]);
	intersection(center[3], direction[3], center[0], direction[0], corners[0]);


	for (int i = 0; i < 4; i++)
	{
		int i1 = i + 1;
		if (i1 > 3) i1 = 0;
		if (truck)
			line(img, corners[i], corners[i1], Scalar(0xff, 0xff, 0x0), 2);
		else
			line(img, corners[i], corners[i1], Scalar(0x0, 0x0, 200), 2);
		std::stringstream textstream;
		int fontFace = FONT_HERSHEY_SIMPLEX;
		double fontScale = 1;
		int thickness = 2;
		Point pos(0., 0.);
		circle(img, center[i], 8, Scalar(0x0, 0x0, 200), -1);
		circle(img, center[i] + direction[i], 8, Scalar(200, 0x0, 0), -1);

	}
}

// check if either a center point or a direction point was selected by the user
int Lane::CheckForHit(Point mouse)
{
	hit = -1;
	for (int i = 0; i < 4; i++)
	{
		Point d = mouse - center[i];
		double l = sqrt(d.x*d.x + d.y*d.y);
		if (l < 10)
		{
			hit = i;
			return hit+1;
		}
		d = mouse - (center[i] + direction[i]);
		l = sqrt(d.x*d.x + d.y*d.y);
		if (l < 10)
		{
			hit = i+10;
			return hit + 1;
		}
	}
	return 0;
}

// this class defines a license plate which travels in a defined direction and contains a collection of possible matches for the plate
// since this is an OCR process the results when reading a plate vary. this class keeps meaningful readings of the license plate together with how often the same reading happend and the highest confidence of this reading
// based on this the most likely license plate string will be chosen
class Plate
{
public:
	Plate(Point _center, Rect _roi, __time64_t _time, string _plateFolder)
	{
		// a new plate was found
		center = _center;
		roi = _roi;
		time = _time; // the first time the plate was spotted
		plateFolder = _plateFolder; // the folder in which a representative thumbnail will be stored
		skippedFrames = 0;
		direction = Point(0, 0);
		maxWidth = 0;
	};
	~Plate()
	{}
	bool CheckForPlate(Point newCenter);
	int GetSkippedFrames() { return skippedFrames; }
	int InsertCandidates(alpr::AlprPlateResult plate, Mat frame);
	bool IsPlateActive() { return skippedFrames < 20; };
	void SkipFrame() { skippedFrames++; };
	bool Write(ofstream &txtfile, ofstream &dir1, ofstream &dir2);
private:
	Point center; // the center of the plate travels with the plate in time
	Point direction; // the direction in which the plate travells based on previews hits
	int skippedFrames = 0; // how many frames passed before the plate was redetected again
	typedef std::map < std::string, std::pair<int, float>> PlatesMap; // this map stores the possible matches for the license plate with the number of times it was found and the highest confidence value of the match
	PlatesMap mapPlates; 
	Mat plateFrame; //  contains the best match for the plate
	string plateFolder; // the folder in which the thumbnai will be stored
	float maxConf = 0.0f; // highest confidence value during OCR
	int maxHits = 0; // maximum number a single plate was identified
	int maxChars = 0; // maxum number of characters in the plate
	int maxWidth; // maximum width in pixels
	std::string szPlate; // final reading of the plate based on a number of criteria
	Rect roi; 
	__time64_t time; // time the plate was read for the first time
};

// is the newly found plate on the path of this plate
// if this is likely, the new matches will be added to this plate
bool Plate::CheckForPlate(Point newCenter)
{
	Point d = newCenter - center;
	double dist = sqrt(d.x*d.x + d.y*d.y);
	if (dist > 180)
	{
		skippedFrames++;
		return false;
	}
	center = newCenter;
	direction += d;
	skippedFrames = 0;
	return true;
}

// possible matches will be added to the map
int Plate::InsertCandidates(alpr::AlprPlateResult plate, Mat frame)
{
	int ins = 0;
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

	int x = left;
	int y = bottom;
	int h = top - bottom;
	int w = right - left;

	//if (h < 10)
	//	continue;
	x = max(0, x);
	x = min(x, roi.width - w - 1);
	y = max(0, y);
	y = min(y, roi.height - h - 1);
#if 0
	if (w > maxWidth)
	{
		maxWidth = w;
		resize(frame, plateFrame, Size(w, h));
		frame(Rect(x, y, w, h)).copyTo(plateFrame);
	}
#endif

	// iterate over all possible permutations
	for (int k = 0; k < plate.topNPlates.size(); k++)
	{
		alpr::AlprPlate candidate = plate.topNPlates[k];
#ifdef _debugout
		std::cout << "    - " << candidate.characters << "\t confidence: " << candidate.overall_confidence;
		std::cout << "\t pattern_match: " << candidate.matches_template << std::endl;
#endif
		// if there are too many or too few characters in the plate skip it
		if (candidate.characters.length() > 7)
			continue;
		if (candidate.characters.length() < 5)
			continue;

		ins++;
		// check if the confidence of this match is higher than previous matches
		// keep the image of this plate to store it together with the plate matches
		if (candidate.overall_confidence > maxConf)
		{
			maxConf = candidate.overall_confidence;
			szPlate = candidate.characters;
			resize(frame, plateFrame, Size(w, h));
			frame(Rect(x, y, w, h)).copyTo(plateFrame);
		}

		if (candidate.characters.length() > maxChars)
			maxChars = candidate.characters.length();
		// check if the map contains this match alreay. increase the number of hits and adjust the confidence value
		auto it = mapPlates.find(candidate.characters);
		if (it != mapPlates.end())
		{
			it->second.first = it->second.first + 1;
			if (it->second.first > maxHits)
				maxHits = it->second.first;
			if (candidate.overall_confidence > it->second.second)
				it->second.second = candidate.overall_confidence;
		}
		else
		{
			// add a new entry to the map
			mapPlates.insert(std::make_pair(candidate.characters, std::make_pair(1, candidate.overall_confidence)));
		}
	}
	return 1;
}

// if the plate is no longer readable write the results to disk
// a .csv and an html file are written for viewing and later processing
bool Plate::Write(ofstream &txtfile, ofstream &dir1, ofstream &dir2 )
{
	//if (maxConf < 50.0)
	//	return;
	if (maxHits < 2)
		return false;
	//if (mapPlates.size() < 10)
	//	return;
	if (direction.x == 0) direction.x = -1;
	if (direction.y == 0) direction.y = -1;
	ofstream *dirfile = &dir1;
	if (direction.x > 0)
		dirfile = &dir2;

	std::cout << "time: " << put_time(localtime(&time), "%c") ;
	txtfile << put_time(localtime(&time), "%H:%M:%S");  //put_time(localtime(&time), "%c");
	*dirfile << put_time(localtime(&time), "%H:%M:%S");  //put_time(localtime(&time), "%c");
														
//	*txtfile << ";" << direction.x/abs(direction.x) << ";" << direction.y/abs(direction.y) ;
	std::stringstream textstream;
	// now the result is in `buffer.str()`.				string text;
	textstream << put_time(localtime(&time), "%H%M%S");
	bool writefile = true;



	int lines = 0;
	// check if the plate with the max confidence is leading by at least 5%
	bool highConf = true;
	for (auto const &it : mapPlates)
	{
		if (it.first != szPlate && it.second.first == maxHits && it.first.length() == maxChars)
		{
			if (it.second.second > maxConf - 5.)
			{
				highConf = false;
				std::cout << "    - " << it.first << "\t confidence: " << it.second.second << std::endl;
			}
		}
	}
	if (highConf)
	{
		std::cout << "    - " << szPlate << "\t confidence: " << maxConf << std::endl;
		txtfile << ";" << szPlate;
		*dirfile << ";" << szPlate;
		lines = 1;
	}
	//			txtfile << "    - " << szPlate << "\t confidence: " << maxConf << std::endl;
	for (int iChars = maxChars; iChars > 3; iChars--)
	{
		for (int hits = maxHits; hits > max(maxHits/3,1) && lines < 5; hits--)
		{
			for (auto const &it : mapPlates)
			{
				if (it.first != szPlate && it.second.first == hits && it.first.length() == iChars)
				{
					std::cout << "    - " << it.first << "\t found: " << it.second.first << "\t confidence: " << it.second.second << std::endl;
					txtfile << ";" << it.first;
					*dirfile << ";" << it.first;
					//txtfile << ";" << it.second.first;
					//txtfile << ";" << it.second.second;
					if (!highConf && writefile)
					{
						writefile = false;
						string outfile = plateFolder + textstream.str() + "_" + it.first + ".jpg";
						cv::imwrite(outfile, plateFrame);
					}
					lines++;
					if (lines > 4) break;
				}
			}
		}
	}
	txtfile << std::endl;
	*dirfile << std::endl;
	if (writefile)
	{
		string outfile = plateFolder + textstream.str() + "_" + szPlate + ".jpg";
		cv::imwrite(outfile, plateFrame);
	}
	return true;

}


class PlateRecognizer
{
public:
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
	
#define _debugwin
#ifdef _debugwin
		namedWindow(plateFolder, CV_WINDOW_KEEPRATIO); //resizable window;
		resizeWindow(plateFolder, 100, 100);
#endif
	}
	~PlateRecognizer() {
	}
	int AnalzyeFrame(Mat src, __time64_t time);
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
	std::vector <Plate*> vecPlates;
	Mat plateFrame;
	Rect roi;
	bool reco = false;

};






void PlateRecognizer::CheckForInactivePlates()
{
	auto it = std::begin(vecPlates);
	while (it != std::end(vecPlates)) {
		// skip if no plate was found

		if (!(*it)->IsPlateActive())
		{
			if ((*it)->Write(txtfile, txtfile1, txtfile2))
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

int PlateRecognizer::AnalzyeFrame(Mat src, __time64_t time)
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
	imshow(plateFolder, frame);
	circle(frame, Point(20, 20), 10, Scalar(0xff, 0xff, 0xff), -1);


#if 1
	processFrames--;
	if (processFrames > 0)
		reco = true;
	else
		reco = false;

	if (!reco || true)
	{
		absdiff(lastFrame, frame, diff);
		// WARNING: this will weight channels differently! - instead you might want some different metric here. e.g. (R+B+G)/3 or MAX(R,G,B)
		cvtColor(diff, diff1Channel, CV_BGR2GRAY);

		mask = diff1Channel >  mthreshold;

		uchar* data = mask.data;
		int hits = 0;
//		for (int i = 0; i < mask.rows && !reco; i++)
		for (int i = 0; i < mask.rows ; i++)
		{
			for (int j = 0; j < mask.cols; j++)
			{
				if ((int)*data > 0)
					hits++;
				data++;
			}
		}
		if ((double)hits / (double)(mask.cols*mask.rows) > 0.01)
		{
			reco = true;
			processFrames = 50;
			motionFrames++;
			noMotionFrames = 0;
		}
		else
		{
			noMotionFrames++;
			if (noMotionFrames == 20)
			{
				if (motionFrames > 40)
				{
					txtfile << put_time(localtime(&time), "%H:%M:%S") << ";-----" << endl;  //put_time(localtime(&time), "%c");
					std::cout << put_time(localtime(&time), "%H:%M:%S") << ";-----" << endl;  //put_time(localtime(&time), "%c");
					motionFrames++;
				}
				motionFrames = 0;
			}
		}
		if (reco)
		{
			circle(frame, Point(20, 20), 10, Scalar(0xff, 0xff, 0xff), -1);
		}
		frame.copyTo(lastFrame);
	}
#else
	reco = true;
#endif
	if (!reco)
	{
		for (auto it = std::begin(vecPlates); it < std::end(vecPlates); it++)
		{
			(*it)->SkipFrame();
		}
		CheckForInactivePlates();

		return 0;
	}

	lane.MaskOut(frame);

	regionsOfInterest.clear();
	AlprResults results = mAlpr->recognize(frame.data, 3, frame.cols, frame.rows, regionsOfInterest);

	for (int i = 0; i < results.plates.size(); i++)
	{
		std::stringstream textstream;
		int fontFace = FONT_HERSHEY_SIMPLEX;
		double fontScale = 1;
		int thickness = 2;
		{
			cout << "Plate " << i ;
			alpr::AlprPlateResult plate = results.plates[i];
			for (int k = 0; k < plate.topNPlates.size() && k < 5; k++)
			{
				alpr::AlprPlate candidate = plate.topNPlates[k];
				cout << " " << candidate.characters << " " << candidate.overall_confidence;
				textstream << candidate.characters;
			}
			cout << endl;
		}

		alpr::AlprPlateResult plate = results.plates[i];
		bool found = false;
		if (plate.topNPlates.size() < 1)
			continue;
		alpr::AlprPlate candidate = plate.topNPlates[0];

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
			if ((*it)->CheckForPlate(center))
			{
				found = true;
				itfound = it;
			}
		}
		int ins = 0;
		if (!found)
		{
			Plate *newPlate = new Plate(center, roi, time, plateFolder);
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
	if (ac == 3) reco = true;



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

	do {
		string szFoundFile = szPath + FindFileData.cFileName;
//		szFoundFile = "D:\\KZVerfolgung\\Ibk\\GP040006.mp4";
		szFoundFile = "D:\\BVR\\StJohann\\V1_1080p_30fps.mp4";
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
		if (create_window)
		{
			namedWindow(window_name, CV_WINDOW_KEEPRATIO); //resizable window;
			resizeWindow(window_name, width / 2, height / 2);
			setMouseCallback(window_name, onMouse, nullptr);
			create_window = false;
			//namedWindow("mask", CV_WINDOW_KEEPRATIO); //resizable window;
			//resizeWindow("mask", width / 2, height / 2);
		}

		capture >> frame;

		int imgFound;
		for (;;)
		{
			if (play)
			{
				capture >> newFrame;
				for (int c = 0; c < speed - 1 && !imgFound && !reco; c++)
				{
					capture >> newFrame;
				}
				auto CurrTime = EncDate + std::chrono::milliseconds((long)capture.get(CV_CAP_PROP_POS_MSEC));
				frameTime = std::chrono::system_clock::to_time_t(CurrTime);
				if ((int)capture.get(CV_CAP_PROP_POS_FRAMES) > frame_count - 5)
					break;
				if (slomo)
					Sleep(1000);
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

			if (!frame.empty())
			{
				if (reco)
				{
					imgFound = 0;
					for (auto it = std::begin(plateRecognizer); it < std::end(plateRecognizer); it++)
					//auto it = std::begin(plateRecognizer);
					{
						imgFound += (*it)->AnalzyeFrame(frame, frameTime);
					}
				}
				else
				{
					imgFound = 0;
				}

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
					putText(frame, textstream.str(), Point(roi.x+10, roi.y+25), fontFace, fontScale,
						Scalar::all(255), thickness, 8);
					i++;
				}

				showImage++;
				if (true) // imgFound || showImage > 5)
				//if (showImage >= speed)
				{
					// now the result is in `buffer.str()`.				string text;
					textstream.str("");
					textstream.clear();
					textstream << put_time(localtime(&frameTime), "%H:%M:%S speed ") << speed << " contrast " << contrast;
					if (reco)
						textstream << " R";
					// then put the text itself
					putText(frame, textstream.str(), Point(10, 25), fontFace, fontScale,
						Scalar(0,0,200), thickness, 8);

					//int pw = plateFrame.cols;
					//int ph = plateFrame.rows;
					//plateFrame.copyTo(lastFrame.rowRange(50, 50 + ph).colRange(0, pw));
					imshow(window_name, frame);
					if (showImage >= speed)
					{
						showImage = 0;
					}
				}
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
							imgFound += (*it)->AnalzyeFrame(frame, frameTime);
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
				case 'l':
					reco = !reco;
					if (reco) play = true;
					speed = 1;
					break;
				case '1':
					roiIndex = 1;
					break;
				case '2':
					if (plateRecognizer.size() < 3)
						plateRecognizer.push_back(new PlateRecognizer(szPath + "Lane2\\",1)); //&direction1);
					roiIndex = 2;
					break;
				case '3':
					if (plateRecognizer.size() < 3)
						plateRecognizer.push_back(new PlateRecognizer(szPath + "Lane3\\",2)); //&direction1);
					roiIndex = 3;
				case '4':
					if (plateRecognizer.size() < 4)
						plateRecognizer.push_back(new PlateRecognizer(szPath + "Lane4\\",3)); //&direction1);
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
					conf << contrast;
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

//		process(capture);
	} while (FindNextFile(hFind, &FindFileData));
	FindClose(hFind);


}
