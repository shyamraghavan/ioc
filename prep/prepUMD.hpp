/*
 *  prepVirat.h
 *  hioc-virat
 *
 *  Created by Kris Kitani on 5/14/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */
#include <iostream>
#include <sstream>
#include <fstream>
#include <stdlib.h>
#include <vector>
#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;

class prepUMD
{
public:

	void set(string root, int nc, int nr);
	string _root;
	int _nc;					// number of cols
	int _nr;					// number of rows

	void load_basenames(string vidseg_info_fp);

	vector<string>	_basename;
	vector<int>		_t_start;
	vector<int>		_t_end;
	vector<int>		_act;

	void prepare_static_features();

	void prepare_trajectory_features();

private:

	stringstream ss;

	void colormap(Mat src, Mat &dst, int do_norm);
	void flt2img(Mat flt, Mat &img);

};
