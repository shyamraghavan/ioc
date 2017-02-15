/*
 *  ccp.cpp
 *  CCP_DEMO
 *
 *  Created by Shyam Raghavan on 02/14/17.
 *  Copyright 2017. All rights reserved.
 *

 Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

 THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include "ccp.hpp"

void CCP::initialize()
{
  _na = 9;
  _h = 1;
}

void CCP::loadBasenames(string input_filename)
{
	cout << "\nLoadBasenames()\n";
	ifstream fs;
	fs.open(input_filename.c_str());
	if(!fs.is_open()){cout << "ERROR: Opening: " << input_filename << endl;exit(1);}
	string str;
	while(fs >> str){
		if(str.find("#")==string::npos) _basenames.push_back(str);
	}
	_nd = (int)_basenames.size();
	if(VERBOSE) cout << "  Number of basenames loaded:" << _nd << endl;
}


void CCP::loadDemoTraj(string input_file_prefix)
{
	cout << "\nLoadDemoTraj()\n";
	for(int d=0;d<_nd;d++)
	{
		_trajgt.push_back(vector<cv::Point>(0));
		_trajob.push_back(vector<cv::Point>(0));

		string input_filename = input_file_prefix + _basenames[d] + "_tracker_output.txt";
		ifstream fs(input_filename.c_str());
		if(!fs.is_open()){cout << "ERROR: Opening: " << input_filename << endl;exit(1);}

		float val[5];
		int k=0;
		while(fs >> val[k++]){
			if(k==5)
			{
				_trajgt[d].push_back(cv::Point(val[1],val[2]));
				_trajob[d].push_back(cv::Point(val[3],val[4]));
				k=0;
			}
		}

		_start.push_back( _trajgt[d][0] );							        // store start state
		_end.push_back  ( _trajgt[d][_trajgt[d].size()-1] );		// store end state

		if(VERBOSE) printf("  %s: trajectory length is %d\n",_basenames[d].c_str(), (int)_trajgt[d].size());
	}
}


void CCP::loadFeatureMaps(string input_file_prefix)
{
	cout << "\nLoadFeatures()\n";

  _featmap.push_back(vector<cv::Mat>(0));

  string input_filename = input_file_prefix + _basenames[0] + "_feature_maps.xml";
  FileStorage fs(input_filename.c_str(), FileStorage::READ);
  if(!fs.isOpened()){cout << "ERROR: Opening: " << input_filename << endl;exit(1);}

  for(int j=0;true;j++)
  {
    stringstream ss;
    ss << "feature_" << j;
    Mat tmp;
    fs[ss.str()] >> tmp;
    if(!tmp.data) break;
    _featmap[0].push_back(tmp+0.0);
  }
  _nf = (int)_featmap[0].size() - 3;
  _size = _featmap[0][0].size();
  if(VERBOSE) printf("  %s: Number of features loaded is %d\n",_basenames[0].c_str(), _nf);
  if(VERBOSE) printf("  %s: State space loaded is %d x %d\n",_basenames[0].c_str(), _size.height, _size.width);
}


void CCP::loadImages(string input_file_prefix)
{
	cout << "\nLoadImages()\n";

	for(int i=0;i<_nd;i++)
	{
		string input_filename = input_file_prefix + _basenames[i] + "_topdown.jpg";
		Mat im = imread(input_filename);
		if(!im.data){cout << "ERROR: Opening:" << input_filename << endl; exit(1);}
		if(VERBOSE) cout << "  Loading: " << input_filename << endl;
		resize(im,im,_featmap[0][0].size());
		_image.push_back(im);
	}
	if(VERBOSE) cout << "  Number of images loaded: " << _image.size() << endl;
}


void colormap(Mat _src, Mat &dst)
{
	if(_src.type()!=CV_32FC1) cout << "ERROR(jetmap): must be single channel float\n";
	double minVal,maxVal;
	Mat src;
	_src.copyTo(src);
	Mat isInf;
	minMaxLoc(src,&minVal,&maxVal,NULL,NULL);
	compare(src,-FLT_MAX,isInf,CMP_GT);
	threshold(src,src,-FLT_MAX,0,THRESH_TOZERO);
	minMaxLoc(src,&minVal,NULL,NULL,NULL);
	Mat im = (src-minVal)/(maxVal-minVal) * 255.0;
	Mat U8,I3[3],hsv;
	im.convertTo(U8,CV_8UC1,1.0,0);
	I3[0] = U8 * 0.85;
	I3[1] = isInf;
	I3[2] = isInf;
	merge(I3,3,hsv);
	cvtColor(hsv,dst,CV_HSV2RGB_FULL);
}


void CCP::estimatePolicy()
{
  cout << "\nEstimatePolicy()\n";

  Mat probs = Mat::zeros(_size, CV_32FC(9));

  for(int y=0;y<_size.height;y+=2)
  {
    for(int x=0;x<_size.width;x+=2)
    {
      vector<float> actionSums(0);

      for(int a=0;a<_na;a++)
      {
        float numerator = 0.0;
        float denominator = 0.0;

        for(int i=0;i<_nd;i++)
        {
          vector<Point> trajgt = _trajgt[i];

          for(int t=0;t<(int)trajgt.size()-1;t++)
          {
            Point x_it = trajgt[t];

            int dx = trajgt[t+1].x - trajgt[t].x;
            int dy = trajgt[t+1].y - trajgt[t].y;

            int a_it = -1;
            if( dx==-1 && dy==-1 ) a_it = 0;
            if( dx== 0 && dy==-1 ) a_it = 1;
            if( dx== 1 && dy==-1 ) a_it = 2;

            if( dx==-1 && dy== 0 ) a_it = 3;
            if( dx== 0 && dy== 0 ) a_it =-1;	// stopping prohibited
            if( dx== 1 && dy== 0 ) a_it = 5;

            if( dx==-1 && dy== 1 ) a_it = 6;
            if( dx== 0 && dy== 1 ) a_it = 7;
            if( dx== 1 && dy== 1 ) a_it = 8;

            if(a_it<0)
            {
              printf("ERROR: Invalid action %d(%d,%d)\n" ,t,dx,dy);
              printf("Preprocess trajectory data properly.\n");
              exit(1);
            }

            numerator += (a == a_it)
              ? exp(-(pow((x_it.x - x) / _h, 2.0) + pow(((float)(x_it.y - y)) / _h, 2)) / 2.0)
              : 0;
            denominator +=
              exp(-(pow(((float)(x_it.x - x)) / _h, 2.0) + pow(((float)(x_it.y - y)) / _h, 2)) / 2.0);
          }
        }
        actionSums.push_back(numerator);
      }

      float actionSum = sum(actionSums)[0];

      probs.at<Vec9f>(y,x)[0] = actionSums[0] / actionSum;
      probs.at<Vec9f>(y,x)[1] = actionSums[1] / actionSum;
      probs.at<Vec9f>(y,x)[2] = actionSums[2] / actionSum;
      probs.at<Vec9f>(y,x)[3] = actionSums[3] / actionSum;
      probs.at<Vec9f>(y,x)[4] = actionSums[4] / actionSum;
      probs.at<Vec9f>(y,x)[5] = actionSums[5] / actionSum;
      probs.at<Vec9f>(y,x)[6] = actionSums[6] / actionSum;
      probs.at<Vec9f>(y,x)[7] = actionSums[7] / actionSum;
      probs.at<Vec9f>(y,x)[8] = actionSums[8] / actionSum;

      printf("  Estimated at state %d,%d\n", x,y);

      if(VISUALIZE)
      {
        vector<Mat> actionProbs(9);
        split(probs, actionProbs);

        Mat dst;
        colormap(actionProbs[0],dst);
        addWeighted(_image[0],0.5,dst,0.5,0,dst);
        imshow("Action " + to_string(0),dst);
        waitKey(1);
      }
    }
  }

	if(VISUALIZE)
	{
    vector<Mat> actionProbs(9);
    split(probs, actionProbs);

    for(int a=0;a<_na;a++)
    {
      Mat dst;
      colormap(actionProbs[a],dst);
      addWeighted(_image[0],0.5,dst,0.5,0,dst);
      imshow("Action " + to_string(a),dst);
      waitKey(0);
    }
	}
}
