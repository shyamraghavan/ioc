/*
 *  transport.cpp
 *  Transfer to separate scene
 *
 *  Created by Shyam Raghavan on 05/06/17.
 *  Copyright 2017. All rights reserved.
 *

 Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

 THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include "transfer.hpp"

void colormap(const Mat _src, Mat &dst)
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

void Transfer::initialize()
{
  _prev_na = 9;
}

void Transfer::loadPrevBasenames(string input_filename)
{
	cout << "\nLoadPrevBasenames()\n";
	ifstream fs;
	fs.open(input_filename.c_str());
	if(!fs.is_open()){cout << "ERROR: Opening: " << input_filename << endl;exit(1);}
	string str;
	while(fs >> str){
		if(str.find("#")==string::npos) _prev_basenames.push_back(str);
	}
	_prev_nd = (int)_prev_basenames.size();
	if(VERBOSE) cout << "  Number of basenames loaded:" << _prev_nd << endl;
}

void Transfer::loadPrevFeatMap(string input_file_prefix)
{
	cout << "\nLoadPrevFeatMap()\n";

	for(int i=0;i<_prev_nd;i++)
	{
		_prev_featmap.push_back(vector<cv::Mat>(0));

		string input_filename = input_file_prefix + _prev_basenames[i] + "_feature_maps.xml";
		FileStorage fs(input_filename.c_str(), FileStorage::READ);
		if(!fs.isOpened()){cout << "ERROR: Opening: " << input_filename << endl;exit(1);}

		for(int j=0;true;j++)
		{
			stringstream ss;
			ss << "feature_" << j;
			Mat tmp;
			fs[ss.str()] >> tmp;
			if(!tmp.data) break;
			_prev_featmap[i].push_back(tmp+0.0);
		}
		_nf = (int)_prev_featmap[i].size() - 3;
		_prev_size = _prev_featmap[i][0].size();
    if(VERBOSE)
    {
      printf(
        "  %s: Number of features loaded is %d\n",
        _prev_basenames[i].c_str(),
        _nf
      );
      printf(
        "  %s: State space loaded is %d x %d\n",
        _prev_basenames[i].c_str(),
        _prev_size.height,
        _prev_size.width
      );
    }
	}
}

void Transfer::loadPrevReward(string input_filename)
{
  cout << "\nLoadPrevReward()\n";

  ifstream fs;
  fs.open(input_filename.c_str());
  if(!fs.is_open()){cout << "ERROR: Opening: " << input_filename << endl;exit(1);}

  string str;
  Mat p(_prev_size.height * _prev_size.width * _prev_nd, 1, CV_32FC1);

  int x = 0;

  while(getline(fs,str))
  {
    int a = 0;
    float sum = 0.0;

    size_t i = 0;
    while(a < _prev_na)
    {
      float r = stof(str, &i);
      if (r == r)
      {
        sum += r;
      }
      str = str.substr(i);

      a++;
    }

    if (sum == 0.0)
    {
      sum = nanf("");
    }

    p.at<float>(x) = sum / _prev_na;
    x++;
  }

  _prev_R = p.clone();
}

void Transfer::reshapePrevFeatMap()
{
  cout << "\nReshapePrevFeatMap()\n";

  _prev_feats = Mat(_prev_size.width * _prev_size.height * _prev_nd, _nf, CV_32FC1);

  for (int t=0;t<_prev_nd;t++)
  {
    for (int y=0;y<_prev_size.height;y++)
    {
      for (int x=0;x<_prev_size.width;x++)
      {
        for (int f=0;f<_nf;f++)
        {
          int index = x + _prev_size.width * y + _prev_size.width * _prev_size.height * t;
          float f_val = _prev_featmap[t][f].at<float>(y,x);

          _prev_feats.at<float>(index,f) = f_val;
        }
      }
    }
  }
}

void Transfer::loadBasenames(string input_filename)
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

void Transfer::loadFeatMap(string input_file_prefix)
{
	cout << "\nLoadFeatMap()\n";

	for(int i=0;i<_nd;i++)
	{
		_featmap.push_back(vector<cv::Mat>(0));

		string input_filename = input_file_prefix + _basenames[i] + "_feature_maps.xml";
		FileStorage fs(input_filename.c_str(), FileStorage::READ);
		if(!fs.isOpened()){cout << "ERROR: Opening: " << input_filename << endl;exit(1);}

		for(int j=0;true;j++)
		{
			stringstream ss;
			ss << "feature_" << j;
			Mat tmp;
			fs[ss.str()] >> tmp;
			if(!tmp.data) break;
			_featmap[i].push_back(tmp+0.0);
		}
		_nf = (int)_featmap[i].size();
		_size = _featmap[i][0].size();
    if(VERBOSE)
    {
      printf(
        "  %s: Number of features loaded is %d\n",
        _basenames[i].c_str(),
        _nf
      );
      printf(
        "  %s: State space loaded is %d x %d\n",
        _basenames[i].c_str(),
        _size.height,
        _size.width
      );
    }
	}
}

void Transfer::visualizeFeats()
{
  cout << "\nVisualizeFeats()\n";

  for(int f=0;f<_nf;f++)
  {
    if(VISUALIZE)
    {
      Mat dst;
      colormap(_featmap[0][f],dst);
      addWeighted(_image[0],0.5,dst,0.5,0,dst);
      imshow("Feature " + to_string(f),dst);
      waitKey(0);
    }
  }
}

void Transfer::loadImages(string input_file_prefix)
{
	cout << "\nLoadImages()\n";

	for(int i=0;i<_nd;i++)
	{
		string input_filename = input_file_prefix + _basenames[i] + "_birdseye.jpg";
		Mat im = imread(input_filename);
		if(!im.data){cout << "ERROR: Opening:" << input_filename << endl; exit(1);}
		if(VERBOSE) cout << "  Loading: " << input_filename << endl;
		resize(im,im,_featmap[0][0].size());
		_image.push_back(im);
	}
	if(VERBOSE) cout << "  Number of images loaded: " << _image.size() << endl;
}

void Transfer::computeNewRewardFunPoint(Transfer *inst, void *args)
{
  par_arg *arg = (par_arg *)args;
  Mat R = *(Mat *)(arg->R);
  int y = arg->y;

  Mat mask = inst->_prev_R==inst->_prev_R;

  for(int x=0;x<inst->_size.width;x++)
  {
    Mat pt(1,inst->_nf,CV_32FC1);
    Mat ptf;
    Mat err;

    Point idx;

    for(int f=0;f<inst->_nf;f++)
    {
      pt.at<float>(0,f) = inst->_featmap[0][f].at<float>(y,x);
    }

    repeat(pt,inst->_prev_size.width*inst->_prev_size.height*inst->_prev_nd,1,ptf);
    multiply(ptf - inst->_prev_feats,ptf - inst->_prev_feats,ptf);
    reduce(ptf,err,1,CV_REDUCE_SUM);
    minMaxLoc(err,NULL,NULL,&idx,NULL,mask);

    int index = idx.y;

    R.at<float>(y * inst->_size.width + x) = inst->_prev_R.at<float>(index);

    printf("Reward Function at %d %d: %f (%d)\n",
      y, x, R.at<float>(y * inst->_size.width + x), index);
  }
}

void Transfer::computeNewRewardFun()
{
  cout << "\nComputeNewRewardFun()\n";

  const size_t NUM_THREADS = 16;

  Mat _R(_size.height * _size.width, 1, CV_32FC1);
  vector<thread> threads;
  par_arg args[NUM_THREADS];

  for(int y=0;y<_size.height;y+=NUM_THREADS)
  {
    for (int i=0;i<NUM_THREADS;i++)
    {
      if (y+i < _size.height)
      {
        args[i].y = y+i;
        args[i].R = &_R;

        threads.push_back(thread(computeNewRewardFunPoint, this, &(args[i])));
      }
    }

    for (int i=0;i<NUM_THREADS;i++)
    {
      if (y+i < _size.height)
      {
        threads[0].join();
        threads.erase(threads.begin());
      }
    }

    printf("Row %d Estimation Completed\n", y);
  }
}

void Transfer::saveNewRewardFun(string output_filename)
{
  cout << "\nSaveNewRewardFun()\n";

  ofstream fs(output_filename.c_str());
  if(!fs.is_open()) cout << "ERROR: Writing: " << output_filename << endl;
  for(int y=0;y<_size.height;y++)
  {
    for(int x=0;x<_size.width;x++)
    {
      int index = x + y * _size.width;
      fs << to_string(_R.at<float>(index)) << endl;
    }
  }
  fs.close();
}
