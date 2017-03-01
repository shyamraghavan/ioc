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

void CCP::initialize()
{
  _na = 9;
  _h = 2;
  _hf = 0.1;

  _a0 = 5;
  _B = .9;
  _E = 0.0000000001;
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
		_nf = (int)_featmap[i].size() - 3;
		_size = _featmap[i][0].size();
		if(VERBOSE) printf("  %s: Number of features loaded is %d\n",_basenames[i].c_str(), _nf);
    if(VERBOSE) printf("  %s: State space loaded is %d x %d\n",_basenames[0].c_str(), _size.height, _size.width);
	}
}


void CCP::visualizeFeats()
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


void CCP::estimatePolicyPoint(CCP *inst, void *args)
{
  par_arg *arg = (par_arg *)args;
  Mat probs = *(Mat *)(arg->probs);
  int y = arg->y;

  for(int x=0;x<inst->_size.width;x+=1)
  {
    vector<float> actionSums(0);

    for(int a=0;a<inst->_na;a++)
    {
      float numerator = 0.0;
      float denominator = 0.0;

      for(int i=0;i<inst->_nd;i++)
      {
        vector<Point> trajgt = inst->_trajgt[i];

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
            printf("ERROR: Invalid action %d(%d,%d)\n",t,dx,dy);
            printf("Preprocess trajectory data properly.\n");
            exit(1);
          }

          float total_exp = 0;
          total_exp += pow(((float)(x_it.x - x)) / inst->_h, 2.0);
          total_exp += pow(((float)(x_it.y - y)) / inst->_h, 2.0);

          for(int f=0;f<inst->_nf;f++)
          {
            float feat =
              inst->_featmap[i][f].at<float>(x_it) -
              inst->_featmap[i][f].at<float>(y,x);
            total_exp += pow(feat / inst->_hf, 2.0);
          }

          numerator += (a == a_it) ? exp(-total_exp / 2.0) : 0;
          denominator += exp(-total_exp / 2.0);
        }
      }
      probs.at<Vec9f>(y,x)[a] = numerator / denominator;
    }
  }

  if (inst->VERBOSE) printf("  Estimated at row %d\n", y);
}

void CCP::estimatePolicy()
{
  cout << "\nEstimatePolicy()\n";

  const size_t NUM_THREADS = 16;

  _probs = Mat::zeros(_size, CV_32FC(9));
  vector<thread> threads;
  par_arg args[NUM_THREADS];

  for(int y=0;y<_size.height;y+=NUM_THREADS)
  {
    for (int i=0;i<NUM_THREADS;i++)
    {
      if (y+i < _size.height)
      {
        args[i].y = y+i;
        args[i].probs = &_probs;

        threads.push_back(thread(estimatePolicyPoint, this, &(args[i])));
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

    if(VISUALIZE)
    {
      vector<Mat> actionProbs(9);
      split(_probs, actionProbs);

      Mat dst;
      colormap(actionProbs[0],dst);
      addWeighted(_image[0],0.5,dst,0.5,0,dst);
      imshow("Action " + to_string(0),dst);
      waitKey(1);
    }
  }

  threshold(_probs, _probs, FLT_MIN, 0, THRESH_TOZERO);

  vector<Mat> actionProbs(9);
  split(_probs, actionProbs);

	if(VISUALIZE)
	{
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

void CCP::savePolicy(string output_filename)
{
  cout << "\nSavePolicy()\n";

  ofstream fs(output_filename.c_str());
  if(!fs.is_open()) cout << "ERROR: Writing: " << output_filename << endl;
  for(int y=0;y<_size.height;y++)
  {
    for(int x=0;x<_size.width;x++)
    {
      for(int a=0;a<_na;a++)
      {
        fs << to_string(_probs.at<Vec9f>(y,x)[a]) << "\t";
      }
      fs << endl;
    }
  }
}

void CCP::readPolicy(string input_filename)
{
  cout << "\nReadPolicy()\n";

  ifstream fs;
  fs.open(input_filename.c_str());
  if(!fs.is_open()){cout << "ERROR: Opening: " << input_filename << endl;exit(1);}

  string str;
  Mat p(1, _size.height * _size.width, CV_32FC(9));

  int x = 0;

  while(getline(fs,str))
  {
    int a = 0;
    size_t l = str.length();

    size_t i = 0;
    while(a < 9)
    {
      p.at<Vec9f>(0,x)[a] = stof(str, &i);
      str = str.substr(i);

      a++;
    }
    x++;
  }

  _probs = p.reshape(0, _size.height).clone();
}

void CCP::estimateGamma()
{
  cout << "\nEstimateGamma()\n";

  vector<float> gamma(0);
  _gammaM = Mat(_size.height, _size.width, CV_32FC(9));

  for(int y=0;y<_size.height;y++)
  {
    for(int x=0;x<_size.width;x++)
    {
      float val = 0.0;

      for(int a=0;a<_na;a++)
      {
        float n = _probs.at<Vec9f>(y,x)[a];
        float d = _probs.at<Vec9f>(y,x)[_a0];

        if(d != 0)
        {
          _gammaM.at<Vec9f>(y,x)[a] = n/d;
          val += log(1.0 + n/d);
        } else {
          _gammaM.at<Vec9f>(y,x)[a] = nanf("");
          val = nanf("");
        }
      }

      gamma.push_back(val);
    }
  }

  if(VERBOSE)
  {
    const string a("./ioc_demo/walk_output/gamma.txt");
    ofstream fs(a.c_str());
    if(!fs.is_open()) cout << "ERROR: Writing: " << a << endl;
    for(int y=0;y<_size.height;y++)
    {
      for(int x=0;x<_size.width;x++)
      {
        for(int a=0;a<_na;a++)
        {
          fs << to_string(_gammaM.at<Vec9f>(y,x)[a]) << "\t";
        }
        fs << endl;
      }
    }
  }

  _gamma = Mat(gamma, true); // _size.height * _size.width x 1
  Mat _stuff = _gamma.clone();

  if(VISUALIZE)
  {
    _gamma = _gamma.reshape(0,_size.height);

    Mat dst;
    colormap(_gamma,dst);
    addWeighted(_image[0],0.5,dst,0.5,0,dst);
    imshow("Gamma Vector",dst);
    waitKey(0);

    _gamma = _gamma.reshape(0,_size.height*_size.width);
  }
}

void CCP::estimateTransitionMatrix()
{
  cout << "\nEstimateTransitionMatrix()\n";

  int ns = _size.height * _size.width;
  int size[] = {ns,ns};

  _T = SparseMat(2,size,CV_32FC1);

  for(int x=0;x<ns;x++)
  {
    int dx = 0;
    int dy = 0;

    if(_a0 == 0) { dx=-1; dy=-1; }
    if(_a0 == 1) { dx=0; dy=-1; }
    if(_a0 == 2) { dx=1; dy=-1; }
    if(_a0 == 3) { dx=-1; dy=0; }
    if(_a0 == 4) { printf("Cannot let _a0 be no movement\n"); exit(0); }
    if(_a0 == 5) { dx=1; dy=0; }
    if(_a0 == 6) { dx=-1; dy=1; }
    if(_a0 == 7) { dx=0; dy=1; }
    if(_a0 == 8) { dx=1; dy=1; }

    int dstate = _size.width * dy + dx;

    if (x+dstate<ns && x+dstate>0)
    {
      _T.ref<float>(x,x+dstate) = 1.0;
    }
  }

  if(VERBOSE) cout << "  Finished writing transition matrix\n";
}


void multiply(SparseMat A, Mat x, Mat *dst)
{
  Mat dst_p = Mat::zeros(A.size(0), 1, CV_32FC1);

  for(SparseMatIterator iter = A.begin(); iter != A.end(); iter++)
  {
    int i = iter.node()->idx[0];
    int j = iter.node()->idx[1];

    dst_p.at<float>(i,0) += iter.value<float>() * x.at<float>(j,0);
  }

  dst_p.copyTo(*dst);
}


bool equal(Mat a, Mat b, int height, float _E)
{
  for(int i=0; i<height; i++)
  {
    if (abs(a.at<float>(i,0) - b.at<float>(i,0)) > _E) return false;
  }

  return true;
}


void CCP::estimateZeroValueFunction()
{
  cout << "\nEstimateZeroValueFunction()\n";

  int ns = _size.height * _size.width;
  int size[] = {ns,ns};

  SparseMat BT;
  _T.copyTo(BT);

  for(SparseMatIterator_<float> x = BT.begin<float>(); x != BT.end<float>(); x++)
  {
    *x = (*x) * _B;
  }

  Mat V0 = Mat(BT.size(0), 1, CV_32FC1, 0.0);
  Mat V0_original = V0.clone();

  multiply(BT, V0_original, &V0);
  V0 = V0 + (_B * _gamma);

  int index = 0;

  while(!equal(V0_original,V0,ns,_E))
  {
    V0.copyTo(V0_original);
    multiply(BT, V0_original, &V0);
    V0 = V0 + (_B * _gamma);

    if (VERBOSE && index % 10 == 0)
      printf("  %d value iterations complete\n", index);

    index++;
  }

  if (VERBOSE) printf("  Converged in %d iterations\n", index);

  V0 = V0.reshape(0, _size.height);
  _gamma = _gamma.reshape(0, _size.height);

  _V0 = V0.clone();

  if (VISUALIZE)
  {
    Mat dst;
    colormap(_V0,dst);
    addWeighted(_image[0],0.5,dst,0.5,0,dst);
    imshow("Value Function for 0-Reward Action",dst);
    waitKey(0);
  }
}

void CCP::estimateValueFunction()
{
  cout << "\nEstimateValueFunction()\n";

  _V = Mat(_size, CV_32FC(9));

  for(int y=0;y<_size.height;y++)
  {
    for(int x=0;x<_size.width;x++)
    {
      for(int a=0; a<_na; a++)
      {
        float n = _probs.at<Vec9f>(y,x)[a];
        float d = _probs.at<Vec9f>(y,x)[_a0];

        if(d != 0 && n != 0)
        {
          _V.at<Vec9f>(y,x)[a] = log(n/d) + _V0.at<float>(y,x);
        } else {
          _V.at<Vec9f>(y,x)[a] = nanf("");
        }
      }
    }
  };
}

void CCP::visualizeValueFunction()
{
  cout << "\nVisualizeValueFunction()\n";

  if (VISUALIZE)
  {
    vector<Mat> value(9);
    split(_V, value);

    for(int a=0;a<_na;a++)
    {
      Mat dst;
      colormap(value[a],dst);
      addWeighted(_image[0],0.5,dst,0.5,0,dst);
      imshow("Action " + to_string(a),dst);
      waitKey(0);
    }
  }
}

void CCP::saveValueFunction(string output_filename)
{
  cout << "\nSaveValueFunction()\n";

  ofstream fs(output_filename.c_str());
  if(!fs.is_open()) cout << "ERROR: Writing: " << output_filename << endl;
  for(int y=0;y<_size.height;y++)
  {
    for(int x=0;x<_size.width;x++)
    {
      for(int a=0;a<_na;a++)
      {
        fs << to_string(_V.at<Vec9f>(y,x)[a]) << "\t";
      }
      fs << endl;
    }
  }
}

void CCP::readValueFunction(string input_filename)
{
  cout << "\nReadValueFunction()\n";

  ifstream fs;
  fs.open(input_filename.c_str());
  if(!fs.is_open()){cout << "ERROR: Opening: " << input_filename << endl;exit(1);}

  string str;
  Mat p(1, _size.height * _size.width, CV_32FC(9));

  int x = 0;

  while(getline(fs,str))
  {
    int a = 0;
    size_t l = str.length();

    size_t i = 0;
    while(a < 9)
    {
      p.at<Vec9f>(0,x)[a] = stof(str, &i);
      str = str.substr(i);

      a++;
    }
    x++;
  }

  _V = p.reshape(0, _size.height).clone();
}

void CCP::estimateRewardFunction()
{
  cout << "\nEstimateRewardFunction()\n";

  Mat R(_size, CV_32FC(9));

  for(int y=0;y<_size.height;y++)
  {
    for(int x=0;x<_size.width;x++)
    {
      for(int a=0;a<_na;a++)
      {
        float V_ax = _V.at<Vec9f>(y,x)[a];

        int dx = 0;
        int dy = 0;

        if(a == 0) { dx=-1; dy=-1; }
        if(a == 1) { dx=0; dy=-1; }
        if(a == 2) { dx=1; dy=-1; }
        if(a == 3) { dx=-1; dy=0; }
        if(a == 4) { dx=0; dy=0; }
        if(a == 5) { dx=1; dy=0; }
        if(a == 6) { dx=-1; dy=1; }
        if(a == 7) { dx=0; dy=1; }
        if(a == 8) { dx=1; dy=1; }

        int y_n = max(0, min(_size.height - 1, y + dy));
        int x_n = max(0, min(_size.width - 1, x + dx));

        float r = V_ax - _B * (_V0.at<float>(y_n,x_n) + _gamma.at<float>(y,x));

        R.at<Vec9f>(y,x)[a] = r;
      }
    }
  }

  _R = R.clone();
}

void CCP::visualizeRewardFunction()
{
  if (VISUALIZE)
  {
    cout << "\nVisualizeRewardFunction()\n";

    vector<Mat> value(9);
    split(_R, value);

    for(int a=0;a<_na;a++)
    {
      Mat dst;
      colormap(value[a],dst);
      addWeighted(_image[0],0.5,dst,0.5,0,dst);
      imshow("Action " + to_string(a),dst);
      waitKey(0);
    }

    Mat act(_size, CV_32FC(9), 0.0);

    for(int y=0;y<_size.height;y++)
    {
      for(int x=0;x<_size.width;x++)
      {
        Point p;

        minMaxLoc(_R.at<Vec9f>(y,x), NULL, NULL, NULL, &p);

        act.at<Vec9f>(y,x)[p.y] = 1.0;
      }
    }

    vector<Mat> value2(9);
    split(act, value2);

    for(int a=0;a<_na;a++)
    {
      Mat dst;
      colormap(value2[a],dst);
      addWeighted(_image[0],0.5,dst,0.5,0,dst);
      imshow("Optimal Action " + to_string(a),dst);
      waitKey(0);
    }
  }
}

void CCP::saveRewardFunction(string output_filename)
{
  cout << "\nSaveRewardFunction()\n";

  ofstream fs(output_filename.c_str());
  if(!fs.is_open()) cout << "ERROR: Writing: " << output_filename << endl;
  for(int y=0;y<_size.height;y++)
  {
    for(int x=0;x<_size.width;x++)
    {
      for(int a=0;a<_na;a++)
      {
        fs << to_string(_R.at<Vec9f>(y,x)[a]) << "\t";
      }
      fs << endl;
    }
  }
}

void CCP::readRewardFunction(string input_filename)
{
  cout << "\nReadRewardFunction()\n";

  ifstream fs;
  fs.open(input_filename.c_str());
  if(!fs.is_open()){cout << "ERROR: Opening: " << input_filename << endl;exit(1);}

  string str;
  Mat p(1, _size.height * _size.width, CV_32FC(9));

  int x = 0;

  while(getline(fs,str))
  {
    int a = 0;
    size_t l = str.length();

    size_t i = 0;
    while(a < 9)
    {
      p.at<Vec9f>(0,x)[a] = stof(str, &i);
      str = str.substr(i);

      a++;
    }
    x++;
  }

  _R = p.reshape(0, _size.height).clone();
}

