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
  _hf = 1;

  _a0 = Point(80,130);
  _a0_t = 0;
  _B = .95;
  _E = 0.0000000001;

  _a_binwidth = 0.3;
  _gamma_binwidth = 1;

  _samp_size = 100;
  _num_samps = 1000;
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
  int t = arg->t;

  for(int x=0;x<inst->_size.width;x+=1)
  {
    for(int a=0;a<inst->_na;a++)
    {
      float numerator = 0.0;
      float denominator = 0.0;

      int dx,dy;

      if(a == 0) { dx=-1; dy=-1; }
      if(a == 1) { dx=0; dy=-1; }
      if(a == 2) { dx=1; dy=-1; }
      if(a == 3) { dx=-1; dy=0; }
      if(a == 4) { continue; }
      if(a == 5) { dx=1; dy=0; }
      if(a == 6) { dx=-1; dy=1; }
      if(a == 7) { dx=0; dy=1; }
      if(a == 8) { dx=1; dy=1; }

      for(int i=0;i<inst->_nd;i++)
      {
        vector<Point> trajgt = inst->_trajgt[i];

        for(int b=0;b<(int)trajgt.size()-1;b++)
        {
          Point x_cur  = trajgt[b];
          Point x_next = trajgt[b+1];

          float total_exp = 0;
          bool include = true;

          for(int f=0;f<inst->_nf;f++)
          {
            float feat =
              inst->_featmap[i][f].at<float>(x_cur) -
              inst->_featmap[t][f].at<float>(y,x);
            total_exp += pow(feat / inst->_hf, 2.0);

            include &=
              abs(inst->_featmap[t][f].at<float>(y+dy, x+dx) -
                  inst->_featmap[i][f].at<float>(x_next)) < inst->_a_binwidth;
          }

          numerator += include ? exp(-total_exp / 2.0) : 0;
          denominator += exp(-total_exp / 2.0);
        }
      }

      int index = x + y * inst->_size.width + t * (inst->_size.width * inst->_size.height);
      probs.at<Vec9f>(index)[a] = numerator / denominator;
    }
  }

  if (inst->VERBOSE) printf("  Estimated at row %d on trajectory %d\n", y, t);
}

void CCP::estimatePolicyPointSubsample(CCP *inst, void *args)
{
  par_arg *arg = (par_arg *)args;
  Mat probs = *(Mat *)(arg->probs);
  int y = arg->y;
  int t = arg->t;

  for(int x=0;x<inst->_size.width;x+=1)
  {
    for(int a=0;a<inst->_na;a++)
    {
      float numerator = 0.0;
      float denominator = 0.0;

      int dx,dy;

      if(a == 0) { dx=-1; dy=-1; }
      if(a == 1) { dx=0; dy=-1; }
      if(a == 2) { dx=1; dy=-1; }
      if(a == 3) { dx=-1; dy=0; }
      if(a == 4) { printf("Cannot let _a0 be no movement\n"); exit(0); }
      if(a == 5) { dx=1; dy=0; }
      if(a == 6) { dx=-1; dy=1; }
      if(a == 7) { dx=0; dy=1; }
      if(a == 8) { dx=1; dy=1; }

      for(int i=0;i<inst->_samp_size;i++)
      {
        vector<Point> trajgt = inst->_trajgt[inst->_currentSampleTraj[i]];
        int b = inst->_currentSamplePoint[i];

        Point x_cur  = trajgt[b];
        Point x_next = trajgt[b == trajgt.size() - 1 ? t-1 : t+1];

        float total_exp = 0;
        bool include = true;

        for(int f=0;f<inst->_nf;f++)
        {
          int trajInd = inst->_currentSampleTraj[i];
          float feat =
            inst->_featmap[trajInd][f].at<float>(x_cur) -
            inst->_featmap[t][f].at<float>(y,x);
          total_exp += pow(feat / inst->_hf, 2.0);

          include &=
            abs(inst->_featmap[t][f].at<float>(y+dy, x+dx) -
                inst->_featmap[i][f].at<float>(x_next)) < inst->_a_binwidth;
        }

        numerator += include ? exp(-total_exp / 2.0) : 0;
        denominator += exp(-total_exp / 2.0);
      }

      int index = x + y * inst->_size.width + t * (inst->_size.width * inst->_size.height);
      probs.at<Vec9f>(index)[a] = numerator / denominator;
    }
  }

  if (inst->VERBOSE) printf("  Estimated at row %d\n", y);
}

void CCP::estimatePolicy(bool subsample)
{
  cout << "\nEstimatePolicy()\n";

  const size_t NUM_THREADS = 16;

  _probs = Mat::zeros(_size.height * _size.width * _nd, 1, CV_32FC(9));
  vector<thread> threads;
  par_arg args[NUM_THREADS];

  for(int t=0;t<_nd;t++)
  {
    for(int y=0;y<_size.height;y+=NUM_THREADS)
    {
      for (int i=0;i<NUM_THREADS;i++)
      {
        if (y+i < _size.height)
        {
          args[i].y = y+i;
          args[i].probs = &_probs;
          args[i].t = t;

          if (subsample)
          {
            threads.push_back(thread(estimatePolicyPointSubsample, this, &(args[i])));
          } else {
            threads.push_back(thread(estimatePolicyPoint, this, &(args[i])));
          }
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

      if(VISUALIZE && !subsample)
      {
        vector<Mat> actionProbs(9);
        split(_probs, actionProbs);
        Mat act = actionProbs[0].rowRange((_size.width * _size.height) * t,(_size.width * _size.height) * (t + 1)).reshape(0, _size.height);

        Mat dst;
        colormap(act,dst);
        addWeighted(_image[t],0.5,dst,0.5,0,dst);
        imshow("Action " + to_string(0),dst);
        waitKey(1);
      }
    }
  }

  threshold(_probs, _probs, FLT_MIN, 0, THRESH_TOZERO);

  vector<Mat> actionProbs(9);
  split(_probs, actionProbs);

	if(VISUALIZE && !subsample)
	{
    for(int a=0;a<_na;a++)
    {
      Mat dst;
      colormap(actionProbs[a].rowRange(0,_size.width*_size.height).reshape(0, _size.height),dst);
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
  for(int t=0;t<_nd;t++)
  {
    for(int y=0;y<_size.height;y++)
    {
      for(int x=0;x<_size.width;x++)
      {
        for(int a=0;a<_na;a++)
        {
          int index = x + y * _size.width + t * (_size.width * _size.height);
          fs << to_string(_probs.at<Vec9f>(index)[a]) << "\t";
        }
        fs << endl;
      }
    }
  }
  fs.close();
}

void CCP::readPolicy(string input_filename)
{
  cout << "\nReadPolicy()\n";

  ifstream fs;
  fs.open(input_filename.c_str());
  if(!fs.is_open()){cout << "ERROR: Opening: " << input_filename << endl;exit(1);}

  string str;
  Mat p(_nd * _size.height * _size.width, 1, CV_32FC(9));

  int x = 0;

  while(getline(fs,str))
  {
    int a = 0;
    size_t l = str.length();

    size_t i = 0;
    while(a < _na)
    {
      p.at<Vec9f>(x)[a] = stof(str, &i);
      str = str.substr(i);

      a++;
    }
    x++;
  }

  _probs = p.clone();

  vector<Mat> actionProbs(9);
  split(_probs, actionProbs);

	if(VISUALIZE)
	{
    for(int a=0;a<_na;a++)
    {
      Mat dst;
      colormap(actionProbs[a].rowRange(0,_size.width*_size.height).reshape(0, _size.height),dst);
      addWeighted(_image[0],0.5,dst,0.5,0,dst);
      imshow("Action " + to_string(a),dst);
      waitKey(0);
    }
	}
}

void CCP::estimateGamma()
{
  cout << "\nEstimateGamma()\n";

  vector<float> gamma(0);

  vector<float> a0f;

  for(int f=0;f<_nf;f++)
  {
    a0f.push_back(_featmap[_a0_t][f].at<float>(_a0));
  }

  Mat a0(a0f, true);

  for(int t=0;t<_nd;t++)
  {
    for(int y=0;y<_size.height;y++)
    {
      for(int x=0;x<_size.width;x++)
      {
        float val = 0.0;

        int index = x + y * _size.width + t * (_size.width * _size.height);
        int min_a = 0;
        float error = FLT_MAX;

        for(int a=0;a<_na;a++)
        {
          vector<float> afv;
          int dx,dy;

          if(a == 0) { dx=-1; dy=-1; }
          if(a == 1) { dx=0; dy=-1; }
          if(a == 2) { dx=1; dy=-1; }
          if(a == 3) { dx=-1; dy=0; }
          if(a == 4) { continue; }
          if(a == 5) { dx=1; dy=0; }
          if(a == 6) { dx=-1; dy=1; }
          if(a == 7) { dx=0; dy=1; }
          if(a == 8) { dx=1; dy=1; }

          for(int f=0;f<_nf;f++)
          {
            afv.push_back(_featmap[t][f].at<float>(y+dy,x+dx));
          }

          Mat af(afv, true);

          float tmperror = sum((af - a0).mul(af - a0))[0];

          if (tmperror < error) {
            error = tmperror;
            min_a = a;
          }
        }

        int dx,dy;
        bool a0_exists = true;

        if(min_a == 0) { dx=-1; dy=-1; }
        if(min_a == 1) { dx=0; dy=-1; }
        if(min_a == 2) { dx=1; dy=-1; }
        if(min_a == 3) { dx=-1; dy=0; }
        if(min_a == 4) { printf("Cannot let _a0 be no movement\n"); exit(0); }
        if(min_a == 5) { dx=1; dy=0; }
        if(min_a == 6) { dx=-1; dy=1; }
        if(min_a == 7) { dx=0; dy=1; }
        if(min_a == 8) { dx=1; dy=1; }

        for(int f=0;f<_nf;f++)
        {
          //a0_exists = a0_exists && (abs(_featmap[t][f].at<float>(y+dy,x+dx) -
          //        _featmap[_a0_t][f].at<float>(_a0)) <= _gamma_binwidth);
        }

        if (a0_exists) {
          for(int a=0;a<_na;a++)
          {
            float n = _probs.at<Vec9f>(index)[a];
            float d = _probs.at<Vec9f>(index)[min_a];

            if(d != 0)
            {
              val += log(1.0 + n/d);
            } else {
              val = nanf("");
            }
          }
        } else {
          val = nanf("");
        }

        gamma.push_back(val);
      }
    }
  }

  _gamma = Mat(gamma, true); // _size.height * _size.width * _nd x 1
  Mat vis_gamma;
  _gamma.copyTo(vis_gamma);

  if(VISUALIZE)
  {
    vis_gamma = vis_gamma.rowRange(0,_size.width*_size.height).reshape(0,_size.height);

    Mat dst;
    colormap(vis_gamma,dst);
    addWeighted(_image[0],0.5,dst,0.5,0,dst);
    imshow("Gamma Vector",dst);
    waitKey(0);
  }
}

void CCP::estimateTransitionMatrix()
{
  cout << "\nEstimateTransitionMatrix()\n";

  int ns = _size.height * _size.width * _nd;
  int size[] = {ns,ns};

  _T = SparseMat(2,size,CV_32FC1);

  vector<float> a0f;

  for(int f=0;f<_nf;f++)
  {
    a0f.push_back(_featmap[_a0_t][f].at<float>(_a0));
  }

  Mat a0(a0f, true);

  for(int t=0;t<_nd;t++)
  {
    for(int x=0;x<_size.width;x++)
    {
      for(int y=0;y<_size.height;y++)
      {
        int index = x + y * _size.width + t * (_size.width * _size.height);

        int min_a = 0;
        float error = FLT_MAX;

        for(int a=0;a<_na;a++)
        {
          vector<float> afv;
          int dx,dy;

          if(a == 0) { dx=-1; dy=-1; }
          if(a == 1) { dx=0; dy=-1; }
          if(a == 2) { dx=1; dy=-1; }
          if(a == 3) { dx=-1; dy=0; }
          if(a == 4) { continue; }
          if(a == 5) { dx=1; dy=0; }
          if(a == 6) { dx=-1; dy=1; }
          if(a == 7) { dx=0; dy=1; }
          if(a == 8) { dx=1; dy=1; }

          for(int f=0;f<_nf;f++)
          {
            afv.push_back(_featmap[t][f].at<float>(y+dy,x+dx));
          }

          Mat af(afv, true);

          float tmperror = sum((af - a0).mul(af - a0))[0];

          if (tmperror < error) {
            error = tmperror;
            min_a = a;
          }
        }

        int dx,dy;

        if(min_a == 0) { dx=-1; dy=-1; }
        if(min_a == 1) { dx=0; dy=-1; }
        if(min_a == 2) { dx=1; dy=-1; }
        if(min_a == 3) { dx=-1; dy=0; }
        if(min_a == 4) { printf("Cannot let _a0 be no movement\n"); exit(0); }
        if(min_a == 5) { dx=1; dy=0; }
        if(min_a == 6) { dx=-1; dy=1; }
        if(min_a == 7) { dx=0; dy=1; }
        if(min_a == 8) { dx=1; dy=1; }

        int index_n = index + dx + (_size.width * dy);

        if (index_n < ns && index_n > 0)
        {
          _T.ref<float>(index,index_n) = 1.0;
        }
      }
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

  int ns = _size.height * _size.width * _nd;
  int size[] = {ns,ns};

  SparseMat BT;
  _T.copyTo(BT);

  for(SparseMatIterator_<float> x = BT.begin<float>(); x != BT.end<float>(); x++)
  {
    *x = (*x) * _B;
  }

  Mat V0 = Mat(BT.size(0), 1, CV_32FC1, 0.0);
  Mat V0_original = V0.clone();

  multiply(BT, V0_original + _gamma, &V0);

  int index = 0;

  while(!equal(V0_original,V0,ns,_E))
  {
    V0.copyTo(V0_original);
    multiply(BT, V0_original + _gamma, &V0);

    if (VERBOSE && index % 10 == 0)
      printf("  %d value iterations complete\n", index);

    index++;
  }

  if (VERBOSE) printf("  Converged in %d iterations\n", index);

  _V0 = V0.clone();

  if (VISUALIZE)
  {
    Mat dst;
    colormap(_V0.rowRange(0,_size.width*_size.height).reshape(0,_size.height),dst);
    addWeighted(_image[0],0.5,dst,0.5,0,dst);
    imshow("Value Function for 0-Reward Action",dst);
    waitKey(0);
  }
}

void CCP::estimateValueFunction()
{
  cout << "\nEstimateValueFunction()\n";

  _V = Mat(_size.width * _size.height * _nd, 1, CV_32FC(9));

  vector<float> a0f;

  for(int f=0;f<_nf;f++)
  {
    a0f.push_back(_featmap[_a0_t][f].at<float>(_a0));
  }

  Mat a0(a0f, true);

  for(int t=0;t<_nd;t++)
  {
    for(int y=0;y<_size.height;y++)
    {
      for(int x=0;x<_size.width;x++)
      {
        int min_a = 0;
        float error = FLT_MAX;

        for(int a=0;a<_na;a++)
        {
          vector<float> afv;
          int dx,dy;

          if(a == 0) { dx=-1; dy=-1; }
          if(a == 1) { dx=0; dy=-1; }
          if(a == 2) { dx=1; dy=-1; }
          if(a == 3) { dx=-1; dy=0; }
          if(a == 4) { continue; }
          if(a == 5) { dx=1; dy=0; }
          if(a == 6) { dx=-1; dy=1; }
          if(a == 7) { dx=0; dy=1; }
          if(a == 8) { dx=1; dy=1; }

          for(int f=0;f<_nf;f++)
          {
            afv.push_back(_featmap[t][f].at<float>(y+dy,x+dx));
          }

          Mat af(afv, true);

          float tmperror = sum((af - a0).mul(af - a0))[0];

          if (tmperror < error) {
            error = tmperror;
            min_a = a;
          }
        }

        int dx,dy;

        if(min_a == 0) { dx=-1; dy=-1; }
        if(min_a == 1) { dx=0; dy=-1; }
        if(min_a == 2) { dx=1; dy=-1; }
        if(min_a == 3) { dx=-1; dy=0; }
        if(min_a == 4) { printf("Cannot let _a0 be no movement\n"); exit(0); }
        if(min_a == 5) { dx=1; dy=0; }
        if(min_a == 6) { dx=-1; dy=1; }
        if(min_a == 7) { dx=0; dy=1; }
        if(min_a == 8) { dx=1; dy=1; }

        for(int a=0;a<_na;a++)
        {
          int index = x + y * _size.width + t * _size.height * _size.width;

          float n = _probs.at<Vec9f>(index)[a];
          float d = _probs.at<Vec9f>(index)[min_a];

          if(d != 0 && n != 0)
          {
            _V.at<Vec9f>(index)[a] = log(n/d) + _V0.at<float>(index);
          } else {
            _V.at<Vec9f>(index)[a] = nanf("");
          }
        }
      }
    }
  }
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
      colormap(value[a].rowRange(0,_size.width*_size.height).reshape(0,_size.height),dst);
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
  for(int t=0;t<_nd;t++)
  {
    for(int y=0;y<_size.height;y++)
    {
      for(int x=0;x<_size.width;x++)
      {
        for(int a=0;a<_na;a++)
        {
          int index = x + y * _size.width + t * _size.width * _size.height;
          fs << to_string(_V.at<Vec9f>(index)[a]) << "\t";
        }
        fs << endl;
      }
    }
  }
  fs.close();
}

void CCP::readValueFunction(string input_filename)
{
  cout << "\nReadValueFunction()\n";

  ifstream fs;
  fs.open(input_filename.c_str());
  if(!fs.is_open()){cout << "ERROR: Opening: " << input_filename << endl;exit(1);}

  string str;
  Mat p(1, _size.height * _size.width * _nd, CV_32FC(9));

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

  _V = p.clone();
}

void CCP::estimateRewardFunction()
{
  cout << "\nEstimateRewardFunction()\n";

  Mat R(_size.width * _size.height * _nd, 1, CV_32FC(9));

  for(int t=0;t<_nd;t++)
  {
    for(int y=0;y<_size.height;y++)
    {
      for(int x=0;x<_size.width;x++)
      {
        for(int a=0;a<_na;a++)
        {
          int index = x + _size.width * y + t * _size.width * _size.height;
          float V_ax = _V.at<Vec9f>(index)[a];

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

          int index_n = x_n + _size.width * y_n + t * _size.width * _size.height;
          float r = V_ax - _B * (_V0.at<float>(index_n) + _gamma.at<float>(index_n));

          R.at<Vec9f>(index)[a] = r;
        }
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
      colormap(value[a].rowRange(0,_size.width*_size.height).reshape(0,_size.height),dst);
      addWeighted(_image[0],0.5,dst,0.5,0,dst);
      imshow("Action " + to_string(a),dst);
      waitKey(0);
    }
  }
}

void CCP::saveRewardFunction(string output_filename)
{
  cout << "\nSaveRewardFunction()\n";

  ofstream fs(output_filename.c_str());
  if(!fs.is_open()) cout << "ERROR: Writing: " << output_filename << endl;
  for(int t=0;t<_nd;t++)
  {
    for(int y=0;y<_size.height;y++)
    {
      for(int x=0;x<_size.width;x++)
      {
        for(int a=0;a<_na;a++)
        {
          int index = x + _size.width * y + t * _size.width * _size.height;
          fs << to_string(_R.at<Vec9f>(index)[a]) << "\t";
        }
        fs << endl;
      }
    }
  }
  fs.close();
}

void CCP::saveTrueRewardFunction()
{
  _R_true = _R.clone();
}

void CCP::readRewardFunction(string input_filename)
{
  cout << "\nReadRewardFunction()\n";

  ifstream fs;
  fs.open(input_filename.c_str());
  if(!fs.is_open()){cout << "ERROR: Opening: " << input_filename << endl;exit(1);}

  string str;
  Mat p(_size.height * _size.width * _nd, 1, CV_32FC(9));

  int x = 0;

  while(getline(fs,str))
  {
    int a = 0;
    size_t l = str.length();

    size_t i = 0;
    while(a < 9)
    {
      p.at<Vec9f>(x)[a] = stof(str, &i);
      str = str.substr(i);

      a++;
    }
    x++;
  }

  _R = p.clone();
}

void CCP::setUpRandomization()
{
  cout << "\nSetUpRandomization()\n";
  total_pairs = 0;

  _currentSampleTraj = vector<int>();
  _currentSamplePoint = vector<int>();

  for (int n=0; n<_nd; n++)
  {
    total_pairs += (int)_trajgt[n].size();
  }

  point_pair current;

  for (int i=0; i<_samp_size; i++)
  {
    getRandomPair(&current);
    _currentSampleTraj.push_back(current.trajectory);
    _currentSamplePoint.push_back(current.data_point);
  }
}

void CCP::getRandomPair(point_pair *result)
{
  int index = rand() % total_pairs;

  for (int n=0; n<_nd; n++)
  {
    if (index < (int)_trajgt[n].size() && index >= 0) {
      result->trajectory = n;
      result->data_point = index;
      return;
    }

    index -= (int)_trajgt[n].size();
  }
}
