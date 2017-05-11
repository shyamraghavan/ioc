/*
 *  qlearn.cpp
 *  QLearning
 *
 *  Created by Shyam Raghavan on 02/14/17.
 *  Copyright 2017. All rights reserved.
 *

 Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

 THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include "qlearn.hpp"

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

void QLearn::initialize()
{
  _nrow = 216;
  _ncol = 384;
  _na = 9;

  _gamma = 0.5;
  _alpha = 0.5;

  _epsilon = 0.99;

  _Q = Mat::zeros(_nrow*_ncol, 1, CV_32FC(9));

  string input_filename = "./ioc_demo/walk_imag/VIRAT_S_000005_12340_13370_2_topdown.jpg";
  Mat im = imread(input_filename);
  if(!im.data){cout << "ERROR: Opening:" << input_filename << endl; exit(1);}
  resize(im,im,Size(_ncol, _nrow));
  _image = im.clone();
}

void QLearn::readRewardFunction(string input_filename)
{
  cout << "\nReadRewardFunction()\n";

  ifstream fs;
  fs.open(input_filename.c_str());
  if(!fs.is_open()){cout << "ERROR: Opening: " << input_filename << endl;exit(1);}

  string str;
  Mat p(_nrow*_ncol,1,CV_32FC(9));

  int x = 0;

  while(getline(fs,str) && x < _nrow*_ncol)
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

void QLearn::setGoal()
{
  cout << "\nSetGoal()\n";

  _goal_state = rand() % (_nrow * _ncol);
  for (int a=0;a<_na;a++)
  {
    _R.at<Vec9f>(_goal_state)[a] = FLT_MAX;
  }

  printf("  Goal State: %d %d\n", _goal_state%_ncol, _goal_state/_ncol);
}

void QLearn::setStart()
{
  cout << "\nSetStart()\n";

  _start_state = rand() % (_nrow * _ncol);

  printf("  Start State: %d %d\n", _start_state%_ncol, _start_state/_ncol);
}

void QLearn::qLearn()
{
  cout << "\nQLearn()\n";

  int state = _start_state;
  int iterations = 0;

  bool converged = false;

  while (!converged)
  {
    Point p;
    double max_q;
    minMaxLoc(_Q.at<Vec9f>(state), NULL, &max_q, NULL, &p);

    int best_action;
    int dx;
    int dy;

    float choice = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
    if (choice < _epsilon || max_q < 0.0000001) {
      dx = rand() % 3 - 1;
      dy = rand() % 3 - 1;;

      if( dx==-1 && dy==-1 ) best_action = 0;
      if( dx== 0 && dy==-1 ) best_action = 1;
      if( dx== 1 && dy==-1 ) best_action = 2;

      if( dx==-1 && dy== 0 ) best_action = 3;
      if( dx== 0 && dy== 0 ) best_action = 4;
      if( dx== 1 && dy== 0 ) best_action = 5;

      if( dx==-1 && dy== 1 ) best_action = 6;
      if( dx== 0 && dy== 1 ) best_action = 7;
      if( dx== 1 && dy== 1 ) best_action = 8;
    } else {
      best_action = p.y;

      if(best_action == 0) { dx=-1; dy=-1; }
      if(best_action == 1) { dx=0; dy=-1; }
      if(best_action == 2) { dx=1; dy=-1; }

      if(best_action == 3) { dx=-1; dy=0; }
      if(best_action == 4) { dx=0; dy=0; }
      if(best_action == 5) { dx=1; dy=0; }

      if(best_action == 6) { dx=-1; dy=1; }
      if(best_action == 7) { dx=0; dy=1; }
      if(best_action == 8) { dx=1; dy=1; }
    }

    int state_p = state + dx + _ncol * dy;

    double max_q_p;
    minMaxLoc(_Q.at<Vec9f>(state), NULL, &max_q_p);

    float r = _R.at<Vec9f>(state)[best_action];

    _Q.at<Vec9f>(state)[best_action] += _alpha * (r + _gamma * (max_q_p - max_q));

    state = state_p;
    converged = state == _goal_state || iterations > 2500;

    iterations += 1;
    if (iterations % 10 == 0)
    {
      printf("  %d iterations complete\n", iterations);
    }

    if (VISUALIZE)
    {
      Mat V;

      Mat act(_nrow*_ncol, 1, CV_32FC1, 0.0);

      for(int i=0;i<_nrow*_ncol;i++)
      {
        double p;
        minMaxLoc(_Q.at<Vec9f>(i), NULL, &p, NULL);

        act.at<float>(i) = (float)p;
      }

      act = act.reshape(0, _nrow);

      Mat dst;
      colormap(act,dst);
      addWeighted(_image,0.5,dst,0.5,0,dst);
      imshow("Value Function",dst);
      waitKey(1);
    }
  }
}
