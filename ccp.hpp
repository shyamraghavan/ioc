/*
 *  ccp.h
 *  IOC_DEMO
 *
 *  Created by Shyam Raghavan on 02/14/17.
 *  Copyright 2017. All rights reserved.
 *

 Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

 THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */
#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;

typedef Vec<float, 9> Vec9f;

class CCP {
  public:
    CCP(){}
    ~CCP(){}

    void initialize	                      ();
    void loadBasenames	                  (string input_filename);
    void loadDemoTraj	                    (string input_file_prefix);
    void loadFeatureMaps                  (string input_file_prefix);
    void loadImages		                    (string input_file_prefix);
    void estimatePolicy                   ();

  private:
    vector < string >				      _basenames;		// file basenames
    vector < vector<cv::Point> >	_trajgt;			// ground truth trajectory
    vector < vector<cv::Point> >	_trajob;			// observed tracker output
    vector < vector<cv::Mat> >		_featmap;			// (physical) feature maps
    vector < cv::Mat >				    _image;				// (physical) feature maps

    vector <cv::Point>				    _end;				  // terminal states
    vector <cv::Point>				    _start;				// start states
    int								            _nd;				  // number of training data
    int								            _nf;				  // number of training data
    int								            _na;				  // number of actions [3x3]
    float                         _h;           // bandwidth
    cv::Size					           	_size;				// current state space size

    bool							            VISUALIZE = true;
    bool							            VERBOSE = true;
};
