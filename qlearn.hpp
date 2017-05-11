/*
 *  qlearn.h
 *  Q-Learning for CCP
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
#include <thread>
#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;

typedef Vec<float, 9> Vec9f;

class QLearn {
  public:
    QLearn(){}
    ~QLearn(){}

    void initialize();

    void readRewardFunction(string input_filename);

    void setGoal();
    void setStart();

    void qLearn();

    cv::Mat _R;                               // reward function
    cv::Mat _Q;                               // Q
    cv::Mat _image;                           // image for overlay

    int _nrow;                                // number of rows in state space
    int _ncol;                                // number of cols in state space
    int _na;                                  // number of actions
    int _start_state;                         // start state
    int _goal_state;                          // goal state
    float _gamma;                             // discount
    float _alpha;                             // learning rate
    float _epsilon;                           // discovery rate

    bool VERBOSE = true;                      // verbose
    bool VISUALIZE = true;                    // visualize
};
