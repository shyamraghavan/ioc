/*
 *  transport.h
 *  Transport to separate scene
 *
 *  Created by Shyam Raghavan on 05/06/17.
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

class Transfer {
  public:
    Transfer(){}
    ~Transfer(){}

    void initialize();
    void loadPrevBasenames(string input_filename);
    void loadPrevReward(string input_filename);
    void loadPrevFeatMap(string input_file_prefix);
    void reshapePrevFeatMap();

    void loadBasenames(string input_filename);

    int _prev_nd;
    int _prev_na;
    cv::Size _prev_size;

    std::vector<string> _prev_basenames;
    cv::Mat _prev_R;
    std::vector<std::vector<cv::Mat>> _prev_featmap;
    cv::Mat _prev_feats;

    int _nf;
    int _nd;
    int _na;
    cv::Size _size;

    std::vector<string> _basenames;
    std::vector<std::vector<cv::Mat>> _featmap;

    bool VERBOSE = true;
};
