/*
 *  trainHIOC.h
 *  HIOC-training-class
 *
 *  Created by Kris Kitani on 8/1/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <math.h>
#include <opencv2/opencv.hpp>
//#include "opencv2/gpu/gpu.hpp"
using namespace std;
using namespace cv;

#define VISUALIZE 0
#define DEBUG 0
#define DISCRETE 1
#define WSIZE 5
#define W_INIT 1.0
#define VISOUT 0


class trainHIOC{
public:
	
	trainHIOC(){};
	~trainHIOC(){};
	
	void init(int cols, int rows);
	int cols;
	int rows;
	
	void setFlags(string action, string exp, int append, string root, float softmax_k);
	int OBSERVATION_ON;
	int FEATURES_ON;
	int APPROACH_CAR;
	int APPEND_RESULTS;
	string FILE_ROOT;
	float SOFTMAXVAL;
	int MARKOV;
	//string exptype;
	
	void setFiles(string actionclass, vector<string> &trainList);
	
	void setTrainFiles(string actionclass, vector<string> &trainList, vector<string> &masterList);
	void setTestFiles(vector<string> &testList);
	
	void runTrainHIOC(int max_iterations, float stepsize);
	void runTestHIOC();
	
	void loadTrajectory(int d, string fileid);				// load from file for training
	void loadTrajectory(int d, string fileid, int test);	// load file, flag for training or testing (leading zero removal)
	void scaleTimeTrajectory(int d);	// optional temporal scaling
	void scaleSpaceTrajectory(int d);
	void loadFeatureMaps(int d,string fileid);
	void updateObsFeatureMaps(int i, int t, int d, string fileid); // observation features
	void computeEmpiricalStatistics(int d);
	void updateActionSpace();
	void getFeatures(Point pt, vector<float> &feats, int d);
	
	int nf;				// Number of features
	int kernelsize;		// Odd number 3-9 
	int na;				// Number of hidden actions
	int ns;				// Number of hidden actions
	int *ba;			// Bins in action dimension i
	int *bs;			// Bins in action dimension i
	float *amx;			// Max values of actions (used for quantization)
	float *amn;			// Min values of actions
	float *smx;			// Max values of actions (used for quantization)
	float *smn;			// Min values of actions
	float lambda;		// Gradient decent step size
	
	vector<string> masterList;
	vector<string> trainList;
	vector<string> testList;
	string trainid;
	
	string actionclass;
	ofstream logl;
	ofstream fll[5];
	ofstream fllm;
	ofstream ftest;
	
	vector< vector<Point2f> > obt;			// observed trajectory
	vector< vector<Point2f> > trt;			// true trajectory
	vector< vector<float> > empirical_f;	// empirical feature count
	vector<float> expected_f;				// expected feature count
	vector<float> empirical_mean_f;
	vector<float> expected_mean_f;
	vector<string> featurelabel;
	vector< vector<Mat> > fmap;
	vector<vector<float> > pax;				// Policy
	vector<float> w;
	
	int ValueIteration(int iteration, int d, string fileid, int goal);
	int ValueIteration(int d, string fileid, vector<int> goals, vector<float> vals);
	int ValueIteration(int d, string fileid, vector<int> goals, vector<float> vals, Mat &logZ);
	int ValueIteration(int d, string fileid, int goal, vector<int> goals,vector<float> vals, Mat &logZ);
	float ComputeLogLikelihood(int d, int tau, float *val);
	ofstream fcl;
	
	double ComputeObservedLogLikelihood(int d, int tau);
	
	void jetmap(Mat src, Mat &dst);
	void jetmapAbs(Mat src, Mat &dst);
	void jetmapProb(Mat src, Mat &dst);
	void actionIndexToValue(int action_index, int* action_d);
	void stateIndexToValue(int state_index, int* state_d);
	int getActionIndex(int *action_d);
	int getStateIndex (int *state_d);
	void loadParameters();
	void saveParameters();
	void runTestHIOCgoals();
	
	void setGoalIndicies(vector<int>&goals);	// heuristic manual goals
	void computeGoals(vector <int> &goals);		// automatic goals
	string action;
	
	int start_f;
	
	float * emp_m_f;
	float * exp_m_f;
	int fn;
	int ds;							// state dimensions
	int da;							// action dimensions
	//int nt;
	Mat H;							// 3D ground plane homography 
	int sample_width;
	
	int getNextState(int x,int a);
	int isValidState(int *state_d);
	int ComputeExactExpectations(int d, string fileid, int st, int s0, vector <int> goals);
	int ComputeExactExpectations(int tau, int d, string fileid, int goal, int st);
	int ComputeExactExpectations(int d,string fileid, int x_goal, int st, int s0, vector <int> goals);
	VideoWriter expect_avi_train;
	VideoWriter expect_avi_test;
	
	VideoWriter avi_cum;
	
	void sampleTrajectories(int d, string fileid);
	void sampleTrajectories(int d,string fileid, vector <int> goals);
	ofstream fdist;
	
	

	void trainGoalClassifier();
	void computeGoals( vector<vector <int> > &goal);

	//==================//
	
	void  runTrainMarkovFeat();
	void  computeParametersMarkovFeat();
	void  runTestMarkovFeat();
	float computeLogLikelihoodMarkovFeat(int d);
	void  setActionSpace(int ks);
	Mat mP;
	
	void  runTrainMarkovMotion();
	void  runTestMarkovMotion();
	void  computeParametersMarkovMotion();
	float computeLogLikelihoodMarkovMotion(int d);
	Mat mPolicy;
	
private:
	
};


