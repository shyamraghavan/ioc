/*
 *  trainHIOC.cpp
 *  runHIOC
 *
 *  Created by Kris Kitani on 11/2/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */

#include "trainHIOC.hpp"

void trainHIOC::setFlags(string action, string exp, int append, string root, float softmax_k, string job, int multigoal, int vis){
	
	this->action = action;
	this->job = job;
	VISUALIZE = vis;
	EXPTYPE = exp;
	FEATURES_ON = 1;
	OBSERVATION_ON = 1;	
	APPROACH_CAR = 0;
	APPEND_RESULTS = 0;
	SOFTMAXVAL = 1.0;
	MARKOV = 0;
	
	if(multigoal==1) MULTI_GOAL_ON = 1;
	else MULTI_GOAL_ON = 0;
	
	if(exp == "feat"){
		FEATURES_ON = 1;
		OBSERVATION_ON = 0;
	}
	else if(exp=="const"){
		FEATURES_ON = 0;
		OBSERVATION_ON = 0;
	}
	
	if(softmax_k>0) SOFTMAXVAL = softmax_k;
	
	FILE_ROOT = root;
	
	if(action.find("approach")!=string::npos) APPROACH_CAR = 1;
	
	if(append!=0) APPEND_RESULTS = 1;
	
	if(exp=="markovfeat" || exp=="markovmotion" || exp=="markovfull") MARKOV = 1;
	
	if(exp=="markovfeat"){
		FEATURES_ON = 1;
		OBSERVATION_ON = 0;
	}
	else if(exp=="markovmotion"){
		FEATURES_ON = 0;
		OBSERVATION_ON = 0;
	}
	else if(exp=="markovfull"){
		FEATURES_ON = 1;
		OBSERVATION_ON = 1;
	}
	
	
	cout << "FEATURES_ON: " << FEATURES_ON << endl;
	cout << "OBSERVATIONS_ON: " << OBSERVATION_ON << endl;
	cout << "APPROACH_CAR: " << APPROACH_CAR << endl;
	cout << "APPEND RESULTS: " << APPEND_RESULTS << endl;
	cout << "FILE ROOT: " << FILE_ROOT << endl;
	cout << "SOFTMAXVAL:" << SOFTMAXVAL << endl;
	cout << "MARKOV:" << MARKOV << endl;
	cout << "MULTI_GOAL:" << MULTI_GOAL_ON << endl;
}

Mat trainHIOC::getH(){
	stringstream sf;
	
	if(job=="transfer"){ // invert
		if(action.find("2") != string::npos) sf << FILE_ROOT <<"geometry/virat-homography.txt";
		else sf << FILE_ROOT <<"geometry/virat2_homography.txt";
	}
	else{ // train or test, no inversion
		if(action.find("2") != string::npos) sf << FILE_ROOT <<"geometry/virat2_homography.txt";
		else sf << FILE_ROOT <<"geometry/virat-homography.txt";
	}
	
	//if(action.find("2") != string::npos && job!="transfer") sf << FILE_ROOT <<"geometry/virat2_homography.txt";
	//else if(action.find("2") != string::npos && job=="transfer") sf << FILE_ROOT <<"geometry/virat-homography.txt";
	//else{
	//	cout << "ERROR: action:" << action << " job:" << job << ", this combination not accounted for.\n";
	//	exit(1);
	//}	
	//if(job=="transfer" || action.find("2") != string::npos) sf << FILE_ROOT <<"geometry/virat2_homography.txt";
	//else sf << FILE_ROOT <<"geometry/virat-homography.txt";
	
	
	ifstream fp(sf.str().c_str());
	float d[16];
	int k=0;
	while(fp>>d[k++]);	
	Point2f pts[8];
	for(int i=0;i<8;i++){
		pts[i].x = d[i*2+0];
		pts[i].y = d[i*2+1];
	}

	for(int i=0;i<8;i++){
		pts[i].x *= cols;
		pts[i].y *= rows;
	}
	return getPerspectiveTransform(&pts[0],&pts[4]);
}

void trainHIOC::trainGoalClassifier(){
	
	if(DEBUG) cout << "Method: trainGoalClassifier" << endl;
	
	Mat svmTrainData;
	Mat svmTrainLabel;
	CvSVMParams params;
	CvSVM svm;
	int wsize = WSIZE;
	
	Point p[2];
	Rect r[2];
	Mat v = Mat::ones(1,1,CV_32FC1);
	
	cout << "==== TRAIN GOAL CLASSIFIER ====" << endl;
	
	for(int d=0;d<(int)trt.size();d++){
		
		for(int t=0;t<1;t++){
			p[0] = trt[d][(int)trt[d].size()-1-t];
			r[0] = Rect(p[0].x-wsize*0.5,p[0].y-wsize*0.5,wsize,wsize);
			if(r[0].x<0) continue;
			if(r[0].y<0) continue;
			
			Mat pos;
			for(int f=0;f<(nf-3);f++){
				cout << w[f] << endl;
				Mat dat = v* mean(w[f] * fmap[d][f](r[0])).val[0];
				if(!pos.data) dat.copyTo(pos);
				else pos.push_back(dat);
			}
			cout << "pos=" << pos << endl;
			pos = pos.reshape(1,1);
			if(!svmTrainData.data) pos.copyTo(svmTrainData);
			else svmTrainData.push_back(pos);
			v = 1;
			if(!svmTrainLabel.data) v.copyTo(svmTrainLabel);
			else svmTrainLabel.push_back(v);
		}
		
		
		for(int t=0;t<1;t++){
			p[1] = trt[d][t];
			
			cout <<"p=" << p[1] << endl;;
			r[1] = Rect(p[1].x-wsize*0.5,p[1].y-wsize*0.5,wsize,wsize);
			if(r[1].x<0) continue;
			if(r[1].y<0) continue;
			Mat neg;
			for(int f=0;f<(nf-3);f++){
				//Mat dat = w[f] * fmap[d][f](r[1]);
				Mat dat = v* mean(w[f] * fmap[d][f](r[1])).val[0];
				if(!neg.data) dat.copyTo(neg);
				else neg.push_back(dat);
			}
			cout << neg << endl;
			neg = neg.reshape(1,1);
			svmTrainData.push_back(neg);
			v = -1;
			cout << "lab " << svmTrainLabel.rows << " " << svmTrainLabel.cols << endl;
			svmTrainLabel.push_back(v);
		}
		
	}
	
	params.term_crit=cvTermCriteria(CV_TERMCRIT_ITER,100,0.000001);
	params.kernel_type=CvSVM::LINEAR;
	params.svm_type=CvSVM::EPS_SVR;
	params.p = 1;
	
	CvMat trainData = svmTrainData;
	CvMat trainLabel = svmTrainLabel;
	cout << "lab " << svmTrainLabel.rows << " " << svmTrainLabel.cols << endl;
	cout << "Number of examples:" << svmTrainData.rows << endl;
	
	svm.train_auto(&trainData,&trainLabel,cv::Mat(),cv::Mat(),params,10);
	
	stringstream ss;
	ss << FILE_ROOT << actionclass << "-goal-svm.xml";
	cout << "saving:" << ss.str().c_str() << endl;
	svm.save(ss.str().c_str());
	cout << "sucessful save." << endl;
	
	waitKey(0);
	
	
	
	cout << " test on train data " << endl;
	for(int d=0;d<(int)trt.size();d++){
		//goal.push_back(vector<int>(0));
		Mat goalmap = fmap[d][0] * 0;
		for(int c=wsize;c<fmap[d][0].cols-wsize;c++){
			for(int r=wsize;r<fmap[d][0].rows-wsize;r++){
				Point p = Point(c,r);
				Rect rec = Rect(p.x-wsize*0.5,p.y-wsize*0.5,wsize,wsize);
				Mat sample;
				for(int f=0;f<(nf-3);f++){
					//Mat dat = w[f]*fmap[d][f](rec) * 1.0;
					Mat dat = v* mean(w[f] * fmap[d][f](rec)).val[0];
					if(!sample.data) dat.copyTo(sample);
					else sample.push_back(dat);
					
				}
				sample = sample.reshape(1,1);
				CvMat _sample = sample;
				double score = svm.predict(&_sample);
				//if(score < 0 || score < 0.5) score =0;
				goalmap.at<float>(r,c) = (float)score;
			}
		}
		double minVal,maxVal;
		Point minLoc,maxLoc;
		minMaxLoc(goalmap,&minVal,&maxVal,&minLoc,&maxLoc);
		int s_d[ds];
		s_d[0] = maxLoc.x;
		s_d[1] = maxLoc.y;
		//int idx = getStateIndex(s_d);
		//goal[d].push_back(idx);
		Mat dst;
		jetmap(goalmap,dst);
		circle(dst,maxLoc,5,CV_RGB(255,255,255),1,CV_AA);
		circle(dst,p[0],10,CV_RGB(255,255,255),1,CV_AA);
		
		ss.str("");
		ss << "goalmap-" << d;
		imshow(ss.str(),dst);
		waitKey(1);
	}
	
	
	waitKey(0);
	
}

void trainHIOC::computeGoals( vector<vector <int> > &goal){
	
	stringstream ss;
	ss << FILE_ROOT << actionclass << "-goal-svm.xml";
	Mat v = Mat::ones(1,1,CV_32FC1);
	CvSVM svm;
	cout << "Loading svm model: " << ss.str() << endl;
	svm.load(ss.str().c_str());
	int wsize = WSIZE;
	for(int d=0;d<(int)testList.size();d++){
		goal.push_back(vector<int>(0));
		Mat goalmap = fmap[d][0] * 0;
		for(int c=wsize;c<fmap[d][0].cols-wsize;c++){
			for(int r=wsize;r<fmap[d][0].rows-wsize;r++){
				Point p = Point(c,r);
				Rect rec = Rect(p.x-wsize*0.5,p.y-wsize*0.5,wsize,wsize);
				Mat sample;
				for(int f=0;f<(nf-3);f++){
					//Mat dat = w[f]*fmap[d][f](rec) * 1.0;
					Mat dat = v* mean(w[f] * fmap[d][f](rec)).val[0];
					if(!sample.data) dat.copyTo(sample);
					else sample.push_back(dat);
					
				}
				sample = sample.reshape(1,1);
				CvMat _sample = sample;
				double score = svm.predict(&_sample);
				
				goalmap.at<float>(r,c) = (float)score;
			}
		}
		double minVal,maxVal;
		Point minLoc,maxLoc;
		minMaxLoc(goalmap,&minVal,&maxVal,&minLoc,&maxLoc);
		int s_d[ds];
		s_d[0] = maxLoc.x;
		s_d[1] = maxLoc.y;
		int idx = getStateIndex(s_d);
		goal[d].push_back(idx);
		Mat dst;
		jetmap(goalmap,dst);
		circle(dst,maxLoc,5,CV_RGB(255,255,255),1,CV_AA);
		ss.str("");
		ss << "goalmap-" << d;
		imshow(ss.str(),dst);
		waitKey(1);
	}
	waitKey(0);
}

void trainHIOC::setGoalIndicies(vector<int>&goals){
	
	vector<Point2f> pos;
	
	float inc = 0.05;		// 0.05 space out for visualization
	if(VISOUT) inc = 0.1;
	
	if(action.find("depart")!=string::npos || action.find("walk")!=string::npos ){
		
		for(float i=inc;i<1.00;i+=inc) pos.push_back(Point2f(0.00,i));	// left
		for(float i=inc;i<1.00;i+=inc) pos.push_back(Point2f(i,0.01));	//0.20 UMD top
		for(float i=inc;i<1.00;i+=inc) pos.push_back(Point2f(0.99,i));	// right
		for(float i=inc;i<1.00;i+=inc) pos.push_back(Point2f(i,0.95));  // 0.95 UMD bot

	}
	else if(action.find("approach")!=string::npos){
		// add points surrounding a car (this is done in each testing step...)
	}
	else{
		cout << "ERROR: Goals not set for action:" << action << endl;
		exit(1);
	}
		
		int s_d[ds];
		for(int i=0;i<(int)pos.size();i++){
			s_d[0] = floor(pos[i].x * cols);
			s_d[1] = floor(pos[i].y * rows);
			int idx = getStateIndex(s_d);
			goals.push_back(idx);
		}
	//}
}

void trainHIOC::runTrainMarkovFeat(){
	
	cout << "===============================\n";
	cout << "  Train Markov Feat:" << actionclass << endl;
	cout << "===============================\n";
	
	for(int d=0;d<(int)trainList.size();d++)
	{
		obt.push_back(vector<Point2f>(0));
		trt.push_back(vector<Point2f>(0));
		loadTrajectory(d,trainList[d]);
		scaleTimeTrajectory(d);
		if(DISCRETE) scaleSpaceTrajectory(d);
		loadFeatureMaps(d,trainList[d]);

	}
	updateActionSpace();
	
	computeParametersMarkovFeat();
	
	string s = FILE_ROOT+"models/"+actionclass+".xml";
	FileStorage fs(s,FileStorage::WRITE);
	fs << "mP" << mP;
	fs << "kernelsize" << kernelsize;
	cout << kernelsize << endl;
	
	
	// ==== Deallocate memory ==== //
	for(int d=0;d<(int)fmap.size();d++) fmap[d].clear(); fmap.clear();
	for(int d=0;d<(int)obt.size() ;d++) obt[d].clear();  obt.clear();
	for(int d=0;d<(int)trt.size() ;d++) trt[d].clear();  trt.clear();
	w.clear();
}

void trainHIOC::runTrainMarkovMotion(){
	
	for(int d=0;d<(int)trainList.size();d++)
	{
		obt.push_back(vector<Point2f>(0));
		trt.push_back(vector<Point2f>(0));
		loadTrajectory(d,trainList[d]);
		scaleTimeTrajectory(d);
		if(DISCRETE) scaleSpaceTrajectory(d);
		loadFeatureMaps(d,trainList[d]);
	}
	updateActionSpace();
	computeParametersMarkovMotion();
	string s = FILE_ROOT+"models/"+actionclass+"-markov-motion.xml";
	FileStorage fs(s,FileStorage::WRITE);
	fs << "mPolicy" << mPolicy;
	fs << "kernelsize" << kernelsize;
	
	
	// ==== Deallocate memory ==== //
	for(int d=0;d<(int)fmap.size();d++) fmap[d].clear(); fmap.clear();
	for(int d=0;d<(int)obt.size() ;d++) obt[d].clear();  obt.clear();
	for(int d=0;d<(int)trt.size() ;d++) trt[d].clear();  trt.clear();
	w.clear();
}

void trainHIOC::computeParametersMarkovMotion(){
	
	cout << "Method: computeParametersMarkovMotion (Counting)" << endl;
	
	int s_t[ds];
	int s_next[ds];
	int a_t[da];
	
	//TODO: needs to be optimized (needs to be better than random)
	mPolicy = Mat::ones(ns,na,CV_32FC1)*1.00; // 0.0001 pseudo-counts
	
	int ida,idx;
	
	for(int d=0;d<(int)trt.size();d++){
		for(int t=0;t<(int)trt[d].size()-1;t++){
			
			s_t[0] = trt[d][t].x;
			s_t[1] = trt[d][t].y;
			s_next[0] = trt[d][t+1].x;
			s_next[1] = trt[d][t+1].y;
			a_t[0] = s_next[0]-s_t[0];
			a_t[1] = s_next[1]-s_t[1];
			
			ida = getActionIndex(a_t);
			idx = getStateIndex(s_t);
			
			if(ida>=0 && ida<na && idx>=0 && idx<ns) mPolicy.at<float>(idx,ida) += 1.0;
		}
	}
	
	for(int r=0;r<mPolicy.rows;r++){
		mPolicy.row(r) =  mPolicy.row(r) / sum(mPolicy.row(r)).val[0];
		//cout << r << ") " << mPolicy.row(r) << endl;
	}
	
	//if(DEBUG) cout << "Policy=" << mPolicy << endl;
}

void trainHIOC::computeParametersMarkovFeat(){
	
	cout << "Method: computeParametersMarkov (Linear Regression)" << endl;
	int s_t[ds];
	int s_next[ds];
	int a_t[da];
	
	Mat A;		// action
	Mat F;		// feature responses
	int idx;
	
	for(int d=0;d<(int)trt.size();d++){
		for(int t=0;t<(int)trt[d].size()-1;t++){
			
			s_t[0] = trt[d][t].x;
			s_t[1] = trt[d][t].y;
			s_next[0] = trt[d][t+1].x;
			s_next[1] = trt[d][t+1].y;
			a_t[0] = s_next[0]-s_t[0];
			a_t[1] = s_next[1]-s_t[1];
			
			// store action p(a|s) 
			Mat a_m = Mat::zeros(1,na,CV_32FC1);
			idx = getActionIndex(a_t);
			a_m.at<float>(0,idx) = 1.0;			// 1.0 true action, 0 elsewhere
			
			if(A.data==NULL) a_m.copyTo(A);
			else A.push_back(a_m);
			
			// get features (only current state)
			//Mat f_m = Mat::zeros(1,nf,CV_32FC1);
			//vector<float> feats(nf,0);
			//getFeatures(Point(s_t[0],s_t[1]),feats,d);
			//f_m = Mat(feats).t();
			
			vector<float> features;
			for(int x=-1;x<=1;x++){
				for(int y=-1;y<=1;y++){
					s_next[0] = s_t[0] + x;
					s_next[1] = s_t[1] + y;
					vector<float> _feats(nf,0);
					int si = getStateIndex(s_next);
					if(si>=0) getFeatures(Point(s_t[0],s_t[1]),_feats,d);
					features.insert(features.end(),_feats.begin(),_feats.end());
				}
			}
			features.push_back(1);		// constant feature
			//cout << "features: " <<  Mat(features) << endl;
			//Mat f_m = Mat::ones(1,nf*9+1,CV_32FC1); // allocate memory plus a constant feature
			//cout << "features.size:" << features.size() << endl;
			
			Mat f_m = Mat(features);
			f_m = f_m.reshape(1,1);
			
			//cout << "f_m.rows: " << f_m.rows << endl;
			//cout << "f_m.cols: " << f_m.cols << endl;
			
			//cout << "pause" << endl;
			//cin.ignore();
			// === push feature into MAT === //
			
			if(F.data==NULL) f_m.copyTo(F);
			else F.push_back(f_m);
		}
	}
	
	cout << "Solve LR with QR decomposition" << endl;
	//cout << "F.rows: " << F.rows << endl;
	//cout << "F.cols: " << F.cols << endl;

	//cout << "A.rows: " << A.rows << endl;
	//cout << "A.cols: " << A.cols << endl;

	// A = mP*F
	solve(F,A,mP,DECOMP_QR); // linear regression
	if(DEBUG) cout << "P=" << mP << endl;
	
	//cout << "mP.rows: " << mP.rows << endl;
	//cout << "mP.cols: " << mP.cols << endl;

}

void trainHIOC::updatePolicyMarkovFeat(int d){
	
	int s_t[ds];
	int s_next[ds];
	
	for(int x=0;x<ns;x++)
	{	
		stateIndexToValue(x,s_t);
		vector<float> features;
		for(int c=-1;c<=1;c++){
			for(int y=-1;y<=1;y++){
				s_next[0] = s_t[0] + c;
				s_next[1] = s_t[1] + y;
				vector<float> _feats(nf,0);
				int si = getStateIndex(s_next);
				if(si>=0) getFeatures(Point(s_t[0],s_t[1]),_feats,d);
				features.insert(features.end(),_feats.begin(),_feats.end());
			}
		}
		features.push_back(1);		// constant feature
		Mat f_m = Mat(features);
		f_m = f_m.reshape(1,1);
		Mat a_m = f_m * mP;
		float maxval = -FLT_MAX;
		for(int a=0;a<na;a++) if(a_m.at<float>(0,a) > maxval) maxval = a_m.at<float>(0,a);
		a_m -= maxval; // makes the max value always zero (negative elsewhere) to avoid over/under flow?
		cv::exp(a_m,a_m);
		for(int a=0;a<na;a++){
			if(EXPTYPE=="markovfull") a_m.at<float>(0,a) = a_m.at<float>(0,a)+0.8; // pseudo-count
			else if(EXPTYPE=="markovfeat") a_m.at<float>(0,a) = a_m.at<float>(0,a)+1.0;
			if(a==4) a_m.at<float>(0,a) = 0;
		}			
		float sum =  cv::sum(a_m).val[0];
		for(int a=0;a<na;a++) a_m.at<float>(0,a) /= sum;			
		for(int a=0;a<na;a++){
			pax[a][x] = a_m.at<float>(0,a);
			if(isnan(pax[a][x]) || isinf(pax[a][x]) ){
				cout <<"ERROR: local pax:" << pax[a][x] << endl;
				exit(1);
			}
		}
	}// DONE updating policy for this scene

}

void trainHIOC::runTestMarkovFeat(){
	
	cout << "===============================\n";
	cout << "  Test Markov Feat:" << actionclass << endl;
	cout << "===============================\n";
	
	string s = FILE_ROOT+"models/"+actionclass+".xml";
	FileStorage fs(s,FileStorage::READ);
	fs["mP"] >> mP;
	fs["kernelsize"] >> kernelsize;	
	
	setActionSpace(kernelsize);
	
	pax = vector< vector<float> > (na);
	for(int a=0;a<na;a++) pax[a] = vector<float>(ns,0);
	
	for(int d=0;d<(int)testList.size();d++)
	{
		obt.push_back(vector<Point2f>(0));
		trt.push_back(vector<Point2f>(0));
		loadTrajectory(d,testList[d]);
		scaleTimeTrajectory(d);
		if(DISCRETE) scaleSpaceTrajectory(d);
		loadFeatureMaps(d,testList[d]);
	}
	
	if(job=="transfer") actionclass = "transfer-"+actionclass;
	
	stringstream ss;
	if(!MULTI_GOAL_ON) ss << FILE_ROOT << "hioc_output/test-results-"<< actionclass <<"-"<< trainList.size() <<"-"<< (int)testList.size() << ".txt";
	else ss << FILE_ROOT << "hioc_output/test-results-"<< actionclass <<"-multigoal-"<< trainList.size() <<"-"<< (int)testList.size() << ".txt";
	if(!fll[0].is_open()) fll[0].open(ss.str().c_str());
	
	ss.str("");
	if(!MULTI_GOAL_ON) ss << FILE_ROOT << "hioc_output/test-results-past-"<< actionclass <<"-"<< trainList.size() <<"-"<< (int)testList.size() << ".txt";
	else ss << FILE_ROOT << "hioc_output/test-results-past-"<< actionclass <<"-multigoal-"<< trainList.size() <<"-"<< (int)testList.size() << ".txt";
	if(!fll[1].is_open()) fll[1].open(ss.str().c_str());
	
	ss.str("");
	if(!MULTI_GOAL_ON) ss << FILE_ROOT << "hioc_output/test-results-future-"<< actionclass <<"-"<< trainList.size() <<"-"<< (int)testList.size() << ".txt";
	else ss << FILE_ROOT << "hioc_output/test-results-future-"<< actionclass <<"-multigoal-"<< trainList.size() <<"-"<< (int)testList.size() << ".txt";
	if(!fll[2].is_open()) fll[2].open(ss.str().c_str());
	
	ss.str("");
	if(!MULTI_GOAL_ON) ss << FILE_ROOT << "hioc_output/test-results-smoothing-"<< actionclass <<"-"<< trainList.size() <<"-"<< (int)testList.size() << ".txt";
	else ss << FILE_ROOT << "hioc_output/test-results-smoothing-"<< actionclass <<"-multigoal-"<< trainList.size() <<"-"<< (int)testList.size() << ".txt";
	if(!fll[3].is_open()) fll[3].open(ss.str().c_str());
	
	ss.str("");
	if(!MULTI_GOAL_ON) ss << FILE_ROOT << "hioc_output/test-results-forecasting-"<< actionclass <<"-"<< trainList.size() <<"-"<< (int)testList.size() << ".txt";
	else ss << FILE_ROOT << "hioc_output/test-results-forecasting-"<< actionclass <<"-multigoal-"<< trainList.size() <<"-"<< (int)testList.size() << ".txt";
	if(!fll[4].is_open()) fll[4].open(ss.str().c_str());
	
	float val[5];
	Point minLoc, maxLoc;
	
	for(int d=0;d<(int)testList.size();d++){
		
		cout << "\n=====================================\n";
		cout << "   Test on " << testList[d] << endl;
		cout << "======================================\n";
		
		// ==== sampling for distance metric ==== //
		float logloss;
		
		
		if(!MULTI_GOAL_ON){ // ===== Known goal inference ===== //
			
			
			fll[0] << actionclass.c_str() << "\t" << testList[d] << "\t"; fll[0].flush();
			fll[1] << actionclass.c_str() << "\t" << testList[d] << "\t"; fll[1].flush();
			fll[2] << actionclass.c_str() << "\t" << testList[d] << "\t"; fll[2].flush();
			fll[3] << actionclass.c_str() << "\t" << testList[d] << "\t"; fll[3].flush();
			fll[4] << actionclass.c_str() << "\t" << testList[d] << "\t"; fll[4].flush();
			
			
			if(OBSERVATION_ON && FEATURES_ON) updateObsFeatureMaps(nf-4,(int)trt[d].size(),d,testList[d]);	// full evidence	
			else if(!OBSERVATION_ON && FEATURES_ON) updateObsFeatureMaps(nf-4,0,d,testList[d]);				// no evidence
			
			//===========================//
			updatePolicyMarkovFeat(d);
			//===========================//
			
			if(VISUALIZE) ComputeExactExpectations((int)trt[d].size(),d,testList[d],-1,-1);
			
			sampleTrajectories(d,testList[d]);
			
			if(OBSERVATION_ON) ComputeLogLikelihood(d,(int)trt[d].size(),val);	// set index to end - full evidence
			else ComputeLogLikelihood(d,0,val);									// set index to start - no evidence
			
			logloss = val[0]/(val[3]+val[4]);
			cout << "Log-loss (total):" << logloss << endl;
			fll[0] << logloss << endl; fll[0].flush();
			
			logloss = val[1]/(val[3]+val[4]);
			cout << "Log-loss (past):" << logloss << endl;
			fll[1] << logloss << endl; fll[1].flush();
			
			logloss = val[2]/(val[3]+val[4]);
			cout << "Log-loss (future):"  << logloss << endl;
			fll[2] << logloss << endl; fll[2].flush();
			
			logloss = val[1]/val[3];
			if(isnan(logloss)) logloss = 0;
			cout << "Log-loss(smoothing):"  << logloss << endl;
			fll[3] << logloss << endl; fll[3].flush();
			
			logloss = val[2]/val[4];
			if(isnan(logloss)) logloss = 0;
			cout << "Log-loss(forecasting):"  << logloss << endl;
			fll[4] << logloss << endl; fll[4].flush();
			
		}
		else{
			
			cout << "// ==== MULTI-GOAL INFERENCE ==== //" << endl;
			
			if(VISOUT){
				stringstream ss;
				ss << FILE_ROOT << "visual/" << actionclass << "-"<< testList[d] << "-hybrid.avi";
				avi_cum.open(ss.str().c_str(),CV_FOURCC('X', 'V', 'I', 'D'),30,fmap[0][0].size(),true);
			}
			
			vector <int> g;			// goals
			int state_d[ds];
			
			fll[0] << actionclass.c_str() << "\t" << testList[d] << "\t"; fll[0].flush();
			fll[1] << actionclass.c_str() << "\t" << testList[d] << "\t"; fll[1].flush();
			fll[2] << actionclass.c_str() << "\t" << testList[d] << "\t"; fll[2].flush();
			fll[3] << actionclass.c_str() << "\t" << testList[d] << "\t"; fll[3].flush();
			fll[4] << actionclass.c_str() << "\t" << testList[d] << "\t"; fll[4].flush();
			
			cout << "// ==== SET GOALS ==== //" << endl;
			
			state_d[0] = trt[d][trt[d].size()-1].x;
			state_d[1] = trt[d][trt[d].size()-1].y;
			int idx = getStateIndex(state_d);				// set true goal as on of possible goals
			g.push_back(idx);
			
			if( action.find("depart") != string::npos || action.find("walk")!= string::npos ){				
				
				setGoalIndicies(g);
				
			}
			else if(action.find("approach") != string::npos){
				
				Mat sm; 
				int car_i = 0;
				
				if(FEATURES_ON){
					for(int i=0;i< (int)featurelabel.size();i++){
						if(featurelabel[i]=="car"){
							car_i = i;
						}
					}
					sm = fmap[d][car_i] + 1.0;
				}
				else {
					// load car image
					
					stringstream fn; 
					fn << FILE_ROOT << "imgset_test/feature_maps/" << testList[d] <<"_features.yml";
					if(DEBUG) cout << "Opening: " << fn.str() << endl;
					
					FileStorage fs(fn.str(),FileStorage::READ);
					if(!fs.isOpened()){
						cout << "ERROR:(test) cannot open feature map " << fn.str() << endl;
						exit(1);
					}
					Mat img;
					fs["car"] >> img;							// add feature map (float)
					resize(img,sm,Size(cols,rows),0,0);				// resize, raw probability as a feature					
				}
				
				Mat bin;
				threshold(sm,bin,0.50,255,THRESH_BINARY);
				
				if(VISUALIZE){
					imshow("bin",bin);
					Mat dsp; jetmap(sm,dsp); imshow("dsp",dsp);
					waitKey(1);
				}
				bin.convertTo(bin,CV_8UC1,1.0,0);
				
				vector<vector<Point> > co;
				vector<Vec4i> hi;
				findContours(bin,co,hi,CV_RETR_EXTERNAL,CV_CHAIN_APPROX_SIMPLE);
				
				for(int i=0;i<(int)co.size();i++){
					vector<Point> pts;
					approxPolyDP(Mat(co[i]),pts,1.0,true);
					for(int j=0;j<(int)pts.size();j++){
						//cout << "pts=" << pts[j] << endl;
						state_d[0] = pts[j].x;
						state_d[1] = pts[j].y;
						g.push_back(getStateIndex(state_d));
					}
				}
				
			}
			else{
				cout << "ERROR:" << action << " is an invalid action type for multi-goal inference" << endl;
				exit(1);
			}
			
			if(DEBUG) cout << "Number of goals:" << (int)g.size() << endl;
			
			
			float stopval = 1.01;					// Evaluate with partial evidence
			if(!OBSERVATION_ON) stopval = 0.05;		// When no evidence (feat, const), evaluate only once with goals
			float step = 0.10;
			if(VISUALIZE) step = 1.0/3.0;
			
			
			for(float p = 0.00; p <= stopval; p += step){
				
				int numobs  = round( p * (trt[d].size()-1));
				
				
				cout << "numobs=" << numobs << " out of " << trt[d].size() << endl; 
				cout << "==================================\n";
				cout << "==== Evidence up to " << numobs << " (p="<< p<<") out of " << trt[d].size() << "(stopval:" << stopval <<")"<<endl;
				cout << "==================================\n";
				

				if(OBSERVATION_ON) updateObsFeatureMaps(nf-4,numobs,d,testList[d]);	// add evidence Z

			
				updatePolicyMarkovFeat(d);
				
				
				if(VISUALIZE){
					int s_t = numobs;		// used only for naming files (reset in ComputeExactExpectations)
					int s_0 = 0;
					ComputeExactExpectations(d,testList[d],s_t,s_0,g);				
				}
				
				// ==== COMPUTE SCORES ===== //
				
				int s_t = round(p*100.0);
				//cout << "s_t:" << s_t << endl;
				if(OBSERVATION_ON) sampleTrajectories(d,testList[d],g,s_t);
				else sampleTrajectories(d,testList[d]);
				
				ComputeLogLikelihood(d,numobs,val);
				
				float logloss;
				
				logloss = val[0]/(val[3]+val[4]);
				cout << "Log-loss (total):" << logloss << endl;
				fll[0] << logloss << "\t"; fll[0].flush();
				
				logloss = val[1]/(val[3]+val[4]);
				cout << "Log-loss (past):" << logloss << endl;
				fll[1] << logloss << "\t"; fll[1].flush();
				
				logloss = val[2]/(val[3]+val[4]);
				cout << "Log-loss (future):"  << logloss << endl;
				fll[2] << logloss << "\t"; fll[2].flush();
				
				logloss = val[1]/val[3];
				if(isnan(logloss)) logloss = 0;
				cout << "Log-loss(smoothing):"  << logloss << endl;
				fll[3] << logloss << "\t"; fll[3].flush();
				
				logloss = val[2]/val[4];
				if(isnan(logloss)) logloss = 0;
				cout << "Log-loss(forecasting):"  << logloss << endl;
				fll[4] << logloss << "\t"; fll[4].flush();				
			} // for each p
			
			fll[0] << endl;
			fll[1] << endl;
			fll[2] << endl;
			fll[3] << endl;
			fll[4] << endl;
		} // IF multi-goal inference
	}
			
			
			
	//if(job=="transfer") actionclass = "transfer-"+actionclass;
	
	//if(!fllm.is_open()){
	//	stringstream ss;
	//	if(MULTI_GOAL_ON) ss << FILE_ROOT << "hioc_output/test-results-"<< actionclass <<"-multigoal-"<< trainList.size() << "-"<< testList.size() <<".txt";
	//	else ss << FILE_ROOT << "hioc_output/test-results-"<< actionclass <<"-"<< trainList.size() << "-"<< testList.size() <<".txt";
	//	fllm.open(ss.str().c_str());
	//}

//	
//	for(int d=0;d<(int)testList.size();d++){
//
//		if(d==0) fllm << actionclass.c_str() << "\t" << testList[d] << "\t";
//		
//		float stopval = 1.01;					// Evaluate with partial evidence
//		if(!OBSERVATION_ON) stopval = 0.05;		// When no evidence (feat, const), evaluate only once with goals
//		
//		float step = 0.10;
//		if(VISUALIZE) step = 1.0/3.0;
//		
//		for(float p = 0.00; p <= stopval; p += step){
//			
//			int numobs  = round( p * (trt[d].size()-1));
//			
//			if(OBSERVATION_ON) updateObsFeatureMaps(nf-4,numobs,d,testList[d]); // add observations
//		}
		
			///////////////////////////
			//      UPDATE POLICY    //
			///////////////////////////
			
//			int s_t[ds];
//			int s_next[ds];
//			
//			for(int x=0;x<ns;x++)
//			{	
//				stateIndexToValue(x,s_t);
//				vector<float> features;
//				for(int c=-1;c<=1;c++){
//					for(int y=-1;y<=1;y++){
//						s_next[0] = s_t[0] + c;
//						s_next[1] = s_t[1] + y;
//						vector<float> _feats(nf,0);
//						int si = getStateIndex(s_next);
//						if(si>=0) getFeatures(Point(s_t[0],s_t[1]),_feats,d);
//						features.insert(features.end(),_feats.begin(),_feats.end());
//					}
//				}
//				features.push_back(1);		// constant feature
//				Mat f_m = Mat(features);
//				f_m = f_m.reshape(1,1);
//				Mat a_m = f_m * mP;
//				float maxval = -FLT_MAX;
//				for(int a=0;a<na;a++) if(a_m.at<float>(0,a) > maxval) maxval = a_m.at<float>(0,a);
//				a_m -= maxval; // makes the max value always zero (negative elsewhere) to avoid over/under flow?
//				cv::exp(a_m,a_m);
//				for(int a=0;a<na;a++){
//					if(EXPTYPE=="markovfull") a_m.at<float>(0,a) = a_m.at<float>(0,a)+0.8; // pseudo-count
//					else if(EXPTYPE=="markovfeat") a_m.at<float>(0,a) = a_m.at<float>(0,a)+1.0;
//					if(a==4) a_m.at<float>(0,a) = 0;
//				}			
//				float sum =  cv::sum(a_m).val[0];
//				for(int a=0;a<na;a++) a_m.at<float>(0,a) /= sum;			
//				for(int a=0;a<na;a++){
//					pax[a][x] = a_m.at<float>(0,a);
//					if(isnan(pax[a][x]) || isinf(pax[a][x]) ){
//						cout <<"ERROR: local pax:" << pax[a][x] << endl;
//						exit(1);
//					}
//				}
//			}// DONE updating policy for this scene
		
//			if(VISUALIZE) ComputeExactExpectations((int)trt[d].size(),d,testList[d],-1,-1);
//			
//			sampleTrajectories(d,testList[d]);
//			float logl = computeLogLikelihoodMarkovFeat(d);
//			fllm << logl << endl;
		
//		if(MULTI_GOAL_ON){
//			float stopval = 
//			
//			for(float p = 0.00;p<=1.05; p+= 0.10)
//			{
//				if(DEBUG) cout << "**** PARTIAL EVIDENCE FORECASTING("<< p <<") ****" << endl;
//				updateObsFeatureMaps(nf-4,round((int)trt[d].size()*p),d,testList[d]);
//				float logl = computeLogLikelihoodMarkovFeat(d);
//				fllm << logl << "\t";
//			}
//			fllm << endl;
//		}
//	}
	
	// ==== Deallocate memory ==== //
	for(int d=0;d<(int)fmap.size();d++) fmap[d].clear(); fmap.clear();
	for(int d=0;d<(int)obt.size() ;d++) obt[d].clear();  obt.clear();
	for(int d=0;d<(int)trt.size() ;d++) trt[d].clear();  trt.clear();
	w.clear();
	
}

void trainHIOC::runTestMarkovMotion(){
	
	cout << "===============================\n";
	cout << "  Test Markov Motion:" << actionclass << endl;
	cout << "===============================\n";
	
	string s = FILE_ROOT+"models/"+actionclass+"-markov-motion.xml";
	FileStorage fs(s,FileStorage::READ);
	fs["mPolicy"] >> mPolicy;
	fs["kernelsize"] >> kernelsize;	
	
	setActionSpace(kernelsize);
	
	pax = vector< vector<float> > (na);
	for(int a=0;a<na;a++) pax[a] = vector<float>(ns,0);
	
	
	for(int d=0;d<(int)testList.size();d++)
	{
		obt.push_back(vector<Point2f>(0));
		trt.push_back(vector<Point2f>(0));
		loadTrajectory(d,testList[d]);
		scaleTimeTrajectory(d);
		if(DISCRETE) scaleSpaceTrajectory(d);
		loadFeatureMaps(d,testList[d]);
	}
		
	if(job=="transfer") actionclass = "transfer-"+actionclass;
	
	stringstream ss;
	if(!MULTI_GOAL_ON) ss << FILE_ROOT << "hioc_output/test-results-"<< actionclass <<"-"<< trainList.size() <<"-"<< (int)testList.size() << ".txt";
	else ss << FILE_ROOT << "hioc_output/test-results-"<< actionclass <<"-multigoal-"<< trainList.size() <<"-"<< (int)testList.size() << ".txt";
	if(!fll[0].is_open()) fll[0].open(ss.str().c_str());
	
	ss.str("");
	if(!MULTI_GOAL_ON) ss << FILE_ROOT << "hioc_output/test-results-past-"<< actionclass <<"-"<< trainList.size() <<"-"<< (int)testList.size() << ".txt";
	else ss << FILE_ROOT << "hioc_output/test-results-past-"<< actionclass <<"-multigoal-"<< trainList.size() <<"-"<< (int)testList.size() << ".txt";
	if(!fll[1].is_open()) fll[1].open(ss.str().c_str());
	
	ss.str("");
	if(!MULTI_GOAL_ON) ss << FILE_ROOT << "hioc_output/test-results-future-"<< actionclass <<"-"<< trainList.size() <<"-"<< (int)testList.size() << ".txt";
	else ss << FILE_ROOT << "hioc_output/test-results-future-"<< actionclass <<"-multigoal-"<< trainList.size() <<"-"<< (int)testList.size() << ".txt";
	if(!fll[2].is_open()) fll[2].open(ss.str().c_str());
	
	ss.str("");
	if(!MULTI_GOAL_ON) ss << FILE_ROOT << "hioc_output/test-results-smoothing-"<< actionclass <<"-"<< trainList.size() <<"-"<< (int)testList.size() << ".txt";
	else ss << FILE_ROOT << "hioc_output/test-results-smoothing-"<< actionclass <<"-multigoal-"<< trainList.size() <<"-"<< (int)testList.size() << ".txt";
	if(!fll[3].is_open()) fll[3].open(ss.str().c_str());
	
	ss.str("");
	if(!MULTI_GOAL_ON) ss << FILE_ROOT << "hioc_output/test-results-forecasting-"<< actionclass <<"-"<< trainList.size() <<"-"<< (int)testList.size() << ".txt";
	else ss << FILE_ROOT << "hioc_output/test-results-forecasting-"<< actionclass <<"-multigoal-"<< trainList.size() <<"-"<< (int)testList.size() << ".txt";
	if(!fll[4].is_open()) fll[4].open(ss.str().c_str());
	
	float val[5];
	Point minLoc, maxLoc;
	
	for(int d=0;d<(int)testList.size();d++){
		
		cout << "\n=====================================\n";
		cout << "   Test on " << testList[d] << endl;
		cout << "======================================\n";
		
		// ==== sampling for distance metric ==== //
		float logloss;
		
		
		if(!MULTI_GOAL_ON){ // ===== Known goal inference ===== //
			
			
			fll[0] << actionclass.c_str() << "\t" << testList[d] << "\t"; fll[0].flush();
			fll[1] << actionclass.c_str() << "\t" << testList[d] << "\t"; fll[1].flush();
			fll[2] << actionclass.c_str() << "\t" << testList[d] << "\t"; fll[2].flush();
			fll[3] << actionclass.c_str() << "\t" << testList[d] << "\t"; fll[3].flush();
			fll[4] << actionclass.c_str() << "\t" << testList[d] << "\t"; fll[4].flush();
			
			
			if(OBSERVATION_ON && FEATURES_ON) updateObsFeatureMaps(nf-4,(int)trt[d].size(),d,testList[d]);	// full evidence	
			else if(!OBSERVATION_ON && FEATURES_ON) updateObsFeatureMaps(nf-4,0,d,testList[d]);				// no evidence
			
			//==========================================//
			for(int x=0;x<ns;x++)		
				for(int a=0;a<na;a++)
					pax[a][x] = mPolicy.at<float>(x,a);
			//==========================================//
			
			if(VISUALIZE) ComputeExactExpectations((int)trt[d].size(),d,testList[d],-1,-1);
			sampleTrajectories(d,testList[d]);
			
			if(OBSERVATION_ON) ComputeLogLikelihood(d,(int)trt[d].size(),val);	// set index to end - full evidence
			else ComputeLogLikelihood(d,0,val);									// set index to start - no evidence
			
			logloss = val[0]/(val[3]+val[4]);
			cout << "Log-loss (total):" << logloss << endl;
			fll[0] << logloss << endl; fll[0].flush();
			
			logloss = val[1]/(val[3]+val[4]);
			cout << "Log-loss (past):" << logloss << endl;
			fll[1] << logloss << endl; fll[1].flush();
			
			logloss = val[2]/(val[3]+val[4]);
			cout << "Log-loss (future):"  << logloss << endl;
			fll[2] << logloss << endl; fll[2].flush();
			
			logloss = val[1]/val[3];
			if(isnan(logloss)) logloss = 0;
			cout << "Log-loss(smoothing):"  << logloss << endl;
			fll[3] << logloss << endl; fll[3].flush();
			
			logloss = val[2]/val[4];
			if(isnan(logloss)) logloss = 0;
			cout << "Log-loss(forecasting):"  << logloss << endl;
			fll[4] << logloss << endl; fll[4].flush();
			
		}
		else{
			
			cout << "// ==== MULTI-GOAL INFERENCE ==== //" << endl;
			
			if(VISOUT){
				stringstream ss;
				ss << FILE_ROOT << "visual/" << actionclass << "-"<< testList[d] << "-hybrid.avi";
				avi_cum.open(ss.str().c_str(),CV_FOURCC('X', 'V', 'I', 'D'),30,fmap[0][0].size(),true);
			}
			
			vector <int> g;			// goals
			int state_d[ds];
			
			fll[0] << actionclass.c_str() << "\t" << testList[d] << "\t"; fll[0].flush();
			fll[1] << actionclass.c_str() << "\t" << testList[d] << "\t"; fll[1].flush();
			fll[2] << actionclass.c_str() << "\t" << testList[d] << "\t"; fll[2].flush();
			fll[3] << actionclass.c_str() << "\t" << testList[d] << "\t"; fll[3].flush();
			fll[4] << actionclass.c_str() << "\t" << testList[d] << "\t"; fll[4].flush();
			
			cout << "// ==== SET GOALS ==== //" << endl;
			
			state_d[0] = trt[d][trt[d].size()-1].x;
			state_d[1] = trt[d][trt[d].size()-1].y;
			int idx = getStateIndex(state_d);				// set true goal as on of possible goals
			g.push_back(idx);
			
			if(  action.find("depart") != string::npos || action.find("walk")!= string::npos ){				
				
				setGoalIndicies(g);
				
			}
			else if(action.find("approach") != string::npos){
				
				Mat sm; 
				int car_i = 0;
				
				if(FEATURES_ON){
					for(int i=0;i< (int)featurelabel.size();i++){
						if(featurelabel[i]=="car"){
							car_i = i;
						}
					}
					sm = fmap[d][car_i] + 1.0;
				}
				else {
					// load car image
					
					stringstream fn; 
					fn << FILE_ROOT << "imgset_test/feature_maps/" << testList[d] <<"_features.yml";
					if(DEBUG) cout << "Opening: " << fn.str() << endl;
					
					FileStorage fs(fn.str(),FileStorage::READ);
					if(!fs.isOpened()){
						cout << "ERROR:(test) cannot open feature map " << fn.str() << endl;
						exit(1);
					}
					Mat img;
					fs["car"] >> img;							// add feature map (float)
					resize(img,sm,Size(cols,rows),0,0);				// resize, raw probability as a feature					
				}
				
				Mat bin;
				threshold(sm,bin,0.50,255,THRESH_BINARY);
				
				if(VISUALIZE){
					imshow("bin",bin);
					Mat dsp; jetmap(sm,dsp); imshow("dsp",dsp);
					waitKey(1);
				}
				bin.convertTo(bin,CV_8UC1,1.0,0);
				
				vector<vector<Point> > co;
				vector<Vec4i> hi;
				findContours(bin,co,hi,CV_RETR_EXTERNAL,CV_CHAIN_APPROX_SIMPLE);
				
				for(int i=0;i<(int)co.size();i++){
					vector<Point> pts;
					approxPolyDP(Mat(co[i]),pts,1.0,true);
					for(int j=0;j<(int)pts.size();j++){
						//cout << "pts=" << pts[j] << endl;
						state_d[0] = pts[j].x;
						state_d[1] = pts[j].y;
						g.push_back(getStateIndex(state_d));
					}
				}
				
			}
			else{
				cout << "ERROR:" << action << " is an invalid action type for multi-goal inference" << endl;
				exit(1);
			}
			
			if(DEBUG) cout << "Number of goals:" << (int)g.size() << endl;
			
			
			float stopval = 1.01;					// Evaluate with partial evidence
			if(!OBSERVATION_ON) stopval = 0.05;		// When no evidence (feat, const), evaluate only once with goals
			float step = 0.10;
			if(VISUALIZE) step = 1.0/3.0;
			
			
			for(float p = 0.00; p <= stopval; p += step){
				
				int numobs  = round( p * (trt[d].size()-1));
				
				
				cout << "numobs=" << numobs << " out of " << trt[d].size() << endl; 
				cout << "==================================\n";
				cout << "==== Evidence up to " << numobs << " (p="<< p<<") out of " << trt[d].size() << "(stopval:" << stopval <<")"<<endl;
				cout << "==================================\n";
				
				
				if(OBSERVATION_ON) updateObsFeatureMaps(nf-4,numobs,d,testList[d]);	// add evidence Z
				
				//==============================================//
				for(int x=0;x<ns;x++)		
					for(int a=0;a<na;a++)
						pax[a][x] = mPolicy.at<float>(x,a);
				//==============================================//
				
				
				if(VISUALIZE){
					int s_t = numobs;		// used only for naming files (reset in ComputeExactExpectations)
					int s_0 = 0;
					ComputeExactExpectations(d,testList[d],s_t,s_0,g);				
				}
				
				// ==== COMPUTE SCORES ===== //
				
				int s_t = round(p*100.0);
				//cout << "s_t:" << s_t << endl;
				if(OBSERVATION_ON) sampleTrajectories(d,testList[d],g,s_t);
				else sampleTrajectories(d,testList[d]);
				
				ComputeLogLikelihood(d,numobs,val);
				
				float logloss;
				
				logloss = val[0]/(val[3]+val[4]);
				cout << "Log-loss (total):" << logloss << endl;
				fll[0] << logloss << "\t"; fll[0].flush();
				
				logloss = val[1]/(val[3]+val[4]);
				cout << "Log-loss (past):" << logloss << endl;
				fll[1] << logloss << "\t"; fll[1].flush();
				
				logloss = val[2]/(val[3]+val[4]);
				cout << "Log-loss (future):"  << logloss << endl;
				fll[2] << logloss << "\t"; fll[2].flush();
				
				logloss = val[1]/val[3];
				if(isnan(logloss)) logloss = 0;
				cout << "Log-loss(smoothing):"  << logloss << endl;
				fll[3] << logloss << "\t"; fll[3].flush();
				
				logloss = val[2]/val[4];
				if(isnan(logloss)) logloss = 0;
				cout << "Log-loss(forecasting):"  << logloss << endl;
				fll[4] << logloss << "\t"; fll[4].flush();				
			} // for each p
			
			fll[0] << endl;
			fll[1] << endl;
			fll[2] << endl;
			fll[3] << endl;
			fll[4] << endl;
		} // IF multi-goal inference
	}

	
	
//	if(!fllm.is_open()){
//		stringstream ss;
//		if(MULTI_GOAL_ON) ss << FILE_ROOT << "hioc_output/test-results-"<< actionclass <<"-multigoal-"<< trainList.size() << "-"<< testList.size() <<".txt";
//		else ss << FILE_ROOT << "hioc_output/test-results-"<< actionclass <<"-"<< trainList.size() << "-"<< testList.size() <<".txt";
//		
//		fllm.open(ss.str().c_str());
//	}
//	
//	
//	for(int d=0;d<(int)testList.size();d++){
//		
//		for(int x=0;x<ns;x++)		
//			for(int a=0;a<na;a++)
//				pax[a][x] = mPolicy.at<float>(x,a);
//				
//		
//		sampleTrajectories(d,testList[d]);
//		fllm << actionclass.c_str() << "\t" << testList[d] << "\t";		
//		float logl = computeLogLikelihoodMarkovMotion(d);
//		cout << "average likelihood: " << logl << endl; 
//		fllm << logl << "\t";
//		fllm << endl;
//		
//		
//		if(VISUALIZE || VISOUT) ComputeExactExpectations((int)trt[d].size(),d,testList[d],-1,-1);
//		
//	}
	
	// ==== Deallocate memory ==== //
	for(int d=0;d<(int)fmap.size();d++) fmap[d].clear(); fmap.clear();
	for(int d=0;d<(int)obt.size() ;d++) obt[d].clear();  obt.clear();
	for(int d=0;d<(int)trt.size() ;d++) trt[d].clear();  trt.clear();
	w.clear();
	
}

void trainHIOC::setActionSpace(int ks){
	
	cout << "Method:setActionSpace" << endl;
	
	kernelsize = ks;
	
	if(kernelsize == 3){
		amn[0] = -1;
		amn[1] = -1;
		amx[0] =  1;
		amx[1] =  1;		
	}
	else if(kernelsize == 5){
		amn[0] = -2;
		amn[1] = -2;
		amx[0] =  2;
		amx[1] =  2;		
	}
	else if(kernelsize == 7){
		amn[0] = -3;
		amn[1] = -3;
		amx[0] =  3;
		amx[1] =  3;
		
	}
	else if(kernelsize == 9){
		amn[0] = -4;
		amn[1] = -4;
		amx[0] =  4;
		amx[1] =  4;
	}
	else if(kernelsize == 11){
		amn[0] = -5;
		amn[1] = -5;
		amx[0] =  5;
		amx[1] =  5;
	}
	else {
		cout << "ERROR:Kernelsize " <<kernelsize << " is invalid" << endl;
		exit(1);
	}
	
	float actiondims = 1;
	for(int i=0;i<da;i++){
		ba[i] = (amx[i]-amn[i]+1);
		actiondims *= (ba[i]);
	}
	
	na = actiondims;
	
	if(DEBUG) cout << "  Total number of action=" << actiondims << endl;
	cout << "  Kernel size is " << kernelsize << endl;
	
	if(na != kernelsize*kernelsize) cout << "ERROR: size of action space incorrect!" << endl;
	
	
}

float trainHIOC::computeLogLikelihoodMarkovMotion(int d){
	
	if(DEBUG) cout << "Method: computeLikelihoodMarkovMotion" << endl;
	
	int s_t[ds];
	int s_next[ds];
	int a_t[da];
	int ida,idx;
	
	float avgloglikelihood=0;
	
	for(int t=0;t<(int)trt[d].size()-1;t++){
		
		s_t[0] = trt[d][t].x;
		s_t[1] = trt[d][t].y;
		s_next[0] = trt[d][t+1].x;
		s_next[1] = trt[d][t+1].y;
		a_t[0] = s_next[0]-s_t[0];
		a_t[1] = s_next[1]-s_t[1];
		
		ida = getActionIndex(a_t);
		idx = getStateIndex(s_t);
		
		if(ida>=0 && ida<na && idx>=0 && idx<ns){
			
			float local_pax = mPolicy.at<float>(idx,ida);
			//cout << local_pax << endl;
			if(isnan(local_pax) || isinf(local_pax) ){
				cout <<"ERROR: local pax:" << local_pax << endl;
			}
			
			avgloglikelihood += log(local_pax);
		}
		
	}
	
	avgloglikelihood /= ((int)trt[d].size()-1);
	return avgloglikelihood;
	
	
}

// Not used...
float trainHIOC::computeLogLikelihoodMarkovFeat(int d){
	
	if(DEBUG) cout << "Method: computeLikelihoodMarkovFeat" << endl;
	
	int s_t[ds];
	int s_next[ds];
	int a_t[da];
	int idx,ida;
	
	float avgloglikelihood=0;
	//int k=0;
	
	for(int t=0;t<(int)trt[d].size()-1;t++){
		
		s_t[0] = trt[d][t].x;
		s_t[1] = trt[d][t].y;
		s_next[0] = trt[d][t+1].x;
		s_next[1] = trt[d][t+1].y;
		a_t[0] = s_next[0]-s_t[0];
		a_t[1] = s_next[1]-s_t[1];
		
		ida = getActionIndex(a_t);
		idx = getStateIndex(s_t);
	
		if(0){
			Mat a_m = Mat::zeros(1,nf,CV_32FC1);
			
			vector<float> feats(nf,0);
			getFeatures(Point(s_t[0],s_t[1]),feats,d);
			Mat f_m = Mat(feats).t();
			
			a_m = f_m * mP;
			
			// approximation of log p(a|s), therefore must be negative!
			float maxval = -FLT_MAX;
			for(int a=0;a<na;a++) 
				if(a_m.at<float>(0,a) > maxval) maxval = a_m.at<float>(0,a);
			a_m -= maxval; // makes the max value always zero (negative elsewhere)
			
			
			//float sum;
			//for(int a=0;a<na;a++) sum += ( exp(a_m.at<float>(0,a))+0.0001 ); // psuedo-count
			cv::exp(a_m,a_m);
			for(int a=0;a<na;a++) a_m.at<float>(0,a) = a_m.at<float>(0,a) + 0.0001;		
			for(int a=0;a<na;a++) a_m.at<float>(0,a) /= cv::sum(a_m).val[0];
			//cout << a_m << " " << cv::sum(a_m).val[0] << endl;
				
			//cout << a_m << " " << cv::sum(a_m).val[0] << endl;
		}
		
		//float local_pax = a_m.at<float>(0,idx);
		
		float local_pax = pax[ida][idx];
		//cout << "localpax:" << local_pax << endl;
		
		if(isnan(local_pax) || isinf(local_pax) ){
			cout <<"ERROR: local pax:" << local_pax << endl;
		}
		
		avgloglikelihood += log(local_pax);
		//k++;
		//cout << avgloglikelihood << endl;
	}
	
	//if(k!=((float)trt[d].size()-1)) cout << k << "!=" << ((float)trt[d].size()-1) << endl;
	avgloglikelihood /= ((float)trt[d].size()-1);
	cout << "average log-likelihood:" << avgloglikelihood << endl;
	return avgloglikelihood;
	
	
}



void trainHIOC::setFiles(string actionclass, vector<string> &trainList){
	this->actionclass = actionclass;
	this->trainList = trainList;
}


void trainHIOC::setTrainFiles(string actionclass, vector<string> &trainList, vector<string> &masterList){
	
	this->masterList = masterList;
	this->actionclass = actionclass;
	this->trainList = trainList;
	
	stringstream ss;
	
	for(int i=0;i<(int)masterList.size();i++){
		int b = 0;
		for(int j=0;j<(int)trainList.size();j++){
			if(masterList[i]==trainList[j]) b = 1;
		}
		ss << b;
	}
	
	trainid = ss.str();
	cout << "id number:" << trainid << endl;
	
	this->actionclass = actionclass + trainid;
	cout << "action class:" << actionclass << endl;
}

void trainHIOC::setTestFiles(vector<string> &testList){
	this->testList = testList;
}


void trainHIOC::runTrainHIOC(int max_iterations, float stepsize){
	
	lambda = stepsize;
	
	// === Compute EMPIRICAL feature response === //
	
	cout << "\n****************************************\n";
	cout << "    Compute EMPIRICAL feature response";
	cout << "\n****************************************\n";
	
	cout << "size of training data:" << trainList.size() << endl; 
	for(int d=0;d<(int)trainList.size();d++)
	{
		cout << d << "] Load training data for " << actionclass << endl;
		obt.push_back(vector<Point2f>(0));
		trt.push_back(vector<Point2f>(0));
		loadTrajectory(d,trainList[d]);
		scaleTimeTrajectory(d);
		if(DISCRETE) scaleSpaceTrajectory(d);
		loadFeatureMaps(d,trainList[d]);
		computeEmpiricalStatistics(d);
		if( empirical_mean_f.size()==0) empirical_mean_f = vector<float> (nf,0);
		for(int f=0;f<nf;f++) empirical_mean_f[f] += empirical_f[d][f];
	}
	cout << "***size of training data:" << trainList.size() << endl;
	
	updateActionSpace();
	for(int f=0;f<nf;f++) empirical_mean_f[f] /= (int)trainList.size();
	cout << "MEAN EMPIRICAL feature count" << endl;
	for(int f=0;f<nf;f++) cout << empirical_mean_f[f] << " "; cout << endl;
	
	
	//trainGoalClassifier();
	
	
	if(1){
		cout << "********************************\n";
		cout << "             HIOC\n";
		cout << "********************************\n";

		// === Memory allocation === //
		w = vector<float> (nf,W_INIT);
		pax = vector< vector<float> > (na);
		for(int a=0;a<na;a++) pax[a] = vector<float>(ns,0);

		// ==== variables ==== //
		float val[5];
		int n=-1;
		
		w_best = vector<float>(nf,0);
		f_diff = vector<float>(nf,0);
		lambda_best = lambda;
		min_loglikelihood = -FLT_MAX;
		
		
		float total_loglikelihood;
		float average_loglikelihood;
		float sum;
		int stat;
		float delta;
		vector<float> tmp((int)trainList.size(),0);
		
		
		stringstream ss;
		ss << FILE_ROOT << "hioc_output/" << actionclass << "-likelihood.txt";
		ofstream logl;
		if(APPEND_RESULTS){
			loadParameters();										// load previous weight parameters
			logl.open(ss.str().c_str(),ios::out|ios::app);			// append
		}
		else logl.open(ss.str().c_str(),ios::out|ios_base::trunc);	// clear
		
		
				
		while(n<max_iterations){
			
			n++;
			
			cout << "\n==============\n";
			cout << " " << n << " " << actionclass;
			cout << "\n==============\n";
			
			cout << "\n*********** WEIGHTS **************\n"; for(int f=0;f<nf;f++) printf("w[%d]=%0.5f (%s)\n",f,w[f],featurelabel[f].c_str());

			
			expected_mean_f = vector<float> (nf,0);		// zero memory
			sum = 0;
			stat = 0;
			delta = 0;
			total_loglikelihood = 0;
			tmp = vector<float> ((int)trainList.size(),0);
			
			for(int d=0;d<(int)trainList.size();d++)
			{
				stat = ValueIteration(n,d,trainList[d],-1);
				if(stat == -1) break; 
				stat = ComputeExactExpectations((int)trt[d].size(),d,trainList[d],-1,-1);
				if(stat == -1) break;
				for(int f=0;f<nf;f++) expected_mean_f[f] += expected_f[f];
				cout << "  Expected f:"; for(int f=0;f<nf;f++) cout << expected_f[f] << " "; cout << endl;
				ComputeLogLikelihood(d,(int)trt[d].size(),val);
				tmp[d] = val[0];
				total_loglikelihood +=val[0];
				sum+=(val[3]+val[4]);
				
			}
			
			average_loglikelihood = total_loglikelihood / sum;
			delta = min_loglikelihood - total_loglikelihood;
			
			cout << "min_loglikelihood = " << min_loglikelihood << endl;
			cout << "total_loglikelihood = " << total_loglikelihood << endl;
			cout << "delta=" << min_loglikelihood << "-"<< total_loglikelihood << "=" <<delta << endl;
			
			if(stat==-1 || delta>0){ // bad convergence or bad likelihood score, lower step size...
				
				cout << "UNSUCCESSFUL! Decrease in log-likelihood or cannot converge!" << endl;
				float dec_rate = 0.7;
				lambda = lambda * dec_rate;
				cout << "Lower step size to: (" << lambda <<"*"<< dec_rate<<")=" << lambda << endl;				
				//cout << "best weights: ";for(int f=0;f<nf;f++) cout << w_best[f] << " ";cout << endl;

				
				if(w_best[0]>0) for(int f=0;f<nf;f++) w[f] = w_best[f] * exp( lambda * ( f_diff[f]) );
				else for(int f=0;f<nf;f++) w[f] *= 1.0/lambda; // increase step size (if first step)
				
				cout << "New weights:";
				for(int f=0;f<nf;f++) cout << w[f] << " ";
				
			}
			else if(delta < 0){ // value function converges, likelihood is better
			
				cout << "SUCCESS! Increase in log-likelihood." << endl;
				for(int f=0;f<nf;f++) expected_mean_f[f] /= (int)trainList.size();
				
				// write to file...
				logl << n << "\t";
				for(int d=0;d<(int)trainList.size();d++) logl << tmp[d] << "\t"; // write: log-likely
				logl << total_loglikelihood   << "\t";		// write: total log-likelihood
				logl << average_loglikelihood << "\t";		// write: average log-likelihood
				logl << lambda << "\t";		// write: step-size
				for(int f=0;f<nf;f++) logl << w[f] << "\t";
				logl << endl;
				
				
				cout << "\n******** AVERAGE LOG-LIKELIHOOD ***********\n";
				cout << "Total Average Log Likelihood=" << average_loglikelihood << endl;
				cout << "\n****** COMPARE MEAN FEATURE COUNTS ********\n";
				cout << "f_EMPIRICAL:"; for(int i=0;i<nf;i++) printf("%04.3f ",empirical_mean_f[i]); cout << endl;
				cout << "f_EXPECTED :"; for(int i=0;i<nf;i++) printf("%04.3f ",expected_mean_f[i] ); cout << endl;		
				

					
				min_loglikelihood = total_loglikelihood;
				cout << "Update BEST! New min log-likelihood: " << min_loglikelihood << endl;
				for(int f=0;f<nf;f++) w_best[f] = w[f];
				for(int f=0;f<nf;f++) f_diff[f] = empirical_mean_f[f] - expected_mean_f[f];
				lambda_best = lambda;
				//saveParameters();
				
				if(delta<-1.0) lambda *= 1.01;	// conservative increase
				else lambda *= 1.5;				// agressive increase
				cout << "New step size to try is:" << lambda << endl;
				
				cout << "Updated weights." << endl;
				for(int f=0;f<nf;f++) w[f] = w[f] * exp( lambda * ( empirical_mean_f[f] - expected_mean_f[f]) );
					

			}
			else{
				cout << "Change in likelihood is ZERO. Done!" << endl;
				n = max_iterations;
			}
			
			saveParameters();
		}// while
	}
	
	// ==== Deallocate memory ==== //
	for(int d=0;d<(int)fmap.size();d++) fmap[d].clear(); fmap.clear();
	for(int d=0;d<(int)obt.size();d++) obt[d].clear(); obt.clear();
	for(int d=0;d<(int)trt.size();d++) trt[d].clear(); trt.clear();
	for(int d=0;d<(int)empirical_f.size();d++) empirical_f[d].clear(); empirical_f.clear();
	expected_f.clear();
	empirical_mean_f.clear();
	expected_mean_f.clear();
	
}


void trainHIOC::runTestHIOCgoals(){
	
	if(DEBUG) cout << "Method: runTestHIOC with goals" << endl;
	
	loadParameters();
	
	if(w.size()==0){
		cout << "ERROR: Parameter file does not exists." << endl;
		exit(1);
	}
	
	setActionSpace(kernelsize);
	pax = vector< vector<float> > (na);
	for(int a=0;a<na;a++) pax[a] = vector<float>(ns,0);
	
	vector<Point2f> pts;
	vector<float> feats;
	
	for(int d=0;d<(int)testList.size();d++)
	{
		obt.push_back(pts);
		trt.push_back(pts);
		loadTrajectory(d,testList[d],1); // testing flag=1
		scaleTimeTrajectory(d);
		if(DISCRETE) scaleSpaceTrajectory(d);
		loadFeatureMaps(d,testList[d]);
	}
	
	if(job=="transfer") actionclass = "transfer-"+actionclass;
	
	stringstream ss;
	if(!MULTI_GOAL_ON) ss << FILE_ROOT << "hioc_output/test-results-"<< actionclass <<"-"<< trainList.size() <<"-"<< (int)testList.size() << ".txt";
	else ss << FILE_ROOT << "hioc_output/test-results-"<< actionclass <<"-multigoal-"<< trainList.size() <<"-"<< (int)testList.size() << ".txt";
	if(!fll[0].is_open()) fll[0].open(ss.str().c_str());
	
	ss.str("");
	if(!MULTI_GOAL_ON) ss << FILE_ROOT << "hioc_output/test-results-past-"<< actionclass <<"-"<< trainList.size() <<"-"<< (int)testList.size() << ".txt";
	else ss << FILE_ROOT << "hioc_output/test-results-past-"<< actionclass <<"-multigoal-"<< trainList.size() <<"-"<< (int)testList.size() << ".txt";
	if(!fll[1].is_open()) fll[1].open(ss.str().c_str());
	
	ss.str("");
	if(!MULTI_GOAL_ON) ss << FILE_ROOT << "hioc_output/test-results-future-"<< actionclass <<"-"<< trainList.size() <<"-"<< (int)testList.size() << ".txt";
	else ss << FILE_ROOT << "hioc_output/test-results-future-"<< actionclass <<"-multigoal-"<< trainList.size() <<"-"<< (int)testList.size() << ".txt";
	if(!fll[2].is_open()) fll[2].open(ss.str().c_str());
	
	ss.str("");
	if(!MULTI_GOAL_ON) ss << FILE_ROOT << "hioc_output/test-results-smoothing-"<< actionclass <<"-"<< trainList.size() <<"-"<< (int)testList.size() << ".txt";
	else ss << FILE_ROOT << "hioc_output/test-results-smoothing-"<< actionclass <<"-multigoal-"<< trainList.size() <<"-"<< (int)testList.size() << ".txt";
	if(!fll[3].is_open()) fll[3].open(ss.str().c_str());
	
	ss.str("");
	if(!MULTI_GOAL_ON) ss << FILE_ROOT << "hioc_output/test-results-forecasting-"<< actionclass <<"-"<< trainList.size() <<"-"<< (int)testList.size() << ".txt";
	else ss << FILE_ROOT << "hioc_output/test-results-forecasting-"<< actionclass <<"-multigoal-"<< trainList.size() <<"-"<< (int)testList.size() << ".txt";
	if(!fll[4].is_open()) fll[4].open(ss.str().c_str());
	
	float val[5];
	Point minLoc, maxLoc;
	
	
	for(int d=0;d<(int)testList.size();d++){
		
		cout << "\n=====================================\n";
		cout << "   Test on " << testList[d] << endl;
		cout << "======================================\n";
		
		// ==== sampling for distance metric ==== //
		int stat = 0;
		float logloss;
		
		
		if(!MULTI_GOAL_ON){ // ===== Known goal inference ===== //
			

			fll[0] << actionclass.c_str() << "\t" << testList[d] << "\t"; fll[0].flush();
			fll[1] << actionclass.c_str() << "\t" << testList[d] << "\t"; fll[1].flush();
			fll[2] << actionclass.c_str() << "\t" << testList[d] << "\t"; fll[2].flush();
			fll[3] << actionclass.c_str() << "\t" << testList[d] << "\t"; fll[3].flush();
			fll[4] << actionclass.c_str() << "\t" << testList[d] << "\t"; fll[4].flush();
			
			
			if(OBSERVATION_ON && FEATURES_ON) updateObsFeatureMaps(nf-4,(int)trt[d].size(),d,testList[d]);	// full evidence	
			else if(!OBSERVATION_ON && FEATURES_ON) updateObsFeatureMaps(nf-4,0,d,testList[d]);				// no evidence
			
			stat = ValueIteration(0,d,testList[d],-1);
			if(stat == -1){
				fll[0] << -FLT_MAX << endl; fll[0].flush();
				fll[1] << -FLT_MAX << endl; fll[1].flush();
				fll[2] << -FLT_MAX << endl; fll[2].flush();
				fll[3] << -FLT_MAX << endl; fll[3].flush();
				fll[4] << -FLT_MAX << endl; fll[4].flush();
				continue;
			}

			if(VISUALIZE) stat = ComputeExactExpectations((int)trt[d].size(),d,testList[d],-1,-1);
			if(stat == -1){
				fll[0] << -FLT_MAX << endl; fll[0].flush();
				fll[1] << -FLT_MAX << endl; fll[1].flush();
				fll[2] << -FLT_MAX << endl; fll[2].flush();
				fll[3] << -FLT_MAX << endl; fll[3].flush();
				fll[4] << -FLT_MAX << endl; fll[4].flush();
				continue;
			}
			
			sampleTrajectories(d,testList[d]);
			
			if(OBSERVATION_ON) ComputeLogLikelihood(d,(int)trt[d].size(),val);	// set index to end - full evidence
			else ComputeLogLikelihood(d,0,val);									// set index to start - no evidence

			logloss = val[0]/(val[3]+val[4]);
			cout << "Log-loss (total):" << logloss << endl;
			fll[0] << logloss << endl; fll[0].flush();
			
			logloss = val[1]/(val[3]+val[4]);
			cout << "Log-loss (past):" << logloss << endl;
			fll[1] << logloss << endl; fll[1].flush();
			
			logloss = val[2]/(val[3]+val[4]);
			cout << "Log-loss (future):"  << logloss << endl;
			fll[2] << logloss << endl; fll[2].flush();
			
			logloss = val[1]/val[3];
			if(isnan(logloss)) logloss = 0;
			cout << "Log-loss(smoothing):"  << logloss << endl;
			fll[3] << logloss << endl; fll[3].flush();
			
			logloss = val[2]/val[4];
			if(isnan(logloss)) logloss = 0;
			cout << "Log-loss(forecasting):"  << logloss << endl;
			fll[4] << logloss << endl; fll[4].flush();
		
		}
		else{		// ==== MULTI-GOAL INFERENCE ==== //
			
			cout << "// ==== MULTI-GOAL INFERENCE ==== //" << endl;
			
			if(VISOUT){
				stringstream ss;
				ss << FILE_ROOT << "visual/" << actionclass << "-"<< testList[d] << "-hybrid.avi";
				avi_cum.open(ss.str().c_str(),CV_FOURCC('X', 'V', 'I', 'D'),30,fmap[0][0].size(),true);
			}
						
			vector <int> g;			// goals
			vector <float> g_val;
			Mat logZ0;
			Mat logZt;
			int state_d[ds];
			
			fll[0] << actionclass.c_str() << "\t" << testList[d] << "\t"; fll[0].flush();
			fll[1] << actionclass.c_str() << "\t" << testList[d] << "\t"; fll[1].flush();
			fll[2] << actionclass.c_str() << "\t" << testList[d] << "\t"; fll[2].flush();
			fll[3] << actionclass.c_str() << "\t" << testList[d] << "\t"; fll[3].flush();
			fll[4] << actionclass.c_str() << "\t" << testList[d] << "\t"; fll[4].flush();
			
			cout << "// ==== SET GOALS ==== //" << endl;
			
			state_d[0] = trt[d][trt[d].size()-1].x;
			state_d[1] = trt[d][trt[d].size()-1].y;
			int idx = getStateIndex(state_d);				// set true goal as on of possible goals
			g.push_back(idx);
			g_val.push_back(0);
			
			if((int)g.size() != (int)g_val.size()){
				cout << "(1)ERROR: Goal size:" << g.size() << " doesn't match val size:" << g_val.size() << endl;
				exit(1);
			}
			
			if(  action.find("depart") != string::npos || action.find("walk")!= string::npos ){				
				
				setGoalIndicies(g);
				int valsize = g_val.size();
				for(int i=0;i<(int)(g.size()-valsize);i++) g_val.push_back(0);
				
			}
			else if(action.find("approach") != string::npos){
				
				Mat sm; 
				int car_i = 0;
				
				if(FEATURES_ON){
					for(int i=0;i< (int)featurelabel.size();i++){
						if(featurelabel[i]=="car"){
							car_i = i;
						}
					}
					sm = fmap[d][car_i] + 1.0;
				}
				else {
					// load car image
					
					stringstream fn; 
					fn << FILE_ROOT << "imgset_test/feature_maps/" << testList[d] <<"_features.yml";
					if(DEBUG) cout << "Opening: " << fn.str() << endl;
					
					FileStorage fs(fn.str(),FileStorage::READ);
					if(!fs.isOpened()){
						cout << "ERROR:(test) cannot open feature map " << fn.str() << endl;
						exit(1);
					}
					Mat img;
					fs["car"] >> img;							// add feature map (float)
					resize(img,sm,Size(cols,rows),0,0);				// resize, raw probability as a feature					
				}
				
				Mat bin;
				threshold(sm,bin,0.50,255,THRESH_BINARY);
				
				if(VISUALIZE){
					imshow("bin",bin);
					Mat dsp; jetmap(sm,dsp); imshow("dsp",dsp);
					waitKey(1);
				}
				bin.convertTo(bin,CV_8UC1,1.0,0);
				
				vector<vector<Point> > co;
				vector<Vec4i> hi;
				findContours(bin,co,hi,CV_RETR_EXTERNAL,CV_CHAIN_APPROX_SIMPLE);
				
				for(int i=0;i<(int)co.size();i++){
					vector<Point> pts;
					approxPolyDP(Mat(co[i]),pts,1.0,true);
					for(int j=0;j<(int)pts.size();j++){
						//cout << "pts=" << pts[j] << endl;
						state_d[0] = pts[j].x;
						state_d[1] = pts[j].y;
						g.push_back(getStateIndex(state_d));
						g_val.push_back(0);
					}
				}
				
			}
			else{
				cout << "ERROR:" << action << " is an invalid action type for multi-goal inference" << endl;
				exit(1);
			}
			
			cout << "Number of goals:" << (int)g.size() << endl;
			if(DEBUG) cout << "Number of goal vals:" << (int)g_val.size() << endl;
			if((int)g.size() != (int)g_val.size()){
				cout << "ERROR: Goal size:" << g.size() << " doesn't match val size:" << g_val.size() << endl;
				exit(1);
			}
				
			cout << "// ==== SET INITIAL STATE S_0 (true state) ==== //" << endl;

			state_d[0] = trt[d][0].x;
			state_d[1] = trt[d][0].y;
			if(DEBUG) cout << "Inital state S_0 (x:" << state_d[0] << ",y:" << state_d[1] << ")\n";
			if(trt[d][0].x==0 && trt[d][0].y==0){
				cout << "ERROR: Leading zero should have been removed according to ground truth!\n";
				exit(1);
			}
						
			if(OBSERVATION_ON) updateObsFeatureMaps(nf-4,0,d,testList[d]); // Intialize to no observations
			
			
			
			
			cout << "// ==== REVERSE VALUE ITERATION TO COMPUTE GOAL VALUES Z(g|s0) ==== //" << endl;
			vector <int> s0;							// start location
			vector <float> s0_val;						// start value
			s0.push_back(getStateIndex(state_d));		// observation 0
			s0_val.push_back(0);						// start value = 0			
			stat = ValueIteration(d,testList[d],s0,s0_val,logZ0);
			if(stat == -1){
				fll[0] << -FLT_MAX<< "\t";
				fll[1] << -FLT_MAX<< "\t";
				fll[2] << -FLT_MAX<< "\t";
				fll[3] << -FLT_MAX<< "\t";
				fll[4] << -FLT_MAX<< "\t";
				continue;
			}
			
			vector <float> z0_val;
			for(int i=0;i<(int)g.size();i++){
				stateIndexToValue(g[i],state_d);
				z0_val.push_back(logZ0.at<float>(state_d[1],state_d[0]));
			}
			
			if(DEBUG) for(int i=0;i<(int)z0_val.size();i++)
				cout << i << ") Z(g|s_0)=" << z0_val[i] << endl;
			
			
			cout << "// ==== REMOVE GOAL BIAS (Makes all goals initially reachable) ==== //" << endl;
			vector <float> log_p_g;
			for(int i=0;i<(int)g.size();i++) log_p_g.push_back( log(1.0 / g.size()) );	// uniform goal prior
			for(int i=0;i<(int)g.size();i++) log_p_g[i] = - z0_val[i];			// normalize for distance
			if(DEBUG) for(int i=0;i<(int)g.size();i++) cout << i << ") " << log_p_g[i] << endl; 
			
			
			
			
			float stopval = 1.01;					// Evaluate with partial evidence
			if(!OBSERVATION_ON) stopval = 0.05;		// When no evidence (feat, const), evaluate only once with goals
			float step = 0.10;
			if(VISUALIZE) step = 1.0/3.0;
			
			if(VISOUT) step = 1.0/10.0;
			
			
			for(float p = 0.00; p <= stopval; p += step){

				int numobs  = round( p * (trt[d].size()-1));
				
				//if(trt[d][p].x==0 || trt[d][p].y==0 ) continue;
				
				cout << "numobs=" << numobs << " out of " << trt[d].size() << endl; 
				cout << "==================================\n";
				cout << "==== Evidence up to " << numobs << " (p="<< p<<") out of " << trt[d].size() << "(stopval:" << stopval <<")"<<endl;
				cout << "==================================\n";
				if(DEBUG) cout << "// ==== REVERSE VALUE ITERATION FOR GOAL VALUE Z(G|S_0; OBSERVATIONS) ==== //" << endl;
				if(DEBUG) cout << "Current state S_t (x:" << state_d[0] << ",y:" << state_d[1] << ")\n";
				
				vector <int> st;						// current state
				vector <float> st_val;					// value of current state
				
				st = s0;											// no change to current state (it's hidden)
				st_val = s0_val;									// no change to the value either
				if(OBSERVATION_ON) updateObsFeatureMaps(nf-4,numobs,d,testList[d]);	// add evidence Z
								
				stat = ValueIteration(d,testList[d],st,st_val,logZt);	// partition function with evidence
				if(stat == -1){											// if fails to converge
					fll[0] << -FLT_MAX<< "\t";
					fll[1] << -FLT_MAX<< "\t";
					fll[2] << -FLT_MAX<< "\t";
					fll[3] << -FLT_MAX<< "\t";
					fll[4] << -FLT_MAX<< "\t";
					continue;
				}
				
				vector <float> zt_val;
				for(int i=0;i<(int)g.size();i++){
					stateIndexToValue(g[i],state_d);
					if(DEBUG) cout <<"y=" <<state_d[1] << ", x=" << state_d[0] << ") ==>" <<logZt.at<float>(state_d[1],state_d[0]) << endl;
					zt_val.push_back(logZt.at<float>(state_d[1],state_d[0]));		// save values for each goal state
				}
				
				
				
				if(DEBUG) cout << "// ==== COMPUTE GOAL POSTERIOR ===== //" << endl;
				if(DEBUG) cout << "g values" << endl;
				float maxval = -INFINITY;
				for(int i=0;i<(int)g.size();i++){
					//stateIndexToValue(g[i],state_d);
					
					float likelihood = (zt_val[i] - z0_val[i]);
					float prior = log_p_g[i];
					float posterior =  likelihood + prior;
					g_val[i] = posterior;
					if(g_val[i]>maxval) maxval = g_val[i];
					
					if(DEBUG) cout << i << "] zt_val:"<<  zt_val[i] <<" - z0_val:"<< z0_val[i] <<" + prior=" << log_p_g[i] << " ";
					if(DEBUG) cout<< i << ") g_val["<<i<<"]=" << g_val[i] << endl;
					
				}
				
				if(DEBUG) cout << "// ==== RESCALE POSTERIORS ==== //" << endl;
				for(int i=0;i<(int)g_val.size();i++) g_val[i] -= maxval;								//  rescale
				if(DEBUG) for(int i=0;i<(int)g_val.size();i++) cout << "g_val["<<i<<"]=" << g_val[i] << endl;			
				
				
				// ==== FORWARD INFERENCE WITH GOALS ==== //
				stat = ValueIteration(d,testList[d],g,g_val); // forward propagation (policy is computed)
				if(stat == -1){
					fll[0] << -FLT_MAX<< "\t";
					fll[1] << -FLT_MAX<< "\t";
					fll[2] << -FLT_MAX<< "\t";
					fll[3] << -FLT_MAX<< "\t";
					fll[4] << -FLT_MAX<< "\t";
					continue;
				}
				
				
				if(VISUALIZE){
					int s_t = numobs;		// used only for naming files (reset in ComputeExactExpectations)
					int s_0 = 0;
					stat = ComputeExactExpectations(d,testList[d],s_t,s_0,g);				
				}
				
				// ==== COMPUTE SCORES ===== //
				
				int s_t = round(p*100.0);
				cout << "s_t:" << s_t << endl;
				if(OBSERVATION_ON) sampleTrajectories(d,testList[d],g,s_t);
				else sampleTrajectories(d,testList[d]);
				
				ComputeLogLikelihood(d,numobs,val);
				
				float logloss;
				
				logloss = val[0]/(val[3]+val[4]);
				cout << "Log-loss (total):" << logloss << endl;
				fll[0] << logloss << "\t"; fll[0].flush();
				
				logloss = val[1]/(val[3]+val[4]);
				cout << "Log-loss (past):" << logloss << endl;
				fll[1] << logloss << "\t"; fll[1].flush();
				
				logloss = val[2]/(val[3]+val[4]);
				cout << "Log-loss (future):"  << logloss << endl;
				fll[2] << logloss << "\t"; fll[2].flush();
				
				logloss = val[1]/val[3];
				if(isnan(logloss)) logloss = 0;
				cout << "Log-loss(smoothing):"  << logloss << endl;
				fll[3] << logloss << "\t"; fll[3].flush();
				
				logloss = val[2]/val[4];
				if(isnan(logloss)) logloss = 0;
				cout << "Log-loss(forecasting):"  << logloss << endl;
				fll[4] << logloss << "\t"; fll[4].flush();				
			} // for each p
			
			fll[0] << endl;
			fll[1] << endl;
			fll[2] << endl;
			fll[3] << endl;
			fll[4] << endl;
		} // IF multi-goal inference
	}
	
	// ==== Deallocate memory ==== //
	for(int d=0;d<(int)fmap.size();d++) fmap[d].clear(); fmap.clear();
	for(int d=0;d<(int)obt.size();d++) obt[d].clear(); obt.clear();
	for(int d=0;d<(int)trt.size();d++) trt[d].clear(); trt.clear();
	w.clear();
	
}

void trainHIOC::loadParameters(){
	
	// ==== LOAD PARAMETERS ==== //
	if(DEBUG) cout << "Load parameters" << endl;
	stringstream ss;
	ss << FILE_ROOT << "hioc_output/reward-parameters-"<< actionclass << ".txt";
	if(DEBUG) cout << "opening: " << ss.str() << endl;
	
	
	
	ifstream param(ss.str().c_str());
	if(!param.is_open()){
		cout << "ERROR: cannot open " << ss.str() << endl; // when appending
	}
	else {
		param >> nf;
		if(DEBUG) cout <<"number of features: " << nf << endl;
		param >> sample_width;
		if(DEBUG) cout << "sample width: " << sample_width << endl;
		w = vector<float>(nf,0);
		for(int f=0;f<nf;f++) param >> w[f];
		for(int f=0;f<nf;f++) cout << f <<"] "<< w[f] << endl;
		param >> amn[0];
		param >> amn[1];
		param >> amx[0];
		param >> amx[1];
		param >> kernelsize;
		if(DEBUG) cout << "policy dimensions:";
		if(DEBUG) cout << amn[0] << ","<< amn[1] << ","<< amx[0] << ","<< amx[1] << endl;
		if(DEBUG) cout << "kernel size:" << kernelsize << endl;
		if(param>>lambda){
			if(DEBUG) cout << "lambda=" << lambda << endl;
			
			param >> lambda_best;			// step size
			if(DEBUG) cout << "lambda_best=" << lambda_best << endl;
			
			param >> min_loglikelihood;
			if(DEBUG) cout << "min_loglikelihood=" << min_loglikelihood << endl;
			
			w_best = vector<float>(nf,0);
			for(int f=0;f<nf;f++) param >> w_best[f];
			if(DEBUG) cout << "w_best:" << endl;
			if(DEBUG) for(int f=0;f<nf;f++) cout << f <<"] "<< w_best[f] << endl;
			
			f_diff = vector<float>(nf,0);
			for(int f=0;f<nf;f++) param >> f_diff[f];
			if(DEBUG) cout << "f_diff:" << endl;
			if(DEBUG) for(int f=0;f<nf;f++) cout << f <<"] "<< f_diff[f] << endl;

		}
		else{
			cout << "This file has no lambda value. Must be old style." << endl;
		}
		param.close();
	}
}

void trainHIOC::updateActionSpace(){
	
	if(DEBUG) cout <<  "Method:Update action space (kernel size)" << endl;
	int max = 0;
	
	int s_t[ds];
	int s_next[ds];
	int a_t[da];
	
	
	for(int d=0;d<(int)trt.size();d++){
		for(int t=0;t<((int)trt[d].size()-1);t++){
			s_t[0] = trt[d][t].x;
			s_t[1] = trt[d][t].y;
			s_next[0] = trt[d][t+1].x;
			s_next[1] = trt[d][t+1].y;
			a_t[0] = s_next[0]-s_t[0];
			a_t[1] = s_next[1]-s_t[1];
			
			if(a_t[0]<amn[0]) amn[0]=a_t[0];
			if(a_t[0]>amx[0]) amn[0]=a_t[0];
			if(a_t[1]<amn[1]) amn[1]=a_t[1];
			if(a_t[1]>amx[1]) amn[1]=a_t[1];
			
			if(abs(a_t[0])>max) max = abs(a_t[0]);
			if(abs(a_t[1])>max) max = abs(a_t[1]);
		}
	}
	
	if(DEBUG) cout << "  Max action value:" << max << endl;
	
	if(max<=1){
		amn[0] = -1;
		amn[1] = -1;
		amx[0] =  1;
		amx[1] =  1;
		kernelsize = 3;
		
	}
	else if(max<=2){
		amn[0] = -2;
		amn[1] = -2;
		amx[0] =  2;
		amx[1] =  2;
		kernelsize = 5;
		
	}
	else if(max <= 3){
		amn[0] = -3;
		amn[1] = -3;
		amx[0] =  3;
		amx[1] =  3;
		kernelsize = 7;
	}
	else if(max <= 4){
		amn[0] = -4;
		amn[1] = -4;
		amx[0] =  4;
		amx[1] =  4;
		kernelsize = 9;
	}
	else if(max <= 5){
		amn[0] = -5;
		amn[1] = -5;
		amx[0] =  5;
		amx[1] =  5;
		kernelsize = 11;
	}
	else {
		cout << "ERROR: Action " << max << " is too big. Change image size." << endl;
		exit(1);
	}
	
	float actiondims = 1;
	for(int i=0;i<da;i++){
		ba[i] = (amx[i]-amn[i]+1);
		actiondims *= (ba[i]);
	}
	
	na = actiondims;
	
	if(DEBUG) cout << "  Total number of action=" << actiondims << endl;
	cout << "  Kernel size is " << kernelsize << endl;
	
	if(na != kernelsize*kernelsize) cout << "ERROR: size of action space incorrect!" << endl;	
	
}

void trainHIOC::loadTrajectory(int d, string fileid){
	loadTrajectory(d,fileid,0);
}
	
void trainHIOC::loadTrajectory(int d, string fileid, int test){
	
	if(DEBUG) cout << "Method: loadTrajectory" << endl;
	
	float dat[2];
	int k=0;
	
	vector<Point2f> obs;		// observed trajectory
	vector<Point2f> gtt;
	
	// ===== LOAD OBSERVATIONS ===== //
	stringstream os;
	os << FILE_ROOT << "tracks/3D_foot_obs/" << fileid << "_3D_foot_obs.txt";
 	ifstream of(os.str().c_str());
	cout << "Opening:" << os.str() << endl;
	
	k=0;
	while (of >> dat[k++]) {
		if(k==2){
			obs.push_back(Point2f(dat[0],dat[1]));
			k=0;
		}
	}
	if(DEBUG) cout << "  Length of observed trajectory is "<< obs.size() << endl;	
	
	// ===== LOAD GROUND TRUTH ===== //	
	stringstream gs;
	//gs << FILE_ROOT << fileid << "/"<< fileid << "-groundtruth-3D-feet-trajectory.txt";
	gs << FILE_ROOT << "tracks/3D_foot_gt/" << fileid << "_3D_foot_gt.txt";
	ifstream gf(gs.str().c_str());
	cout << "Opening:" << gs.str() << endl;
	
	k=0;
	while (gf >> dat[k++]) {
		if(k==2){
			gtt.push_back(Point2f(dat[0],dat[1]));
			k=0;
		}
	}
	if(DEBUG) cout << "  Length of gt trajectory is "<< gtt.size() << endl;
	
	if((int)gtt.size()==0){
		cout << "ERROR: Size of trajectory is zero!" << endl;
		exit(1);
	}
	
	// ==== REMOVE LEADING AND TRAILING ZEROS ==== //
	k=0;
	for(int i=0;i<(int)gtt.size();i++){
		if( (gtt[i].x<=0) || (gtt[i].y<=0) || (gtt[i].x>=cols) || (gtt[i].y>=rows) ) k++;
		else break;
	}
	if(DEBUG) cout << "  " << k << " leading zeros to be removed (according to ground truth)." << endl;
	start_f = k;
	gtt.erase(gtt.begin(),gtt.begin()+k);
	obs.erase(obs.begin(),obs.begin()+k);
	
	
	
	k=0;
	for(int i=gtt.size()-1;i>=0;i--){
		if( (gtt[i].x<=0) || (gtt[i].y<=0) || (gtt[i].x>=cols) || (gtt[i].y>=rows) ) k++;
		else break;
	}
	if(DEBUG) cout << "  " << k << " ending zeros to be removed." << endl;
	gtt.erase(gtt.end()-k,gtt.end());
	obs.erase(obs.end()-k,obs.end());
	if(gtt.size()!=obs.size()) cout << "ERROR2: GT and OBS lengths do not match!" << endl;

	for(int i=0;i<(int)gtt.size();i++){
		if( (gtt[i].x<=0) || (gtt[i].y<=0) || (gtt[i].x>=cols) || (gtt[i].y>=rows) ){
			cout << "ERROR: Invalid ground truth value: " << gtt[i] << endl;
		}
	}
	
	cout << "  Total length of sequence("<< fileid <<") is: " << gtt.size() << endl;
	obt[d] = obs;
	trt[d] = gtt;
}


void trainHIOC::scaleSpaceTrajectory(int d){
	
	if(DEBUG) cout << "Method: scaleSpaceTrajectory." << endl;
	
	vector<Point2f> gt,ob;
	
	for(int t=0;t<(int)trt[d].size();t++){
		
		if(t==0){
			gt.push_back(trt[d][t]);
			ob.push_back(obt[d][t]);
			continue;
		}
		
		if(trt[d][t].x>=cols || trt[d][t].y>=rows) continue; // do not save, out of bounds
		
		float dx = trt[d][t].x - gt[gt.size()-1].x;		// change in gt.x
		float dy = trt[d][t].y - gt[gt.size()-1].y;		// change in gt.y
		
		//cout << gt[gt.size()-1] << " (" << dx << "," << dy << ")\n";
		
		if(dx==0 && dy==0){
			//cout << "no motion, so skip." << endl;
		}
		else if( fabs(dx)>1 || fabs(dy)>1){
			
			//cout << "interpolate" << endl;
			
			float sx = 1;
			float sy = 1;
			if(dx<0) sx = -1;
			if(dy<0) sy = -1;
			
			int x=1;
			int y=1;
			
			for(;(x<fabs(dx) || y<fabs(dy));x++,y++){
				gt.push_back(Point2f(gt[gt.size()-1].x+sx*x,gt[gt.size()-1].y+sy*y));		// interpolated gt
				ob.push_back(Point2f(0,0));							// no observations
				//cout << gt[gt.size()-1] << " " << ob[ob.size()-1] << endl;
			}
			gt.push_back(trt[d][t]);
			ob.push_back(obt[d][t]);
			//cout << gt[gt.size()-1] << " " << ob[ob.size()-1] << endl;
			
		}
		else{ // valid move (3x3)
			//cout << "valid move." << endl;
			float dist = hypot(dx,dy);
			if(dist>sqrt(2)) cout << "ERROR:" << dx << "," << dy << " is invalid." << dist << endl;
			gt.push_back(trt[d][t]);
			ob.push_back(obt[d][t]);
		}
		
		
	}
	
	obt[d] = ob;
	trt[d] = gt;
	
	if((int)obt[d].size()!=(int)trt[d].size()) cout << "ERROR: incorrect lengths." << endl;
	//cout << "  Spatially rescaled trajectory length is " << (int)trt[d].size() << endl;
	
	
	if(DEBUG) for(int i=0;i<(int)trt[d].size();i++) cout << trt[d][i] << " " << obt[d][i] << endl;
	
	//exit(1); //kk
	
}

void trainHIOC::scaleTimeTrajectory(int d){
	
	if(DEBUG) cout << "Method: scaleTimeTrajectory." << endl;
 	
	int target_length = 1000;
	sample_width = 1;
	
	if((int)trt[d].size()>target_length) sample_width = floor((int)trt[d].size()/target_length);
	else target_length = (int)trt[d].size();
	
	if(DEBUG) cout << "  Sample width=" << sample_width << endl;
	
	vector<Point2f> gt,ob;
	Point2f gp,op;
	
	float s;
	for(int t=0;t<target_length;t++){
		s = 0;
		gp = Point2f(0,0);
		for(int j=(t*sample_width);j<(t*sample_width+sample_width);j++){
			gp += trt[d][j];
			if(trt[d][j].x>0) s++;
		}
		if(s>0) gt.push_back(Point2f(floor(gp.x*cols/s),floor(gp.y*rows/s)));
		else gt.push_back(Point2f(0,0));
		//if(DEBUG) cout << "  " << t << ") gt:" << gt[t] << " ";
		
		s = 0;
		op = Point2f(0,0);
		for(int j=(t*sample_width);j<(t*sample_width+sample_width);j++){
			op += obt[d][j];
			if(obt[d][j].x>0) s++;
		}
		if(s>0) ob.push_back(Point2f(floor(op.x*cols/s),floor(op.y*rows/s)));
		else ob.push_back(Point2f(0,0));
		//if(DEBUG) cout << t << ") ob:" << ob[t] << " ";
		//if(DEBUG) cout << endl;
	}
	
	obt[d] = ob;
	trt[d] = gt;
	if((int)obt[d].size()!=(int)trt[d].size()) cout << "ERROR: incorrect lengths." << endl;
	cout << "  Rescaled length is " << (int)trt[d].size() << endl;
	
}

void trainHIOC::loadFeatureMaps(int d, string fileid){
	
	if(DEBUG) cout << "Method: Load feature maps for " << fileid <<  endl;
	
	stringstream ss;
	Mat dist,dst,im,gnd;
	
	if(VISUALIZE){
		ss << FILE_ROOT << "birdseye/" << fileid << "_birdseye.jpg";
		if(DEBUG) cout << "  Opening " << ss.str() << endl;
		im = imread(ss.str(),1);
		if(!im.data && DEBUG) cout << "ERROR: Could not read " << ss.str() << endl;
		resize(im,gnd,Size(cols,rows),0,0);
		imshow("floor",gnd);
	}
	
	int i=0;
	featurelabel.clear();
	
	vector<Mat> vmat;
	fmap.push_back(vmat);
	
	// ===== CONSTANT ==== //
	featurelabel.push_back("constant");
	fmap[d].push_back(Mat::zeros(rows,cols,CV_32FC1));		// constant feature
	fmap[d][i] -= 1.0;
	
	nf = i+1;
	if(DEBUG) cout << "after adding constant feature, nf=" << nf << endl;
	
	
	if(FEATURES_ON)
	{
		
		stringstream cm_name; 
		cm_name << FILE_ROOT << "imgset_test/parameter_files/colormap/VIRAT_colormap.txt";
		ifstream cm_in(cm_name.str().c_str());
		if(!cm_in.is_open()) cout << "Cannot open:" << cm_name.str() << endl;
		
		vector<string> labels;
		{
			string val;
			int x=0;
			while(cm_in >> val){
				x++;
				if(x==2) labels.push_back(val);
				else if(x==5) x=0;
			}
		}
		
		if(DEBUG){
			cout << "List features" << endl;
			for(int j=0;j<(int)labels.size();j++) cout << labels[j] << endl;
		}
		
		stringstream fn; 
		fn << FILE_ROOT << "imgset_test/feature_maps/" << fileid <<"_features.yml";
		if(DEBUG) cout << "Opening: " << fn.str() << endl;
		
		FileStorage fs(fn.str(),FileStorage::READ);
		
		if(!fs.isOpened()){
			cout << "ERROR: cannot open feature map " << fn.str() << " for " << fileid << endl;
			exit(1);
		}
		
		for(int j=0;j<(int)labels.size();j++){
			
			if(labels[j]=="vobj") continue;
			if(labels[j]=="tree") continue;
			if(labels[j]=="statue") continue;
			if(labels[j]=="cart") continue;
			if(labels[j]=="bike") continue;
			if(labels[j]=="wall") continue;
			
			featurelabel.push_back(labels[j]);				// add label text
			
			if(DEBUG) cout << "Loading fmap for " << labels[j] << endl;
			Mat img;
			fs[labels[j]] >> img;							// add feature map (float)
			Mat sm;
			resize(img,sm,Size(cols,rows),0,0);				// resize, raw probability as a feature ...			
			
			//
			if(labels[j]=="car") erode(sm, sm, Mat(), Point(-1,-1),5,BORDER_CONSTANT);
			
			i++;
			fmap[d].push_back(Mat());
			fmap[d][i] = sm - 1.0;								// invert and make negative
			
			
			if(VISUALIZE){
				jetmap(fmap[d][i],dst);
				addWeighted(dst,0.5,gnd,0.5,0.0,dst);
				imshow(labels[j],dst);
				ss.str("");
				ss << FILE_ROOT << "visual/" << fileid << "_" << labels[j] << "_prob.jpg";
				if(VISOUT) imwrite(ss.str(),dst);
				waitKey(1);
			}
			
			
			// === 'DISTANCE TO' features === //
			
			float thresh = 0.12;
			//if(labels[j]=="fence") thresh = 0.1;
			
			Mat bin;
			threshold(sm,bin,thresh,255,THRESH_BINARY_INV);
			
			//if(labels[j]=="car"){
			//	imshow("carthresh",bin);
			//}
			
			Mat dist;
			bin.convertTo(bin,CV_8UC1,1.0,0.0);
			distanceTransform(bin,dist,CV_DIST_L2, CV_DIST_MASK_PRECISE);
			
			dist /= 255.0;			// normlize [0,1]
			
			Mat dmap[3];
			cv::exp(dist/-0.01,dmap[0]); //
			cv::exp(dist/-0.05,dmap[1]); //
			cv::exp(dist/-0.10,dmap[2]); //
			
			stringstream ss;
			
			ss.str("");
			ss << labels[j] << "_dist0";
			i++;
			featurelabel.push_back(ss.str());
			fmap[d].push_back(Mat());
			fmap[d][i] = dmap[0]-1.0;
			
			ss.str("");
			ss << labels[j] << "_dist1";
			i++;
			featurelabel.push_back(ss.str());
			fmap[d].push_back(Mat());
			fmap[d][i] = dmap[1]-1.0;
			
			ss.str("");
			ss << labels[j] << "_dist2";
			i++;
			featurelabel.push_back(ss.str());
			fmap[d].push_back(Mat());
			fmap[d][i] = dmap[2]-1.0;
			
			if(VISUALIZE){
				for(int k=0;k<3;k++){
					jetmap(dmap[k],dst);
					addWeighted(dst,0.5,gnd,0.5,0.0,dst);
					stringstream title;
					title << labels[j] << "-distance-" << k;
					imshow(title.str(),dst);
					ss.str("");
					ss << FILE_ROOT << "visual/" << fileid << "_" << title.str() << ".jpg";
					if(VISOUT) imwrite(ss.str(),dst);
					waitKey(1);
				}
			}
		} 
		
		waitKey(1);
				
		nf = i+1;	// one-based conversion ('i' is zero-based)
	}
	
	if(VISUALIZE) waitKey(1);
	
	// ==== observations ==== //
	if(OBSERVATION_ON){
		fmap[d].push_back(fmap[d][0]*0); featurelabel.push_back("observation 1");
		fmap[d].push_back(fmap[d][0]*0); featurelabel.push_back("observation 2");
		fmap[d].push_back(fmap[d][0]*0); featurelabel.push_back("observation 3");
		
		updateObsFeatureMaps(i, (int)trt[d].size(), d, fileid);
		nf = i+4;
	}
	
	if(DEBUG) cout << "Number of features:" << nf << endl;
	
}

// ORIGINAL VERSION

//void trainHIOC::loadFeatureMaps(int d, string fileid){
//	
//	if(DEBUG) cout << "Method: Load feature maps for " << fileid <<  endl;
//	
//	stringstream ss;
//	Mat dist,dst,im,gnd;
//	
//	if(VISUALIZE){
//		ss << FILE_ROOT << fileid << "/"<< fileid << "-3Dfloor-f0.bmp";
//		im = imread(ss.str(),1);
//		if(DEBUG) cout << "  Opening " << ss.str() << endl;
//		resize(im,gnd,Size(cols,rows),0,0);
//		imshow("floor",gnd);
//	}
//	
//	int i=0;
//	featurelabel.clear();
//	
//	vector<Mat> vmat;
//	fmap.push_back(vmat);
//	
//	// ===== CONSTANT ==== //
//	featurelabel.push_back("constant");
//	fmap[d].push_back(Mat::zeros(rows,cols,CV_32FC1));		// constant feature
//	fmap[d][i] -= 1.0;
//	
//	nf = i+1;
//	if(DEBUG) cout << "after adding constant feature, nf=" << nf << endl;
//	
//	
//	if(FEATURES_ON)
//	{
//		// ===== PAVEMENT ==== //
//		i++;
//		featurelabel.push_back("pavement");
//		ss.str("");
//		ss << FILE_ROOT << fileid << "/"<< fileid << "-feature-pavement-response.bmp";
//		im = imread(ss.str(),0);
//		fmap[d].push_back(im);
//		resize(im,fmap[d][i],Size(cols,rows),0,0);
//		medianBlur(fmap[d][i],im,5);
//		
//		GaussianBlur(im,im,Size(101,101),0,0,BORDER_CONSTANT);
//		// blurmap(im,im);
//		
//		//imshow("pavement",im);
//		//waitKey(0);
//		
//		im.convertTo(fmap[d][i],CV_32FC1,1.0/255.0,0);
//		normalize(fmap[d][i],fmap[d][i],0,1.0,NORM_MINMAX,-1);
//		fmap[d][i] -= 1.0;
//		
//		if(VISUALIZE){
//			jetmap(fmap[d][i],dst);
//			addWeighted(dst,0.5,gnd,0.5,0.0,dst);
//			imshow("pavement response(101)",dst);
//			ss.str("");
//			ss << FILE_ROOT << fileid << "/"<< fileid << "-feature-pavement-blur.jpg";
//			imwrite(ss.str(),dst);
//			waitKey(1);
//		}
//		
//		// ===== NOT PAVEMENT ==== //
//		i++;
//		featurelabel.push_back("not pavement");
//		ss.str("");
//		ss << FILE_ROOT << fileid << "/"<< fileid << "-feature-pavement-response.bmp";
//		im = imread(ss.str(),0);
//		fmap[d].push_back(im);
//		resize(im,fmap[d][i],Size(cols,rows),0,0);
//		medianBlur(fmap[d][i],im,5);
//		GaussianBlur(im,im,Size(101,101),0,0,BORDER_CONSTANT);
//		im.convertTo(fmap[d][i],CV_32FC1,1.0/255.0,0);
//		normalize(fmap[d][i],fmap[d][i],0,1.0,NORM_MINMAX,-1);
//		fmap[d][i] = -fmap[d][i]; // negate and invert
//		
//		if(VISUALIZE){
//			jetmap(fmap[d][i],dst);
//			addWeighted(dst,0.5,gnd,0.5,0.0,dst);
//			imshow("not pavement response(101)",dst);
//			ss.str("");
//			ss << FILE_ROOT << fileid << "/"<< fileid << "-feature-not-pavement-blur.jpg";
//			imwrite(ss.str(),dst);
//			waitKey(1);
//		}
//		
//		
//		// ===== CAR ==== //
//		i++;
//		featurelabel.push_back("car");
//		ss.str("");
//		ss << FILE_ROOT << fileid << "/"<< fileid << "-feature-car-response.bmp";
//		im = imread(ss.str(),0);	
//		fmap[d].push_back(im);
//		resize(im,fmap[d][i],Size(cols,rows),0,0);
//		medianBlur(fmap[d][i],im,5);
//		//GaussianBlur(im,im,Size(51,51),0,0,BORDER_CONSTANT);
//		for(int j=0;j<500;j++){
//			GaussianBlur(im,im,Size(3,3),0,0,BORDER_CONSTANT);
//		}
//		im.convertTo(fmap[d][i],CV_32FC1,1.0,0);
//		normalize(fmap[d][i],fmap[d][i],0,1.0,NORM_MINMAX,-1);
//		fmap[d][i] -= 1.0;
//		if(VISUALIZE){
//			jetmap(fmap[d][i],dst);
//			addWeighted(dst,0.5,gnd,0.5,0.0,dst);
//			imshow("car response (51)",dst);
//			ss.str("");
//			ss << FILE_ROOT << fileid << "/"<< fileid << "-feature-car-blur.jpg";
//			imwrite(ss.str(),dst);
//			waitKey(1);	
//		}
//		
//		// ===== NOT CAR (101) ==== //
//		i++;
//		featurelabel.push_back("not car");
//		ss.str("");
//		ss << FILE_ROOT << fileid << "/"<< fileid << "-feature-car-response.bmp";
//		im = imread(ss.str(),0);	
//		fmap[d].push_back(im);
//		resize(im,fmap[d][i],Size(cols,rows),0,0);
//		medianBlur(fmap[d][i],im,5);
//		GaussianBlur(im,im,Size(51,51),0,0,BORDER_CONSTANT);
//		im.convertTo(fmap[d][i],CV_32FC1,1.0/255.0,0);
//		normalize(fmap[d][i],fmap[d][i],0,1.0,NORM_MINMAX,-1);
//		fmap[d][i] = - fmap[d][i]; // invert
//		if(VISUALIZE){
//			jetmap(fmap[d][i],dst);
//			addWeighted(dst,0.5,gnd,0.5,0.0,dst);
//			imshow("not car response (51)",dst);
//			ss.str("");
//			ss << FILE_ROOT << fileid << "/"<< fileid << "-feature-not-car-blur.jpg";
//			imwrite(ss.str(),dst);
//			waitKey(1);	
//		}
//		
//		
//		
//		// ===== GRASS (101) ==== //
//		i++;
//		featurelabel.push_back("grass");
//		ss.str("");
//		ss << FILE_ROOT << fileid << "/"<< fileid << "-feature-grass-response.bmp";
//		im = imread(ss.str(),0);
//		fmap[d].push_back(im);
//		resize(im,fmap[d][i],Size(cols,rows),0,0);
//		medianBlur(fmap[d][i],im,5);
//		GaussianBlur(im,im,Size(101,101),0,0,BORDER_CONSTANT);
//		im.convertTo(fmap[d][i],CV_32FC1,1.0/255.0,0);
//		normalize(fmap[d][i],fmap[d][i],0,1.0,NORM_MINMAX,-1);
//		fmap[d][i] -= 1.0;
//		if(VISUALIZE){
//			jetmap(fmap[d][i],dst);
//			addWeighted(dst,0.5,gnd,0.5,0.0,dst);
//			imshow("grass response",dst);
//			ss.str("");
//			ss << FILE_ROOT << fileid << "/"<< fileid << "-feature-grass-blur.jpg";
//			imwrite(ss.str(),dst);
//			waitKey(1);
//		}
//		
//		
//		// ===== NOT GRASS ==== //
//		i++;
//		featurelabel.push_back("not grass");
//		ss.str("");
//		ss << FILE_ROOT << fileid << "/"<< fileid << "-feature-grass-response.bmp";
//		im = imread(ss.str(),0);
//		fmap[d].push_back(im);
//		resize(im,fmap[d][i],Size(cols,rows),0,0);
//		medianBlur(fmap[d][i],im,5);		
//		GaussianBlur(im,im,Size(101,101),0,0,BORDER_CONSTANT);
//		im.convertTo(fmap[d][i],CV_32FC1,1.0/255.0,0);
//		normalize(fmap[d][i],fmap[d][i],0,1.0,NORM_MINMAX,-1);
//		fmap[d][i] = - fmap[d][i]; // negate and invert
//		if(VISUALIZE){
//			jetmap(fmap[d][i],dst);
//			addWeighted(dst,0.5,gnd,0.5,0.0,dst);
//			imshow("not grass response (301)",dst);
//			ss.str("");
//			ss << FILE_ROOT << fileid << "/"<< fileid << "-feature-not-grass-blur.jpg";
//			imwrite(ss.str(),dst);
//			waitKey(1);
//		}
//		
//		
//		// ==== BUILDING ===== //
//		i++;
//		featurelabel.push_back("building");
//		ss.str("");
//		ss << FILE_ROOT << fileid << "/"<< fileid << "-feature-building-response.bmp";
//		im = imread(ss.str(),0);	
//		fmap[d].push_back(im);
//		resize(im,fmap[d][i],Size(cols,rows),0,0);
//		medianBlur(fmap[d][i],im,5);
//		//GaussianBlur(im,im,Size(101,101),0,0,BORDER_CONSTANT);
//		for(int j=0;j<50;j++){
//			GaussianBlur(im,im,Size(11,11),0,0,BORDER_CONSTANT);
//		}
//		im.convertTo(fmap[d][i],CV_32FC1,1.0/255.0,0);
//		normalize(fmap[d][i],fmap[d][i],0,1.0,NORM_MINMAX,-1);
//		fmap[d][i] -= 1.0;
//		if(VISUALIZE){
//			jetmap(fmap[d][i],dst);
//			addWeighted(dst,0.5,gnd,0.5,0.0,dst);
//			imshow("building response(101)",dst);
//			ss.str("");
//			ss << FILE_ROOT << fileid << "/"<< fileid << "-feature-building-blur.jpg";
//			imwrite(ss.str(),dst);
//			waitKey(1);
//		}
//		
//		
//		// ==== NOT BUILDING ===== //
//		i++;
//		featurelabel.push_back("not building");
//		ss.str("");
//		ss << FILE_ROOT << fileid << "/"<< fileid << "-feature-building-response.bmp";
//		im = imread(ss.str(),0);
//		fmap[d].push_back(im);
//		resize(im,fmap[d][i],Size(cols,rows),0,0);
//		medianBlur(fmap[d][i],im,5);
//		GaussianBlur(im,im,Size(101,101),0,0,BORDER_CONSTANT);
//		im.convertTo(fmap[d][i],CV_32FC1,1.0/255.0,0);
//		normalize(fmap[d][i],fmap[d][i],0,1.0,NORM_MINMAX,-1);
//		fmap[d][i] = -fmap[d][i];
//		if(VISUALIZE){
//			jetmap(fmap[d][i],dst);
//			addWeighted(dst,0.5,gnd,0.5,0.0,dst);
//			imshow("not building response (301)",dst);
//			ss.str("");
//			ss << FILE_ROOT << fileid << "/"<< fileid << "-feature-not-building-blur.jpg";
//			imwrite(ss.str(),dst);
//			waitKey(1);
//		}
//		
//		nf = i+1;
//	}
//	
//	if(VISUALIZE) waitKey(1);
//	
//	// ==== observations ==== //
//	if(OBSERVATION_ON){
//		fmap[d].push_back(fmap[d][0]*0); featurelabel.push_back("observation 1");
//		fmap[d].push_back(fmap[d][0]*0); featurelabel.push_back("observation 2");
//		fmap[d].push_back(fmap[d][0]*0); featurelabel.push_back("observation 3");
//		
//		updateObsFeatureMaps(i, (int)trt[d].size(), d, fileid);
//		nf = i+4;
//	}
//	
//	if(DEBUG) cout << "Number of features:" << nf << endl;
//	
//}

void trainHIOC::updateObsFeatureMaps(int i, int tau, int d, string fileid){
	
	if(DEBUG) cout << "Method:updateObsFeatureMaps" << endl;
	if(DEBUG) if(tau<(int)trt[d].size()) cout << "Using partial evidence:" << tau << " of " << (int)trt[d].size() << endl;
	Mat dst,gnd,im;
	stringstream ss;
	
	if(VISUALIZE){
		ss << FILE_ROOT << "birdseye/" << fileid << "_birdseye.jpg";
		if(DEBUG) cout << "read image:" << ss.str() << endl;
		im = imread(ss.str(),1);
		resize(im,gnd,Size(cols,rows),0,0);
		imshow("floor",gnd);
	}
	
	Mat obsmap = fmap[d][0]*0;
	
	if(OBSERVATION_ON){
		if(DEBUG) cout << "Load observations" << endl;
		for(int t=0;t<tau;t++)
			if( obt[d][t].x>0 && obt[d][t].y>0 ) obsmap.at<float>(obt[d][t].y,obt[d][t].x) = 1.0; // set to one, do no add
	}
	
	//if(tau==-1) obsmap = 1.0;
	
	if(VISOUT){
		if(DEBUG) cout << "Write observation map" << endl;
		Mat omap;
		omap = gnd*1.0;
		for(int t=0;t<tau;t++)
			if( obt[d][t].x>0 && obt[d][t].y>0 ){
				//circle(omap,Point(obt[d][t].x,obt[d][t].y),1,CV_RGB(255,255,0),-1,CV_AA);
				omap.at<Vec3b>(obt[d][t].y,obt[d][t].x) = Vec3b(0,255,255);
			}
		imshow("obs",omap);
		ss.str("");
		ss << FILE_ROOT << "visual/" << actionclass << "-"<< fileid<< "-observation.jpg";
		imwrite(ss.str().c_str(),omap);
		waitKey(1);

	}
	
	// === sigma:11 === //
	i++;
	GaussianBlur(obsmap,fmap[d][i],Size(11,11),0,0,BORDER_CONSTANT);
	normalize(fmap[d][i],fmap[d][i],0,1.0,NORM_MINMAX,-1);
	fmap[d][i] -= 1.0;
	if(VISUALIZE){
		jetmap(fmap[d][i],dst);
		addWeighted(dst,0.5,gnd,0.5,0.0,dst);
		imshow("observation response (11)",dst);
		ss.str("");
		ss << FILE_ROOT << "visual/" << fileid << "-feature-observation-11-"<<tau<<".jpg";
		imwrite(ss.str(),dst);
		waitKey(1);
	}
	
	
	// === sigma:21 === //
	i++;
	GaussianBlur(obsmap,fmap[d][i],Size(21,21),0,0,BORDER_CONSTANT);
	normalize(fmap[d][i],fmap[d][i],0,1.0,NORM_MINMAX,-1);
	fmap[d][i] -= 1.0;
	if(VISUALIZE){
		jetmap(fmap[d][i],dst);
		addWeighted(dst,0.5,gnd,0.5,0.0,dst);
		imshow("observation response (21)",dst);
		ss.str("");
		ss << FILE_ROOT << "visual/" << fileid << "-feature-observation-21-"<<tau<<".jpg";
		imwrite(ss.str(),dst);
		waitKey(1);
	}
	
	// === sigma:31 === //
	i++;
	GaussianBlur(obsmap,fmap[d][i],Size(31,31),0,0,BORDER_CONSTANT);
	normalize(fmap[d][i],fmap[d][i],0,1.0,NORM_MINMAX,-1);
	fmap[d][i] -= 1.0;
	if(VISUALIZE){
		jetmap(fmap[d][i],dst);
		addWeighted(dst,0.5,gnd,0.5,0.0,dst);
		imshow("observation response (31)",dst);
		ss.str("");
		ss << FILE_ROOT << "visual/" << fileid << "-feature-observation-31-"<<tau<<".jpg";
		imwrite(ss.str(),dst);
		waitKey(1);
	}
}

void trainHIOC::getFeatures(Point pt, vector<float> &feats,int d){
	for(int f=0;f<nf;f++) feats[f] = fmap[d][f].at<float>(pt.y,pt.x);
}

void trainHIOC::computeEmpiricalStatistics(int d){
	if(DEBUG) cout << "Method: Compute empirical statistics." << endl;
	vector<float> feats(nf,0);
	empirical_f.push_back(feats);
	//empirical_f[d] = feats;	
	for(int t=0;t<(int)trt[d].size();t++){
		getFeatures(trt[d][t],feats,d); 
		if(DEBUG) cout << "  features= ";
		if(DEBUG) for(int f=0;f<nf;f++) printf("%f ", feats[f]); 
		if(DEBUG) cout << endl;
		for(int f=0;f<nf;f++) empirical_f[d][f] += feats[f];
	}
	if(DEBUG) cout << "  EMPIRICAL feature counts ...\n"; 
	if(DEBUG) for(int f=0;f<nf;f++) cout << empirical_f[d][f] << " "; 
	if(DEBUG) cout << endl;
}

// overloaded function
int trainHIOC::ValueIteration(int d,string fileid, vector<int> goals,vector<float> vals){
	Mat logZ;
	return ValueIteration(d,fileid,-1,goals,vals,logZ);
}

// overloaded function
int trainHIOC::ValueIteration(int iteration, int d,string fileid, int x_goal){
	vector<int> goals;
	vector<float> vals;
	Mat logZ;
	return ValueIteration(d,fileid,x_goal,goals,vals, logZ);
}

// overloaded function
int trainHIOC::ValueIteration(int d,string fileid, vector<int> goals,vector<float> vals, Mat &logZ){
	return ValueIteration(d,fileid,-1,goals,vals,logZ);
}

// overloaded function
int trainHIOC::ValueIteration(int d,string fileid, int x_goal, vector<int> goals,vector<float> vals, Mat &logZ){
		
	cout << "Method: Value Iteration" << endl;
	
	Mat LZ = Mat::ones(rows,cols,CV_32FC1) * -FLT_MAX;
	Mat R = Mat::zeros(rows,cols,CV_32FC1);
	Mat dst;
	Mat F = Mat::zeros(rows,cols,CV_32FC1);
	
	for(int f=0;f<nf;f++) R += w[f]*fmap[d][f];

	Mat gnd;
	
	if(VISOUT || VISUALIZE)
	{
		if(DEBUG) cout << "Visualize ground" << endl;
		stringstream ss;
		ss.str("");
		ss << FILE_ROOT << "birdseye/" << fileid << "_birdseye.jpg";
		Mat im = imread(ss.str(),1);
		resize(im,gnd,Size(cols,rows),0,0);
		if(VISUALIZE) imshow("floor",gnd);
		
		Mat Rc;
		if(DEBUG) cout << "Visualize reward" << endl;
		jetmap(R,Rc);
		if(VISUALIZE) imshow("Reward",Rc);
		if(VISOUT) ss.str("");
		if(VISOUT) ss << FILE_ROOT << "visual/" << actionclass <<"-"<< fileid << "-reward.jpg";
		if(VISOUT) imwrite(ss.str().c_str(),Rc);
		
		if(DEBUG) cout << "Visualize blend reward" << endl;
		addWeighted(Rc,0.5,gnd,0.5,0.0,dst);
		imshow("blend reward",dst);
		
		if(VISOUT) ss.str("");
		if(VISOUT) ss << FILE_ROOT << "visual/" << actionclass <<"-"<< fileid << "-reward-blend.jpg";	
		if(VISOUT) imwrite(ss.str(),dst);
		
		// === image plane reward function === //
		Mat H = getH();		
		//cout << H << endl;

		ss.str("");
		ss << FILE_ROOT << "imgset_test/jpg/" << fileid << ".jpg";
		Mat img = imread(ss.str(),1);
		if(!img.data){
			cout << "ERROR: Cannot open image: " << ss.str() << endl;
			exit(1);
			
		}
		Mat sm;
		resize(img,sm,Size(cols,rows),0,0);
		
		Mat imR = sm*1.0;
		warpPerspective(Rc,imR,H.inv(),Rc.size(),INTER_CUBIC,BORDER_CONSTANT);
		//imshow("imR",imR);
		
		addWeighted(imR,0.5,sm,0.5,0.0,dst);
		if(VISUALIZE) imshow("2d reward",dst);
		
		if(VISOUT) ss.str("");
		if(VISOUT) ss << FILE_ROOT << "visual/" << actionclass <<"-"<< fileid << "-reward-blend-2D.jpg";	
		if(VISOUT) imwrite(ss.str(),dst);
		
		waitKey(1);
	}
	
	
	cout << "Initialize start and goal states" << endl;
	
	int state_d[ds];
	
	if((int)goals.size()!=(int)vals.size()){
		cout << "ERROR: number of goals and values do not match." << goals.size() << "!=" << vals.size() << endl;
		exit(1);
	}
	
	if(x_goal < 0 && (int)goals.size()==0){
		if(DEBUG) cout << "No goals specified. Use single ground truth goal state." << endl;
		state_d[0]  = trt[d][(int)trt[d].size()-1].x;
		state_d[1]  = trt[d][(int)trt[d].size()-1].y;
		x_goal = getStateIndex(state_d);
		LZ.at<float>(state_d[1],state_d[0]) = 0.0;		// zero goal state
		
	}
	else if (x_goal > -1 && (int)goals.size()==0){
		if(DEBUG) cout << "Single goal specified. Use single user defined goal" << endl;
		stateIndexToValue(x_goal,state_d);
		LZ.at<float>(state_d[1],state_d[0]) = 0.0;		// zero goal state
	}
	else if((int)goals.size()>0) {
		if(DEBUG) cout << "Multiple or single goal(s) specified! (only used for goal inference)" << endl;
		x_goal = goals[0];
		for(int i=0;i<(int)goals.size();i++){
			stateIndexToValue(goals[i],state_d);
			LZ.at<float>(state_d[1],state_d[0]) = vals[i];		// zero goal state
		}
	}
	else{
		cout << "ERROR: Invalid." << endl;
	}
	
	if(DEBUG && (int)goals.size()==0) cout << "x_goal:" << x_goal << " (" << state_d[0] << "," << state_d[1] << ")\n";
	
	if(state_d[0] >= cols || state_d[1]>=rows){
		cout << "ERROR: Goal is out of bounds!" << endl;
		cout << "x_goal:" << x_goal << " (" << state_d[0] << "," << state_d[1] << ")\n";
		exit(1);
	}
	cout << "Value Iteration" << endl;
	Mat res;
	double minVal, maxVal;
	int n=0;
	while(1)
	{
		
		int w = kernelsize;
		int h = kernelsize;
		Mat padLZ;
		Mat lz = LZ * 1.0;
		copyMakeBorder(lz,padLZ,floor(h*0.5),floor(h*0.5),floor(w*0.5),floor(w*0.5),BORDER_CONSTANT,Scalar::all(-FLT_MAX));
		padLZ *= 1.0;
		
		for(int col=0;col< (padLZ.cols-w+1);col++){
			for(int row=0;row< (padLZ.rows-h+1);row++){
				Rect r(col,row,w,h);
				Mat sub = padLZ(r);
				
				minMaxLoc(sub,&minVal,&maxVal,NULL,NULL);
				if(minVal==-FLT_MAX && maxVal==-FLT_MAX) continue;
				
				for(int y=0;y<w;y++){
					for(int x=0;x<h;x++){
						if( DISCRETE && y==1 && x==1) continue;
						
						minVal = MIN(LZ.at<float>(row,col),sub.at<float>(y,x));
						maxVal = MAX(LZ.at<float>(row,col),sub.at<float>(y,x));
						
						if(maxVal<=-FLT_MAX) LZ.at<float>(row,col) = -FLT_MAX;
						else if(maxVal == 0){
							LZ.at<float>(row,col) = 0;
						}
						else{
							float k = SOFTMAXVAL;
							float softmax = maxVal + (1./k)*log(1 + exp(k*minVal - k*maxVal));
							
							if( softmax>0){
								
								if(DEBUG){cout << "soft V:" <<  softmax <<" min:" << minVal << " max:" << maxVal << endl;
									cout << "sub:" << sub << endl;
									cout << "exp(k*minVal - k*maxVal):" << exp(k*minVal - k*maxVal) << endl;
									cout << "log(1 + exp(k*minVal - k*maxVal)):" << log(1 + exp(k*minVal - k*maxVal)) << endl;
									cout << "LZ.at<float>(row,col)" <<  LZ.at<float>(row,col) << endl;
									cout << "sub.at<float>(y,x)" << sub.at<float>(y,x) << endl;
								}
								//return -1;
							}
							
							LZ.at<float>(row,col) = softmax;
						}
					}
				}
				
				LZ.at<float>(row,col) += R.at<float>(row,col);
				
				if(LZ.at<float>(row,col)>0){
					cout << "\nERROR (return -1): Value function cannot be positive. Value function cannot converge. " << LZ.at<float>(row,col) << " " << row << "," << col << endl;
					return -1;
				}
			}
		}
		
		
		if((int)goals.size()>0){ // multiple final states with values
			for(int i=0;i<(int)goals.size();i++){
				stateIndexToValue(goals[i],state_d);
				LZ.at<float>(state_d[1],state_d[0]) = vals[i];
			}
		}
		else{
			LZ.at<float>(state_d[1],state_d[0]) = 0.0; // set goal to 0
		}
		
		absdiff(LZ,F,res);
		minMaxLoc(res,&minVal,&maxVal,NULL,NULL);
		if(maxVal<0.9) break;
		if(DEBUG) if(maxVal< 1.0) cout << "maxVal=" << maxVal << endl;
		
		LZ.copyTo(F);
		
		
		if(VISUALIZE||VISOUT){
			//exp(LZ,dst);
			jetmap(LZ,dst);
			addWeighted(gnd,0.5,dst,0.5,0,dst);
			if(VISUALIZE) imshow("Value Function",dst);
			waitKey(1);
		}
		
		n++;
		
		if(n%10==0) cout << n << " ";
		if(n>1000){ 
			cout << "ERROR: Max number of iterations." << endl;
			return -1;
		}
		

		
	}
	cout << endl;
	
	LZ.copyTo(logZ);
	
	if(VISOUT){
		stringstream ss;
		ss << FILE_ROOT << "visual/" << actionclass << "-" << fileid << "-valuefunction.jpg";
		imwrite(ss.str().c_str(),dst);
	}
	
	// ===== compute policy ===== //
	
	if((int)goals.size()==0 || (int)goals.size()>1){ // only when there is a single goal or multiple goals
		
		if(DEBUG) cout << "  Computing policy" << endl;
		
		int w = kernelsize;
		int h = kernelsize;
		Mat padLZ;
		copyMakeBorder(LZ,padLZ,floor(h*0.5),floor(h*0.5),floor(w*0.5),floor(w*0.5),BORDER_CONSTANT,Scalar(-INFINITY));
		
		for(int col=0;col<=padLZ.cols-h;col++){
			for(int row=0;row<=padLZ.rows-w;row++){
				state_d[0]=col;
				state_d[1]=row;
				int x = getStateIndex(state_d);
				Rect r(col,row,w,h);
				Mat sub = padLZ(r);
				minMaxLoc(sub,&minVal,&maxVal,NULL,NULL);
				Mat p = sub - maxVal;				// rescaling
				exp(p,p);							// Z(x,a)
				Scalar su = sum(p);					// sum (denominator)
				if(su.val[0]>0) p /= su.val[0];		// normalize (compute policy(x|a))
				else p = 1.0/na;					// uniform distribution
				p = p.reshape(1,1);					// vectorize
				for(int a=0;a<na;a++) pax[a][x] = p.at<float>(0,a); // update policy
				//if(x==x_goal) for(int a=0;a<na;a++) pax[a][x] = 0; // never propagated during forward pass			
			}
		}
	}
	
	return 1;
}

void trainHIOC::saveParameters(){
	stringstream ss;
	ss << FILE_ROOT << "hioc_output/reward-parameters-"<< actionclass << ".txt";
	ofstream param(ss.str().c_str());
	param << nf << endl;
	param << sample_width << endl;
	for(int f=0;f<nf;f++) param << w[f] << endl;
	param << amn[0] << endl;
	param << amn[1] << endl;
	param << amx[0] << endl;
	param << amx[1] << endl;
	param << kernelsize << endl;
	param << lambda << endl;
	param << lambda_best << endl;			// step size
	param << min_loglikelihood << endl;
	for(int f=0;f<nf;f++) param << w_best[f] << endl;
	for(int f=0;f<nf;f++) param << f_diff[f] << endl;
	param.close();
}





void trainHIOC::init(int cols, int rows){
	
	if(DEBUG) cout << "Method:Initialize" << endl;
	
	this->cols = cols;
	this->rows = rows;
	this->ds = 2;
	this->da = 2;
	this->ba = new int [2];
	this->bs = new int [2];
	
	smx = new float [ds];
	smn = new float [ds];
	smn[0] = 0;
	smx[0] = cols-1;
	smn[1] = 0;
	smx[1] = rows-1;
	
	amx = new float [da];
	amn = new float [da];
	for(int i=0;i<da;i++) amx[i] =  1;
	for(int i=0;i<da;i++) amn[i] = -1;
	
	kernelsize = 3;
	
	float statedims = 1;
	for(int i=0;i<ds;i++){
		bs[i] = (smx[i]-smn[i]+1); // since max is inclusive
		statedims *= (bs[i]);
	}
	if(DEBUG) cout << "  Total number of states=" << statedims << endl;
	
	float actiondims = 1;
	for(int i=0;i<da;i++){
		ba[i] = (amx[i]-amn[i]+1);
		actiondims *= (ba[i]);
	}
	if(DEBUG) cout << "  Total number of action=" << actiondims << endl;
	
	ns = statedims;
	na = actiondims;
	
}

void trainHIOC::actionIndexToValue(int action_index, int* action_d){
	int icols = ba[0];
	action_d[0] = action_index % icols + amn[0];
	action_d[1] = floor(action_index*1.0/icols) + amn[1];	
}

void trainHIOC::stateIndexToValue(int state_index, int* state_d){
	int icols = bs[0];
	state_d[0] = state_index % icols + smn[0];
	state_d[1] = floor(state_index*1.0/icols) + smn[1];	
}

float trainHIOC::ComputeLogLikelihood(int d, int tau, float *val){
	
	
	//////////////////////////////
	//		Log-likelihood		//
	//////////////////////////////
	
	float loglikelihood = 0;
	float smoothing_likelihood = 0;
	float forecasting_likelihood = 0;
	
	int s_t[ds];
	int s_next[ds];
	int a_t[da];
	int k[2];
	k[0]=0;
	k[1]=0;
	
	for(int t=0;t<(int)trt[d].size()-1;t++){
		
		s_t[0] = trt[d][t].x;
		s_t[1] = trt[d][t].y;
		s_next[0] = trt[d][t+1].x;
		s_next[1] = trt[d][t+1].y;
		a_t[0] = s_next[0]-s_t[0];
		a_t[1] = s_next[1]-s_t[1];
		
		int a = getActionIndex(a_t);
		int x = getStateIndex(s_t);
		float pas;
		
		if(a<0) pas = FLT_MIN; // out of current range
		else pas = pax[a][x];
		
		if(pas<=FLT_MIN){
			cout << "    (t="<< t << ") +++++++++ too small" << endl;
			pas = FLT_MIN;
		}
		
		pas = log(pas);
		
		
		loglikelihood += pas;
		if(t<tau){
			smoothing_likelihood += pas;
			k[0]++;
		}
		else{
			forecasting_likelihood += pas;
			k[1]++;
		}
		
	}
	
	val[0] = loglikelihood;
	val[1] = smoothing_likelihood;
	val[2] = forecasting_likelihood;	
	val[3] = k[0];
	val[4] = k[1];
	
	//cout << "\n****************** LIKELIHOOD ************************\n";
	cout << "  " << d << "] loglikelihood: " << loglikelihood << "/" << (k[0]+k[1])<< endl;
	cout << "  " << d << "] smoothing loglikelihood: " << smoothing_likelihood << "/" << k[0] << endl;
	cout << "  " << d << "] forecasting loglikelihood: " << forecasting_likelihood << "/" << k[1] << endl;	
	//cout << "\n******************************************************\n";
	
	return loglikelihood;
}

//double trainHIOC::ComputeObservedLogLikelihood(int d, int tau){
//	
//	// compute likelihood for d-th trajectory
//	// partial observations up to 'tau'
//	
//	//////////////////////////////
//	//		Log-likelihood		//
//	//////////////////////////////
//	
//	double loglikelihood = 0;
//	
//	int s_t[ds];
//	int s_next[ds];
//	int a_t[da];
//	
//	for(int t=0;t<tau;t++){
//		
//		s_t[0] = obt[d][t].x;
//		s_t[1] = obt[d][t].y;
//		s_next[0] = obt[d][t+1].x;
//		s_next[1] = obt[d][t+1].y;
//		a_t[0] = s_next[0]-s_t[0];
//		a_t[1] = s_next[1]-s_t[1];
//		
//		if(s_t[0]==0 || s_t[1]==0 || s_next[0]==0 || s_next[1]==0) continue;
//		
//		int a = getActionIndex(a_t);
//		int x = getStateIndex(s_t);
//		double pas;
//		
//		if(a<0 || x<0 || a>=na || x>=ns) pas = FLT_MIN; // out of current range
//		else pas = pax[a][x];
//		
//		if(pas<=FLT_MIN){
//			cout << t << ") +++++++++ use small value." << endl;
//			pas = FLT_MIN;
//		}
//		
//		pas = log(pas);
//		
//		loglikelihood += pas;
//		
//	}
//	
//	cout << "  [" << d << "] Observed loglikelihood: " << loglikelihood << endl;
//	
//	return loglikelihood;
//}

void trainHIOC::jetmap(Mat _src, Mat &dst)
{
	if(_src.type()!=CV_32FC1) cout << "ERROR(jetmap): must be single channel float\n";
	double minVal,maxVal;
	Mat src;
	_src.copyTo(src);
	Mat isInf;
	minMaxLoc(src,&minVal,&maxVal,NULL,NULL);
	//cout << "min:" << minVal << " max:" << maxVal << endl;
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

void trainHIOC::jetmapProb(Mat _src, Mat &dst)
{
	if(_src.type()!=CV_32FC1) cout << "ERROR(jetmap): must be single channel float\n";
	double minVal,maxVal;
	Mat src;
	_src.copyTo(src);
	Mat isInf;
	minVal = 1e-4; // 1e-4
	maxVal = 0.1;  // 0.2
	if(MARKOV) maxVal = 0.5;
	threshold(src,src,minVal,0,THRESH_TOZERO);		// 0 if less than minVal
	compare(src,0,isInf,CMP_GT);					// 255 if  true (greater than '0')
	Mat im = (src-minVal)/(maxVal-minVal) * 255.0;	// negative values will be ignored...
	//threshold(src,src,255,255,THRESH_TRUNC);		// truncate at 255
	Mat U8,I3[3],hsv;
	im.convertTo(U8,CV_8UC1,1.0,0);					// negative values set to zero?
	I3[0] = U8*1.0;
	
	Mat pU;
	U8.convertTo(pU,CV_64F,1.0,0);
	pU = (255.0-pU)/255.0;
	pow(pU,0.5,pU);					// 0.25 smaller: white area is smaller, larger: white area bigger, smoother
	pU *= 255.0;
	pU.convertTo(U8,CV_8UC1,1.0,0);
	I3[1] = U8*1.0;
	
	I3[2] = isInf;
	merge(I3,3,hsv);
	cvtColor(hsv,dst,CV_HSV2RGB_FULL);
	
	//imshow("S",I3[1]);	
	//imshow("V",I3[2]);
	//waitKey(1);
	//minMaxLoc(src,&minVal,&maxVal,NULL,NULL);
	//compare(src,FLT_MIN,isInf,CMP_GT);					// 255 if  true (greater than '0')
	//threshold(src,src,FLT_MIN,0,THRESH_TOZERO);			//
	//minMaxLoc(src,&minVal,NULL,NULL,NULL);
	//cout << "min:" << minVal << " max:" << maxVal << endl;
	//minVal = 0;
}

void trainHIOC::jetmapAbs(Mat _src, Mat &dst)
{
	if(_src.type()!=CV_32FC1) cout << "ERROR(jetmap): must be single channel float\n";
	double minVal,maxVal;
	Mat src;
	_src.copyTo(src);
	Mat isInf;
	compare(src,-88.0,isInf,CMP_GT);			// -88 min value that opencv log returns ?
	threshold(src,src,-88.0,0,THRESH_TOZERO);
	minMaxLoc(src,&minVal,&maxVal,NULL,NULL);
	maxVal = 0;
	Mat im = (src-minVal)/(maxVal-minVal) * 255.0;
	Mat U8,I3[3],hsv;
	im.convertTo(U8,CV_8UC1,1.0,0);
	I3[0] = U8 * 0.80;
	I3[1] = isInf;
	I3[2] = isInf;
	merge(I3,3,hsv);
	cvtColor(hsv,dst,CV_HSV2RGB_FULL);
}

void trainHIOC::sampleTrajectories(int d,string fileid){
	vector <int> goals;
	int s_d[ds];
	s_d[0] = trt[d][(int)trt[d].size()-1].x; 
	s_d[1] = trt[d][(int)trt[d].size()-1].y;
	goals.push_back(getStateIndex(s_d));
	int s_t = -1;
	sampleTrajectories(d, fileid, goals,s_t);
}

void trainHIOC::sampleTrajectories(int d,string fileid, vector <int> goals, int s_t){
	
	if(DEBUG) cout << "Method: sampleTrajectories" << endl;
	
	
	// ==== INITIALIZE START AND GOAL ==== //
	
	vector <vector <Point> > samples;
	srand(0);
	
	vector <int> stepSize;
	vector <float> hausDist;
	vector <float> eucDist;
	
	int numReachGoal = 0;
	
	Mat dsp;
	Mat gnd;
	
	if(VISUALIZE || VISOUT){
		stringstream ss;
		ss << FILE_ROOT << "birdseye/" << fileid << "_birdseye.jpg";
		Mat im = imread(ss.str(),1);
		if(DEBUG) cout << "  Opening " << ss.str() << endl;
		resize(im,gnd,Size(cols,rows),0,0);
		if(VISUALIZE) imshow("floor",gnd);
		gnd.copyTo(dsp);
	}
	
	if(VISOUT){		// ==== OUTPUT GROUND TRUTH ==== //
		Mat gtpath = gnd * 1.0; 
		for(int i=0;i< (int)trt[d].size();i++){
			circle(gtpath,Point(trt[d][i].x,trt[d][i].y),1,CV_RGB(0,255,0),-1,CV_AA);
		}
		stringstream ss;
		ss << FILE_ROOT << "visual/" << actionclass << "-" << fileid << "-GT.jpg";
		imwrite(ss.str().c_str(),gtpath);
	}
	
	
	// ===== SAMPLING TRAJECTORIES ===== //
	int s_d[ds];
	int a_d[da];
	int n=0;
	int length_max = round(trt[d].size() * 10.0);

	while(1){
		
		n++;
		
		s_d[0] = trt[d][0].x; 
		s_d[1] = trt[d][0].y;
		
		int steps = 0;
		
		Scalar color = Scalar(rand()%255,rand()%255,rand()%255);
		vector<Point> traj;
		
		for(int i=0;i<length_max;i++){
			
			float r = rand()/(float)RAND_MAX; // 0 to 1.0
			//cout << i << ") r=" << r << endl;
			
			Point p = Point(s_d[0],s_d[1]);
			traj.push_back(p);
			
			if(VISUALIZE||VISOUT) dsp.at<Vec3b>(p.y,p.x) = Vec3b(color.val[0],color.val[1],color.val[2]);
			
			int idx = getStateIndex(s_d);
				
			for(int j=0;j<(int)goals.size();j++){
				if(idx==goals[j]){
					numReachGoal++; // end trajectory if goal is reached
					break;
				}
			}
			
			// ===== SAMPLE ACTION ===== //
			int ida = ns-1;
			float sum = 0;
			for(int a=0;a<na;a++){
				sum += pax[a][idx];
				//cout << a <<") " << sum << endl;
				if( sum >=r ){
					ida = a;
					//cout << "Take action: " << ida << endl;
					break;
				}
			}
			
			if(ida<0){
				cout << "ERROR: Action was not chosen! End trajectory." << endl;
				break;
			}
			
			actionIndexToValue(ida,a_d);

			//cout << "action= (" << a_d[0]	<< "," << a_d[1] << ")" << endl;

			s_d[0] += a_d[0];
			s_d[1] += a_d[1];
			
			//cout << "next state= (" << s_d[0]	<< "," << s_d[1] << ")" << endl;
			
			if(!isValidState(s_d)){
				//cout << "ERROR: Not a valid next state! End trajectory." << endl;
				break;
			}
			
			steps = i; // keep incomplete paths (allow distance function to show termination success...)
		}

		if(steps==-1) continue;
		// only compute statistics when the end is reached ...
		
		// ==== Compute distance to ground truth ==== //
		//cout << "steps:" << steps << endl;
		stepSize.push_back(steps);
		
		
		int min_length = MIN((int)trt[d].size(), (int)traj.size());
		

		if(1){ // EUCLIDEAN DISTANCE (min length match)

			float dist_total = 0;
			float dist_norm = 0;

			for(int i=0;i<min_length;i++){
				float dx = trt[d][i].x - traj[i].x;
				float dy = trt[d][i].y - traj[i].y;
				dist_total += hypot(dx,dy);
			}
			dist_norm = dist_total / min_length;
			eucDist.push_back(dist_norm);
			//cout << "Euclidean distance: " << dist_norm << endl;
			
		}
		

		
		if(1){ // MODIFIED HAUSDORF DISTANCE?

			float dist_total = 0;
			float dist_norm = 0;
			float dist = 0;
			int win_size = 30;
			int k = 0;
			
			for(int i=0;i<min_length;i++)
			{
				float dist_min = FLT_MAX;
				
				for(int j=i-win_size;j<=i+win_size;j++)
				{
					if( j<0 || j >= min_length) continue;
					
					float dx = trt[d][j].x - traj[i].x;
					float dy = trt[d][j].y - traj[i].y;
					dist = hypot(dx,dy);
					
					if(dist<dist_min) dist_min = dist;
				}
				if(dist_min < FLT_MAX){
					dist_total += dist_min;
					k++;
				}
			}
			
			dist_norm = dist_total / (float)k;			
			hausDist.push_back(dist_norm);
			//cout << "Hausdorff distance: " << dist_norm << endl;
		}
		
		if(eucDist[eucDist.size()-1] < hausDist[hausDist.size()-1]){
			cout << "euc should be smaller!" << endl;
			exit(1);
		}
		
		//cout << "dist_total:" << dist_total << endl;
		//cout << "dist_norm:" << dist_norm << endl;
		//cout << n << ") dist_mean:" << dist_mean << endl;
		
		
		if(VISUALIZE){
			imshow("dsp",dsp);
			int c = waitKey(1);
			if(c==27) break;
		}
		
		if(n==500) break;
		
	} // while loop
	

	if(VISOUT){
		stringstream ss;
		ss << FILE_ROOT << "visual/" << actionclass << "-" <<fileid <<"-sample.jpg";
		Point start, end;
		end = Point(trt[d][(int)trt[d].size()-1].x,trt[d][(int)trt[d].size()-1].y);
		start = Point(trt[d][0].x,trt[d][0].y);
		
		circle(dsp,start,8,CV_RGB(0,0,0),-1,CV_AA);
		circle(dsp,start,6,CV_RGB(255,255,255),-1,CV_AA);

		circle(dsp,end,8,CV_RGB(255,255,255),-1,CV_AA);
		circle(dsp,end,6,CV_RGB(0,0,0),-1,CV_AA);
		imwrite(ss.str().c_str(),dsp);
	}
	
	float mean;
	float stdev;
	
	float val[8];
	
	val[0] = (float)trt[d].size(); // true trajectory length
	val[1] = numReachGoal;
	
	mean = 0;
	stdev = 0;
	for(int i=0;i<(int)stepSize.size();i++) mean += stepSize[i];
	mean /= (float)stepSize.size();
	for(int i=0;i<(int)stepSize.size();i++) stdev += (stepSize[i]-mean)*(stepSize[i]-mean);
	stdev = sqrt(stdev/(float)stepSize.size());
	val[2] = mean;  // mean
	val[3] = stdev;  // stdev
	
	
	mean = 0;
	stdev = 0;
	for(int i=0;i<(int)eucDist.size();i++) mean += eucDist[i];
	mean /= (float)eucDist.size();
	for(int i=0;i<(int)eucDist.size();i++) stdev += (eucDist[i]-mean)*(eucDist[i]-mean);
	stdev = sqrt(stdev/(float)eucDist.size());
	val[4] = mean;  // mean
	val[5] = stdev;  // stdev

	
	mean = 0;
	stdev = 0;
	for(int i=0;i<(int)hausDist.size();i++) mean += hausDist[i];
	mean /= (float)hausDist.size();
	for(int i=0;i<(int)hausDist.size();i++) stdev += (hausDist[i]-mean)*(hausDist[i]-mean);
	stdev = sqrt(stdev/(float)hausDist.size());
	val[6] = mean;  // mean
	val[7] = stdev;  // stdev
	
	
	if(!fdist[0].is_open()){
		stringstream ss;
		if(MULTI_GOAL_ON) ss << FILE_ROOT << "hioc_output/test-results-eucdist-"<< actionclass << "-multigoal-" << trainList.size() <<"-"<< (int)testList.size() << ".txt";
		else ss << FILE_ROOT << "hioc_output/test-results-eucdist-"<< actionclass << "-" << trainList.size() <<"-"<< (int)testList.size() << ".txt";
		fdist[0].open(ss.str().c_str());
		if(!fdist[0].is_open()){ 
			cout << "ERROR: cannot open file " <<  ss.str() << endl;
			exit(1);
		}
	}

	if(!fdist[1].is_open()){
		stringstream ss;
		if(MULTI_GOAL_ON) ss << FILE_ROOT << "hioc_output/test-results-hausdist-"<< actionclass << "-multigoal-" << trainList.size() <<"-"<< (int)testList.size() << ".txt";
		else ss << FILE_ROOT << "hioc_output/test-results-hausdist-"<< actionclass << "-" << trainList.size() <<"-"<< (int)testList.size() << ".txt";
		fdist[1].open(ss.str().c_str());
		if(!fdist[1].is_open()){ 
			cout << "ERROR: cannot open file " <<  ss.str() << endl;
			exit(1);
		}
	}
	
	//cout << "s_t is " << s_t << endl;
	
	if(s_t==-1){
		fdist[0] << actionclass << "\t" <<fileid << "\t" << val[4] << endl; fdist[0].flush(); // Euclidean distance
		fdist[1] << actionclass << "\t" <<fileid << "\t" << val[6] << endl; fdist[1].flush(); // Hausdorf distance
	}
	else if(s_t==0){
		fdist[0] << actionclass << "\t" <<fileid << "\t" << val[4]; fdist[0].flush(); // Euclidean distance
		fdist[1] << actionclass << "\t" <<fileid << "\t" << val[6]; fdist[1].flush(); // Hausdorf distance
	}
	else if(s_t>=100){
		fdist[0] << "\t" << val[4]<< endl;
		fdist[1] << "\t" << val[6]<< endl;
	}
	else {
		fdist[0] << "\t" << val[4]; fdist[0].flush();// Euclidean distance
		fdist[1] << "\t" << val[6]; fdist[1].flush();// Hausdorf distance
	}
	
	return;
	
} 

int trainHIOC::ComputeExactExpectations(int d, string fileid, int st, int s0, vector <int> goals){
	return ComputeExactExpectations(d,fileid,-1, st, s0, goals);
}
	
// overloading
int trainHIOC::ComputeExactExpectations(int tau, int d,string fileid, int x_goal, int st){
	vector <int> goals;
	int s0 = st;			// initial state and current start state is the same
	return ComputeExactExpectations( d, fileid, x_goal, st, s0, goals);
}

// overloading
int trainHIOC::ComputeExactExpectations(int d, string fileid, int x_goal, int st, int s0, vector <int> goals){
	
	cout << "Method: Compute Exact Expection " << d <<  endl;
	cout << "Number of goals:" << goals.size() << endl;
	cout << "Start state (st):" << st << endl;
	int s_d[ds];
	
	// added this for visualization...
	int index = st;
	st = 0;

	////////////////////
	//      START     //
	////////////////////

//	if(st>=0){ // current state, first observed
//		int k=0;
//		int nonzero = 0;
//		
//		while( k<=st || !nonzero){
//			
//			if( obt[d][k].x>0 && obt[d][k].y>0) // first one is guaranteeed to be non-zero for testing
//			{
//				s_d[0] = obt[d][k].x;		// store last non-zero entry
//				s_d[1] = obt[d][k].y;
//				nonzero = 1;
//			}			
//			k++;
//			//cout << k << endl;
//		}
//		
//	}
//	else if(st == -1){	// first ground truth
		s_d[0] = trt[d][0].x;
		s_d[1] = trt[d][0].y;
//	}
//	else {
//		// using the t-th step of the trajectory...
//		cout << "ERROR: invalid index." << endl;
//	}

	int x_start = getStateIndex(s_d);
	
	if(DEBUG) cout << "x_start=" << x_start << " (" << s_d[0]	<< "," << s_d[1] << ")" << endl;
	
	
	////////////////////
	//      GOAL      //
	////////////////////
	
	if(x_goal==-1 && goals.size()==0){
		if(DEBUG) cout << "Use ground truth goal" << endl;
		s_d[0] = trt[d][(int)trt[d].size()-1].x; s_d[1] = trt[d][(int)trt[d].size()-1].y;
		x_goal = getStateIndex(s_d);	
	}
	else if( x_goal>-1 && goals.size()==0) {
		stateIndexToValue(x_goal,s_d);			// use x_goal value...
	}
	else if(goals.size()>0){
		// multiple end goals!
		cout << "multiple goals!" << endl;
	}
	
	if(DEBUG) cout << "x_goal=" << x_goal << " (" << s_d[0]	<< "," << s_d[1] << ")" << endl;
	
	
	////////////////////
	//    VARIABLES   //
	////////////////////
	
	vector <float> px(ns,0);
	vector <float> _px(ns,0);
	
	px[x_start] = 1.0;
	
	Mat V;
	Mat sumV;
		
	vector<float> feats(nf,0);
	expected_f = feats;		// zero out feature counts
	
	Mat gnd;
	Mat dst;
	Mat dstIm;
	Mat H;
	Mat img;
	
	if(VISUALIZE || VISOUT){
		
		stringstream ss;
		ss << FILE_ROOT << "birdseye/" << fileid << "_birdseye.jpg";
		Mat im = imread(ss.str(),1);
		if(DEBUG) cout << "  Opening " << ss.str() << endl;
		resize(im,gnd,Size(cols,rows),0,0);
		if(VISUALIZE) imshow("floor",gnd);
		H = getH();
		
		ss.str("");
		ss << FILE_ROOT << "imgset_test/jpg/" << fileid << ".jpg";
		im = imread(ss.str(),1);
		resize(im,img,Size(cols,rows),0,0);		
		
	}
	
	VideoWriter avi;
	
	int t = -1;
	while(1)
	{	
		t++;
		
		//if(DEBUG) cout << t << endl;
		if((int)goals.size()==0){
			for(int x=0;x<ns;x++){
				if(px[x]==0) continue;
				stateIndexToValue(x,s_d);
				getFeatures(Point(s_d[0],s_d[1]),feats,d);
				for(int f=0;f<nf;f++) expected_f[f] += feats[f]*px[x];
			}		
		}
		
		if(VISUALIZE || VISOUT){
			
			Mat_<float> _N(1,ns,CV_32FC1);
			for(int x=0;x<ns;x++) _N(0,x) = px[x];//N[t][x]; // probability
			V = _N.reshape(bs[1]) * 1.0;
			
			if(!sumV.data) V.copyTo(sumV);
			else sumV += V;

			jetmapProb(sumV,dst);
			
			
			if(st>s0)
				for(int i=0;i<st;i++){
				//for(int i=0;i<obt[d].size();i++)
					if(obt[d][i].x >0 && obt[d][i].y >0) dst.at<Vec3b>(obt[d][i].y,obt[d][i].x) = Vec3b(255,255,255);
				}
			
			
			warpPerspective(dst,dstIm,H.inv(),dst.size(),INTER_CUBIC,BORDER_CONSTANT);

			addWeighted(dst,0.5,gnd,0.5,0.0,dst);
			
			addWeighted(dstIm,0.3,img,0.7,0.0,dstIm);
			
			// draw observations
			//if(obt[d][i].x >0 && obt[d][i].y >0) circle(dst,Point(obt[d][i].x,obt[d][i].y),1,Scalar::all(255),-1,CV_AA);
			
			stringstream ss;
			
			if(VISOUT && VISUALIZE){
				if(!avi.isOpened()){
					ss.str("");
					ss << FILE_ROOT << "visual/" << actionclass << "_"<< fileid<<"_"<< index << "_sum-expectation.avi";
					avi.open(ss.str().c_str(),CV_FOURCC('X', 'V', 'I', 'D'),30,dst.size(),true);
				}
				avi << dstIm;
			}
			
			if(VISUALIZE) imshow("N",dst);
			if(VISUALIZE) imshow("Nim",dstIm);
			waitKey(1);
		}

		
		_px = vector<float>(ns,0);
		
		int is_goal;
		
		for(int x=0;x<ns;x++){
			
			if(px[x]<=FLT_MIN) continue;				// skip zero prob states (do not propagate)
			else if(x==x_goal && goals.size()==0) continue;				// skip goal state (do not propagate)
			
			//is_goal = 0;
			//for(int i=0;i<(int)goals.size();i++){
			//	if(x==goals[i]){
			//		cout << "hit." << goals[i] << endl;
			//		is_goal = 1;
			//	}
			//}
			//if(is_goal==1) continue;					// skip (multiple) goal state, do not propagate
			
			
			for(int a=0;a<na;a++){
				
				int next_state = getNextState(x,a);		// this is slow ?
				if(next_state==-1) continue;
				if(next_state==x_goal && (int)goals.size()==0) continue;		// absorption
				
				is_goal = 0;
				for(int i=0;i<(int)goals.size();i++){
					if(next_state==goals[i]){
						//cout << "next state is goal:" << goals[i] << endl;
						is_goal = 1;
					}
				}
				if(is_goal==1) continue;					// multi-goal absorption, do not increment
				
				_px[next_state] += (px[x] * pax[a][x]);
				//cout <<  px[x] << "*"<< pax[a][x] << "==>" << _px[next_state] << endl;
			}
		}
		
		
		
		for(int x=0;x<ns;x++) px[x] = _px[x];
		
		
		// compute sum
		
		float sum = 0;
		for(int x=0;x<ns;x++){
			
			//is_goal = 0;
			//for(int i=0;i<(int)goals.size();i++){
			//	if(x==goals[i]) is_goal = 1;
			//}
			//if(is_goal==1) px[x]=0;						// overwrite goal state with 0
			
			if(px[x] < FLT_MIN) px[x] = 0;				// truncate values
			//else if(x==x_goal) px[x] = 0;				// absorption state (needed?)
			else sum += px[x];							// sum ...
		}
		
		//cout << "sum=" << sum << endl;
		if(sum<0.1) break;
		if(t%10==0) cout <<t<< " ";
		if(t>2000) return -1; // does not converge!
		
		if(MARKOV && t> ((int)trt[d].size()*2) ) break;
		
	}
	cout << endl;
	
	if(VISOUT){
		stringstream ss;
		ss << FILE_ROOT << "visual/" << actionclass << "_"<< fileid<< "_"<< index <<"_sum_expectation.jpg";
		imwrite(ss.str().c_str(),dstIm);
	}
	
	if(VISOUT && VISUALIZE){
		if(avi_cum.isOpened()) avi_cum << dstIm; 
	}
		
	return 0;
}




int trainHIOC::getNextState(int x,int a){
	int state_d[ds];
	int action_d[da];
	stateIndexToValue(x,state_d);
	actionIndexToValue(a,action_d);
	state_d[0] += action_d[0];
	state_d[1] += action_d[1];
	if(isValidState(state_d)) return getStateIndex(state_d);
	else return -1;
}

int trainHIOC::isValidState(int *state_d){
	int valid = 1;
	if( state_d[0]<smn[0] || state_d[0]>smx[0] || state_d[1]<smn[1] || state_d[1]>smx[1]) valid=0;
	return valid;
}

int trainHIOC::getStateIndex(int *s_d){
	return (s_d[1]-smn[1])*bs[0] + (s_d[0]-smn[0]);	
}

int trainHIOC::getActionIndex(int *a_d){
	int action_index = (a_d[1]-amn[1])*ba[0] + (a_d[0]-amn[0]);
	if( a_d[0]<amn[0] || a_d[0]>amx[0] || a_d[1]<amn[1] || a_d[1]>amx[1]){
		cout << "ERROR: action is out of bounds!" << endl;
		cout << a_d[0] << " " << a_d[1] << endl; 
		action_index = -1;
	}
	return action_index;	
}




