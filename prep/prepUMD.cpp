/*
 *  prepVirat.cpp
 *  hioc-virat
 *
 *  Created by Kris Kitani on 5/14/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */

#include "prepUMD.hpp"

void prepUMD::prepare_trajectory_features()
{
	if((int)_basename.size()==0) cerr << "ERROR: Load basenames first.\n";

	cout << "PROBLEM HERE: trajectories need to be scaled first to image size, then normalized and rescale?" << endl;


	cout << "\n---------------------------------------------\n";
	cout << "Normalize trajectories to constant velocity";
	cout << "\n---------------------------------------------\n";

	for(int i=0;i<(int)_basename.size();i++)
	{

		ss.str("");
		ss << _basename[i] + "_" << _t_start[i] << "_" << _t_end[i] << "_" << _act[i];
		string fileid = ss.str();

		float dat[2];
		int k=0;
		ifstream fs;
		vector<Point2f> obs;		// observed trajectory
		vector<Point2f> gtt;


		////////////////////////////////
		// OBSERVED TRAJECTORY
		//
		ss.str("");
		ss << _root + "/trajectories_obs/" + fileid + "_3D_foot_obs.txt";
		fs.open(ss.str().c_str());
		k=0;
		while (fs >> dat[k++]) {
			if(k==2){
				obs.push_back(Point2f(dat[0],dat[1]));
				k=0;
			}
		}
		//cout << "  Length of observed trajectory is "<< obs.size() << endl;
		fs.close();


		////////////////////////////////
		// TRUE TRAJECTORY
		//
		ss.str("");
		ss << _root + "/trajectories_gt/" + fileid + "_3D_foot_gt.txt";
		fs.open(ss.str().c_str());
		k=0;
		while (fs >> dat[k++]) {
			if(k==2){
				gtt.push_back(Point2f(dat[0],dat[1]));
				k=0;
			}
		}
		//cout << "  Length of gt trajectory is "<< gtt.size() << endl;
		fs.close();


		/////////////////////////////////////////////////
		// REMOVE LEADING ZEROS
		//
		k=0;
		for(int i=0;i<(int)gtt.size();i++){
			if( (gtt[i].x<=0) || (gtt[i].y<=0) || (gtt[i].x>=_nc) || (gtt[i].y>=_nr) ) k++;
			else break;
		}
		//cout << "  " << k << " leading zeros to be removed (according to ground truth)." << endl;
		//start_f = k;
		gtt.erase(gtt.begin(),gtt.begin()+k);
		obs.erase(obs.begin(),obs.begin()+k);


		/////////////////////////////////////////////////
		// REMOVE TRAILING ZEROS
		//
		k=0;
		for(int i=gtt.size()-1;i>=0;i--){
			if( (gtt[i].x<=0) || (gtt[i].y<=0) || (gtt[i].x>=_nc) || (gtt[i].y>=_nr) ) k++;
			else break;
		}
		//cout << "  " << k << " ending zeros to be removed." << endl;
		gtt.erase(gtt.end()-k,gtt.end());
		obs.erase(obs.end()-k,obs.end());

		if(gtt.size()!=obs.size()) cout << "ERROR2: GT and OBS lengths do not match!" << endl;

		///////////////////////////////////////////////
		// REMOVE OUT OF BOUNDS AND STATIONARY MOTION

		k = 0;

		for(int t=0;t<(int)gtt.size();t++)
		{
			if(gtt[t].x>= 1.0) continue;				// reject gt, out of bounds
			if(gtt[t].y>= 1.0) continue;				// reject gt, out of bounds

			float dx,dy;

			if(k>0)	// remove stationary motion at target grid resolution
			{
				dx = floor(gtt[t].x*_nc) - floor(gtt[k-1].x*_nc);	// change in gt.x
				dy = floor(gtt[t].y*_nr) - floor(gtt[k-1].y*_nr);	// change in gt.y

				if(dx==0 && dy==0) continue;						// remove stationary motion

				if(dx>1.0 || dy>1.0) cout << "ERROR: Interpolation not handled by preprocessing\n";
				if(dx>1.0 || dy>1.0) exit(1);
			}

			gtt[k] = gtt[t];
			obs[k] = obs[t];
			k++;
		}

		gtt.resize(k);
		obs.resize(k);

		/////////////////////////////////////////////////
		// WRITE NORMALIZED TRAJECTORIES TO FILE
		//
		ofstream ofs;

		ss.str("");
		ss << "mkdir -p " + _root + "/trajectories/" + fileid;
		system(ss.str().c_str());

		ss.str("");
		ss << _root + "/trajectories/" + fileid + "/traj_gt.txt";
		ofs.open(ss.str().c_str());
		for(int i=0;i<(int)gtt.size();i++) ofs << floor(gtt[i].x*_nc) << " " << floor(gtt[i].y*_nr) << endl;
		ofs.close();

		ss.str("");
		ss << _root + "/trajectories/" + fileid + "/traj_obs.txt";
		ofs.open(ss.str().c_str());
		for(int i=0;i<(int)obs.size();i++) ofs << floor(obs[i].x*_nc) << " " << floor(obs[i].y*_nr) << endl;
		ofs.close();

	}

}

void prepUMD::prepare_static_features()
{

	if((int)_basename.size()==0) cerr << "ERROR: Load basenames first.\n";

	cout << "\n---------------------\n";
	cout << "Load/prepare semantic features";
	cout << "\n---------------------\n";

	ss.str("");
	ss << _root +  "/data_params/colormap.txt";	// parameter file from semantic scene labeling

	ifstream ifs(ss.str().c_str());
	if(!ifs.is_open()) cerr << "ERROR: Opening " << ss.str() << endl;

	vector<string> semantic_labels;
	string val[5];
	int d = 0;
	while(ifs>>val[d++])
	{
		if(d==5) semantic_labels.push_back(val[1].c_str());
		if(d==5) d=0;
	}

	cout << "\n--------------------------------\n";
	cout << "Output semantic features as XML";
	cout << "\n--------------------------------\n";


	for(int i=0;i<(int)_basename.size();i++)
	{

		ss.str("");
		ss << _basename[i] + "_" << _t_start[i] << "_" << _t_end[i] << "_" << _act[i];
		string fileid = ss.str();

		ss.str("");
		ss << _root + "/ioc_demo/transfer_feat/" + fileid + "_features.yml";
		cout << "Opening: " << ss.str() << endl;

		FileStorage fs(ss.str(),FileStorage::READ);

    ss.str("");
    ss << _root + "/ioc_demo/transfer_feat/" + fileid + "_feature_maps.xml";
    FileStorage fsw(ss.str(),FileStorage::WRITE);

    Mat im, tmp, sm;
    fs[semantic_labels[0]] >> tmp;

    set(_root, tmp.size().width, tmp.size().height);

    Mat constnt(_nr, _nc, CV_32FC1, -1.0);
    fsw << "feature_0" << constnt;

    int k = 1;

		for(int j=0;j<(int)semantic_labels.size();j++)
		{
			if(semantic_labels[j]=="vobj")	continue;
			if(semantic_labels[j]=="tree")	continue;
			if(semantic_labels[j]=="statue")continue;
			if(semantic_labels[j]=="cart")	continue;
			if(semantic_labels[j]=="bike")	continue;
			if(semantic_labels[j]=="wall")	continue;

			fs[semantic_labels[j]] >> im;						// add feature map (float)

			Mat b = im>0;
			b.convertTo(b,CV_32FC1,1./255);						// set negative values to zero
			im = im.mul(b);

			resize(im,sm,Size(_nc,_nr));						// linear interpolation

			if(semantic_labels[j]=="car") erode(sm, sm, Mat(), Point(-1,-1),5,BORDER_CONSTANT);

      string feat = "feature_" + to_string(k);
      fsw << feat << (sm - 1);
      k++;

			float thresh = 0.12;
			Mat bin;
			threshold(sm,bin,thresh,1.0,THRESH_BINARY_INV);

			bin.convertTo(bin,CV_8UC1);

			Mat dist;

#ifdef __linux
			distanceTransform(bin,dist,CV_DIST_L2, CV_DIST_MASK_PRECISE);
#elif __APPLE__
			distanceTransform(bin,dist,DIST_L2, DIST_MASK_PRECISE);
#endif

			dist /= 255.0;

			Mat dmap[3];
			cv::exp(dist/-0.01,dmap[0]);						// exponentiate Euclidean distance
			cv::exp(dist/-0.05,dmap[1]);
			cv::exp(dist/-0.10,dmap[2]);

      feat = "feature_" + to_string(k);
      fsw << feat << (dmap[0] - 1);
      k++;

      feat = "feature_" + to_string(k);
      fsw << feat << (dmap[1] - 1);
      k++;

      feat = "feature_" + to_string(k);
      fsw << feat << (dmap[2] - 1);
      k++;
		}
	}


}

void prepUMD::set(string root, int nc, int nr)
{
	_root = root;
	_nc = nc;
	_nr = nr;
}

void prepUMD::load_basenames(string vidseg_info_fp)
{

	cout << "\n--------------------------------\n";
	cout << "Load master file of basenames";
	cout << "\n--------------------------------\n";

	ifstream ifs( (_root+vidseg_info_fp).c_str());
	cout << "Opening " << (_root+vidseg_info_fp) << endl;
	if(!ifs.is_open()) cerr << "ERROR\n";

	cout << "Assumed format: [NAME] [START] [END] [ACTION_CODE(0-2)]\n";

	string val[4];
	int d = 0;
	while(ifs >> val[d++])
	{
		if(d==4)
		{
			d=0;
			if(val[0].find("%")!=string::npos) continue;
			_basename.push_back	(val[0]);
			_t_start.push_back	(atoi(val[1].c_str()));
			_t_end.push_back	(atoi(val[2].c_str()));
			_act.push_back		(atoi(val[3].c_str()));

		}
	}

	for(int i=0;i<(int)_basename.size();i++) cout << _basename[i] << " " << _t_start[i] << " " << _t_end[i] << " " << _act[i] << endl;

}

void prepUMD::colormap(Mat src, Mat &dst, int do_norm)
{

	Mat im;
	im = src.clone();

	if(do_norm)
	{
		double minVal,maxVal;
		minMaxLoc(src,&minVal,&maxVal,NULL,NULL);
		im = (src-minVal)/(maxVal-minVal);						// normalization [0 to 1]
		printf("min:%f max:%f\n",minVal,maxVal);				//
	}

	// my HSV mapping with masking
	Mat mask;
	mask = Mat::ones(src.size(),CV_8UC1)*255.0;

	//compare(im,FLT_MIN,mask,CMP_GT);						// one color values greater than X

	Mat U8;
	im.convertTo(U8,CV_8UC1,255,0);

	Mat I3[3],hsv;
	I3[0] = U8 * 0.85;
	I3[1] = mask;
	I3[2] = mask;
	merge(I3,3,hsv);
	cvtColor(hsv,dst,COLOR_HSV2RGB_FULL);

}

void prepUMD::flt2img(Mat src, Mat &img)
{
	// assume that incoming flt is normalized from 0 to 1
	Mat flt;
	src.convertTo(flt,CV_32FC1,768,0);							// quantize

	Mat_<Vec3b> s = Mat::zeros(flt.size(),CV_8UC3);

	for(int r=0;r<flt.rows;r++)
	{
		for(int c=0;c<flt.cols;c++)
		{
			float val = flt.at<float>(r,c);
			s(r,c)[0] = (int)MIN(255,val);
			if(val>=256) s(r,c)[1] = (int)MIN(255,val-256);
			if(val>=512) s(r,c)[2] = (int)MIN(255,val-512);
		}
	}
	img = s.clone();
}
