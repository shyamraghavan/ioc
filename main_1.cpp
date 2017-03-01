#include <iostream>
#include "ccp.hpp"
using namespace std;

int main (int argc, char * const argv[])
{

	string basenames_txt_path		     = "./ioc_demo/walk_basenames.txt";
	string demontraj_txt_path_prefix = "./ioc_demo/walk_traj/";
	string feat_maps_xml_path_prefix = "./ioc_demo/walk_feat/";
	string rect_imag_jpg_path_prefix = "./ioc_demo/walk_imag/";
	string output_params_path		     = "./ioc_demo/walk_output/walk_reward_params.txt";

	CCP model;

  model.initialize      ();
	model.loadBasenames		(basenames_txt_path);
	model.loadDemoTraj		(demontraj_txt_path_prefix);
	model.loadFeatureMaps	(feat_maps_xml_path_prefix);
	model.loadImages		  (rect_imag_jpg_path_prefix);
	model.visualizeFeats	();

  //model.readPolicy("./ioc_demo/walk_output/policy.txt");
  model.estimatePolicy();
  model.savePolicy("./ioc_demo/walk_output/policy.txt");

  model.estimateGamma();
  model.estimateTransitionMatrix();
  model.estimateZeroValueFunction();
  model.estimateValueFunction();
  model.saveValueFunction("./ioc_demo/walk_output/valuefun.txt");

  //model.readValueFunction("./ioc_demo/walk_output/valuefun.txt");
  model.visualizeValueFunction();

  model.estimateRewardFunction();
  model.saveRewardFunction("./ioc_demo/walk_output/rewardfun.txt");

  //model.readRewardFunction("./ioc_demo/walk_output/rewardfun.txt");
  model.visualizeRewardFunction();
  return 0;
}
