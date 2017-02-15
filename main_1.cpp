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

  model.estimatePolicy();

  return 0;
}
