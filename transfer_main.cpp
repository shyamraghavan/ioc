#include <iostream>
#include "transfer.hpp"
using namespace std;

int main (int argc, char * const argv[])
{

	string prev_basenames_txt_path        = "./ioc_demo/walk_basenames.txt";
	string prev_feat_maps_xml_path_prefix = "./ioc_demo/walk_feat/";
	string prev_reward_fun_path           = "./ioc_demo/walk_output/rewardfun.txt";

	string basenames_txt_path        = "./ioc_demo/transfer_basename_1.txt";
	string feat_maps_xml_path_prefix = "./ioc_demo/transfer_feat/";

  Transfer exp;

  exp.initialize();
  exp.loadPrevBasenames(prev_basenames_txt_path);
  exp.loadPrevFeatMap(prev_feat_maps_xml_path_prefix);
  exp.loadPrevReward(prev_reward_fun_path);
  exp.reshapePrevFeatMap();

  exp.loadBasenames(basenames_txt_path);
  exp.loadFeatMap(feat_maps_xml_path_prefix);

  return 0;
}
