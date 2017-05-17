#include <iostream>
#include "prepUMD.hpp"
using namespace std;

int main (int argc, char * const argv[])
{
  cout << "\n================================\n";
  cout << "PREPARE UMD DATA FOR LEARNING";
  cout << "\n================================\n";

  string root = "../ioc_demo/scene_b";
  prepUMD pu;

  string vidseg_info_fp	= "/data_params/segment_info.txt";
  string basenames_txt_path = "/walk_basenames.txt";

  pu.set(root.c_str(),394,216);
  pu.load_basenames(basenames_txt_path);
  //pu.prepare_static_features();
  pu.prepare_trajectory_features();

	return 0;
}
