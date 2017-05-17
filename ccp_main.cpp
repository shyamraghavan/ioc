#include <iostream>
#include "ccp.hpp"
using namespace std;

int main (int argc, char * const argv[])
{

  string basenames_txt_path        = "./ioc_demo/walk_basenames.txt";
  string demontraj_txt_path_prefix = "./ioc_demo/walk_traj/";
  string feat_maps_xml_path_prefix = "./ioc_demo/walk_feat/";
  string rect_imag_jpg_path_prefix = "./ioc_demo/walk_imag/";
  string output_params_path        = "./ioc_demo/mohit_output/ccp_output_1/walk_reward_params.txt";

  CCP model;

  model.initialize      ();
  model.loadBasenames   (basenames_txt_path);
  model.loadDemoTraj    (demontraj_txt_path_prefix);
  model.loadFeatureMaps (feat_maps_xml_path_prefix);
  model.loadImages      (rect_imag_jpg_path_prefix);
  //model.visualizeFeats  ();

  //model.readPolicy("./ioc_demo/walk_output/policy.txt");
  model.estimatePolicy(false);
  model.savePolicy("./ioc_demo/mohit_output/ccp_output_1/policy.txt");

  model.estimateGamma();
  model.estimateTransitionMatrix();
  model.estimateZeroValueFunction();
  model.estimateValueFunction();
  model.saveValueFunction("./ioc_demo/mohit_output/ccp_output_1/valuefun.txt");

  //model.readValueFunction("./ioc_demo/walk_output/valuefun.txt");
  model.visualizeValueFunction();

  model.estimateRewardFunction();
  model.saveRewardFunction("./ioc_demo/mohit_output/ccp_output_1/rewardfun.txt");

  //model.readRewardFunction("./ioc_demo/walk_output/rewardfun.txt");
  //model.saveTrueRewardFunction();
  model.visualizeRewardFunction();

  /*ofstream fs("./ioc_demo/walk_output/subsampling_distance.txt");
  if(!fs.is_open()) cout << "ERROR: Writing: Subsampling Distance" << endl;

  for(int i=0;i<model._num_samps;i++)
  {
    model.setUpRandomization();

    model.estimatePolicy(true);
    model.estimateGamma();
    model.estimateTransitionMatrix();
    model.estimateZeroValueFunction();
    model.estimateValueFunction();
    model.estimateRewardFunction();

    float total = 0;

    for(int y=0;y<model._size.height;y++)
    {
      for(int x=0;x<model._size.width;x++)
      {
        for(int a=0;a<model._na;a++)
        {
          float r = model._R.at<Vec9f>(y,x)[a];
          float rt = model._R_true.at<Vec9f>(y,x)[a];

          if (!isnanf(r) && !isnanf(rt))
          {
            total += (r - rt) * (r - rt);
          }
        }
      }
    }

    cout << "Distance: " << to_string(sqrt(total));
    fs << to_string(sqrt(total));
    fs << endl;
  }
  */

  return 0;
}
