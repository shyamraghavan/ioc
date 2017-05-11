#include <iostream>
#include "qlearn.hpp"
using namespace std;

int main (int argc, char * const argv[])
{
  srand(time(NULL));

  const int NUM_EPISODES = 1000;
  QLearn model;

  model.initialize();
  model.readRewardFunction("./ioc_demo/walk_output/rewardfun.txt");
  model.setGoal();

  for (int i=0;i<NUM_EPISODES;i++)
  {
    model.setStart();
    model.qLearn();
  }
}
