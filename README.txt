Intermediate data to run the HIOC algorithm of Kitani et al. 2012

* only contains data for the activity 'walk through the scene'
* please contact Kris at kkitani@cs.cmu.edu if you would like the other activities (approach car, depart car)
* contains data for two different parking lot scenes

Activity Forecasting.
Kris M. Kitani, Brian D. Ziebart, Drew Bagnell and Martial Hebert. 
European Conference on Computer Vision (ECCV 2012).

Folders:
basenames			list of basenames for demonstrated trajectories
reward_features			output of IOC (used to generate reward function)
empirical_feature_counts	sum of features over a demonstrated trajectory
feature_maps			output of semantic scene labeling algorithm (munoz et al. 2010)
reward_function_forecasting	reward function using learned_weights.txt
topdown_images			birds eye view image of scene
tracker_output			observed and demonstrated trajectories

Using this data, you can implement softmax value iteration. Using 
the reward maps you can learn the soft value function. You can compute 
the policy p(a|x) from the value function and use it to propagate 
probabilities from a specified start and end location. In the paper,
the distribution over trajectories is visualized using the cumilative
sum of probabilities over N steps.
