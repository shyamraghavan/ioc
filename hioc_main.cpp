#include <iostream>
#include "trainHIOC.hpp"
using namespace std;

int main (int argc, char * const argv[]) {

	int trainSetSize;		//
	int testSetSize;		//
	int reps;				// number of repetition
	int rep_start = 0;			// 
	int append;
	int trainSizePercent;
	int multigoal;
	int visualize = 0;
	
	float softmax_k = 1.0;
	string dataset;
	string action;
	string job;
	string run;
	string root;
	string s;
	
	cout << "argc:" << argc << endl;

	if(argc==1){
		dataset = "VIRAT";			// UCSD
		action	= "approach";		// action
		run		= "full";	// experiment type (add markovfull)
		job		= "test";			// job test/train/transfer
		trainSizePercent = 80;
		reps = 1;
		rep_start = 0;				// max is 4  
		append = 0;					// load current parameters and update
		multigoal = 1;
		visualize = 0;
		root = "/"+dataset+"/";
	}
	else if(argc==9){
		dataset = argv[1];
		action	= argv[2];				// action
		run		= argv[3];				// experiment type
		job		= argv[4];				// job test/train
		trainSizePercent = atoi(argv[5]);
		reps = atoi(argv[6]);
		rep_start = atoi(argv[7]);
		append = atoi(argv[8]);
		multigoal = 0;
		root = "../"+dataset+"/";
	}
	else if(argc==10){
		dataset = argv[1];
		action	= argv[2];				// action
		run		= argv[3];				// experiment type
		job		= argv[4];				// job test/train
		trainSizePercent = atoi(argv[5]);
		reps = atoi(argv[6]);
		rep_start = atoi(argv[7]);
		append = atoi(argv[8]);
		multigoal = atoi(argv[9]);
		root = "../"+dataset+"/";
	}
	else{
		cout << "Wrong number of arguments." << endl;
		cout << "[dataset] [action] [const/feat/full] [train/test/both] [trainsize%] [nfold] [rep_start] [append] ([multigoal])" << endl;
		return 0;
	}
	
	if(multigoal) cout << "MULTI-GOAL INFERENCE" << endl;

	
	trainHIOC hioc;
	float scale = 0.30;
	int width = round(1280*scale);
	int height = round(720*scale);
	hioc.init(width,height);
	
	hioc.setFlags(action,run,append,root,softmax_k,job,multigoal,visualize);
	
	vector<string> file_id;
	
	if(action=="approach"){

		file_id.push_back("VIRAT_S_000004_6900_7800_1"); // GOOD: non-linear
		file_id.push_back("VIRAT_S_000001_9660_10170_1");	
		file_id.push_back("VIRAT_S_000002_3400_4200_1");
		file_id.push_back("VIRAT_S_000003_5340_5680_1");
		file_id.push_back("VIRAT_S_000003_17160_17700_1");
		file_id.push_back("VIRAT_S_000004_14540_15300_1");
		file_id.push_back("VIRAT_S_000005_780_1560_1"); // moving car, remove later
		file_id.push_back("VIRAT_S_000006_7490_7780_1");
		file_id.push_back("VIRAT_S_000006_15550_15790_1"); // this is a valid approach
		file_id.push_back("VIRAT_S_000008_1100_1814_1"); // replacement
		
		file_id.push_back("VIRAT_S_000000_9520_10100_1"); // GOOD: non-linear
		file_id.push_back("VIRAT_S_000007_1080_1680_1"); // linear long path
		file_id.push_back("VIRAT_S_000007_2525_3090_1"); // walk on side walk
		file_id.push_back("VIRAT_S_000007_5500_6130_1");
		file_id.push_back("VIRAT_S_000007_9980_10450_1");
		file_id.push_back("VIRAT_S_000007_14455_14910_1");
		file_id.push_back("VIRAT_S_000007_14590_15000_1"); // slight curve
		file_id.push_back("VIRAT_S_000008_1560_1990_1"); // slight curve
		file_id.push_back("VIRAT_S_000006_4110_4880_1"); // slow walk
		file_id.push_back("VIRAT_S_000007_9055_9500_1"); // hand shake, slight curve
		
		file_id.push_back("VIRAT_S_000005_5800_6400_1"); // partial?
		file_id.push_back("VIRAT_S_000006_15060_15730_1"); // hand off (avoids other car?) test
		
		//file_id.push_back("VIRAT_S_000005_15970_16170_1"); // new
		//file_id.push_back("VIRAT_S_000006_4620_4880_1"); // very short
		//file_id.push_back("VIRAT_S_000000_3070_3455_1"); // bad tracking (remove?)
		//file_id.push_back("VIRAT_S_000003_12270_12770_1"); // bad tracking
		//file_id.push_back("VIRAT_S_000008_1100_1814_1"); // replacement
		//file_id.push_back("VIRAT_S_000006_200_400_1"); // too short?
		

		// 22
	}
	else if(action=="depart"){
		
		file_id.push_back("VIRAT_S_000005_6720_7200_0");	// typical path (train)
		file_id.push_back("VIRAT_S_000001_10240_10600_0");	// medium
		file_id.push_back("VIRAT_S_000001_17426_17800_0");	// straight line
		file_id.push_back("VIRAT_S_000002_4200_4910_0");	// long
		file_id.push_back("VIRAT_S_000004_7800_8700_0");	// GOOD test: missing obs
		file_id.push_back("VIRAT_S_000003_5680_6150_0");	// straight medium
		file_id.push_back("VIRAT_S_000007_9970_10350_0");	// train: slight short curve
		file_id.push_back("VIRAT_S_000004_15990_16600_0");	// clean curve
		file_id.push_back("VIRAT_S_000007_1930_2280_0");	// short 
		file_id.push_back("VIRAT_S_000003_12950_13235_0");	// too short sparse detection
		
		file_id.push_back("VIRAT_S_000003_17700_18500_0");	// GOOD test: bad tracker association
		file_id.push_back("VIRAT_S_000006_16100_16740_0");	// GOOD: test, sparse tracking		
		file_id.push_back("VIRAT_S_000006_5730_6000_0");	// good test: avoid car?
		file_id.push_back("VIRAT_S_000006_7160_7435_0");	// medium
		file_id.push_back("VIRAT_S_000006_16100_16560_0");
		file_id.push_back("VIRAT_S_000007_4830_5290_0");	// side walk
		file_id.push_back("VIRAT_S_000007_6670_7345_0");	// long
		file_id.push_back("VIRAT_S_000005_2750_3000_0");	// bad tracking, short path
		file_id.push_back("VIRAT_S_000007_11020_11690_0");
		file_id.push_back("VIRAT_S_000007_15350_15910_0");

		//20
		
		//file_id.push_back("VIRAT_S_000006_5380_5600_0"); // short bad track
		//file_id.push_back("VIRAT_S_000005_20726_20950_0"); // short, bad track
		//file_id.push_back("VIRAT_S_000006_580_740_0"); // short bad track
		//file_id.push_back("VIRAT_S_000007_15275_15725_0"); // bad score (short bad track)
		//file_id.push_back("VIRAT_S_000000_3800_3985_0"); // too short
		//file_id.push_back("VIRAT_S_000008_1860_1990_0"); // short bad,
		//file_id.push_back("VIRAT_S_000008_2380_2930_0"); // short bad
	}
	else if(action=="walk"){
		
		file_id.push_back("VIRAT_S_000005_12340_13370_2");	// diag, avoid car
		file_id.push_back("VIRAT_S_000000_4820_5865_2");	// side walk
		file_id.push_back("VIRAT_S_000000_17200_18350_2");	// Test sparse, miss detect
		file_id.push_back("VIRAT_S_000005_15100_15970_2");	// diag avoid car (this is a walk?)
		file_id.push_back("VIRAT_S_000000_12020_13200_2");	// side walk
		file_id.push_back("VIRAT_S_000003_7500_8435_2");	// shady
		file_id.push_back("VIRAT_S_000006_1945_2820_2");	// straigh diag
		file_id.push_back("VIRAT_S_000003_7900_8920_2");	// diaganal
		file_id.push_back("VIRAT_S_000001_5900_7050_2");	// side walk
		file_id.push_back("VIRAT_S_000007_2950_3985_2");	// unique, avoid car, side walk
		
		file_id.push_back("VIRAT_S_000008_4930_5800_2");	// partial diag
		file_id.push_back("VIRAT_S_000000_0_590_2");		// start mid (bad score!)		
		file_id.push_back("VIRAT_S_000003_19110_20200_2");	// Test sparse, miss detect
		file_id.push_back("VIRAT_S_000007_15070_16190_2");	// Test walk behind car, partial detect
		
	}
	else if(action=="walk2"){
		file_id.push_back("VIRAT_S_040100_01_155_1020_2");
		file_id.push_back("VIRAT_S_040103_01_1200_1878_2");
		file_id.push_back("VIRAT_S_040103_05_0_850_2");
		file_id.push_back("VIRAT_S_040103_07_1120_2260_2");
		file_id.push_back("VIRAT_S_040103_08_190_920_2");
		file_id.push_back("VIRAT_S_040104_00_0_1250_2");
		file_id.push_back("VIRAT_S_040104_01_690_1590_2");
		file_id.push_back("VIRAT_S_040104_01_2870_3315_2");
		file_id.push_back("VIRAT_S_040104_01_3680_4240_2");
		file_id.push_back("VIRAT_S_040104_01_4110_4660_2");
		
		file_id.push_back("VIRAT_S_040104_01_4330_5790_2");
		file_id.push_back("VIRAT_S_040104_01_4760_6120_2");
		file_id.push_back("VIRAT_S_040104_01_5600_6886_2");
		// 13
	}
	else if(action=="depart2"){
		file_id.push_back("VIRAT_S_040100_02_880_1000_0");
		file_id.push_back("VIRAT_S_040100_03_1481_1738_0");
		file_id.push_back("VIRAT_S_040100_05_1330_1480_0");
		file_id.push_back("VIRAT_S_040103_00_262_530_0");
		file_id.push_back("VIRAT_S_040103_01_1040_1650_0");
		file_id.push_back("VIRAT_S_040103_03_400_795_0");
		file_id.push_back("VIRAT_S_040103_03_3490_3980_0");
		file_id.push_back("VIRAT_S_040103_05_1490_2050_0");
		file_id.push_back("VIRAT_S_040103_06_1750_2005_0");
		file_id.push_back("VIRAT_S_040103_07_900_1420_0");
		
		file_id.push_back("VIRAT_S_040104_01_1710_2150_0");
		file_id.push_back("VIRAT_S_040104_01_4490_4820_0");
		file_id.push_back("VIRAT_S_040104_01_5650_6050_0");
		// 13
	}
	else if(action=="approach2"){
		file_id.push_back("VIRAT_S_040100_00_1505_1765_1");
		file_id.push_back("VIRAT_S_040100_01_1140_1425_1");
		file_id.push_back("VIRAT_S_040100_02_425_660_1");
		file_id.push_back("VIRAT_S_040100_04_125_390_1");
		file_id.push_back("VIRAT_S_040103_02_180_660_1");
		file_id.push_back("VIRAT_S_040103_03_2610_3140_1");
		file_id.push_back("VIRAT_S_040104_00_230_770_1");
		file_id.push_back("VIRAT_S_040104_00_310_910_1");
		file_id.push_back("VIRAT_S_040104_01_4850_5230_1");
		file_id.push_back("VIRAT_S_040104_01_4920_5230_1");
		// 10
	}
	else{
		cout << "Invalid action name." << endl;
		return 0;
	}
	

	
	
	vector<string> tfile_id;
	if(action=="walk"){
		tfile_id.push_back("VIRAT_S_040100_01_155_1020_2");
		tfile_id.push_back("VIRAT_S_040103_01_1200_1878_2");
		tfile_id.push_back("VIRAT_S_040103_05_0_850_2");
		tfile_id.push_back("VIRAT_S_040103_07_1120_2260_2");
		tfile_id.push_back("VIRAT_S_040103_08_190_920_2");
		tfile_id.push_back("VIRAT_S_040104_00_0_1250_2");
		tfile_id.push_back("VIRAT_S_040104_01_690_1590_2");
		tfile_id.push_back("VIRAT_S_040104_01_2870_3315_2");
		tfile_id.push_back("VIRAT_S_040104_01_3680_4240_2");
		tfile_id.push_back("VIRAT_S_040104_01_4110_4660_2");
		
		tfile_id.push_back("VIRAT_S_040104_01_4330_5790_2");
		tfile_id.push_back("VIRAT_S_040104_01_4760_6120_2");
		tfile_id.push_back("VIRAT_S_040104_01_5600_6886_2");
		// 13
	}
	else if(action=="depart"){
		tfile_id.push_back("VIRAT_S_040100_02_880_1000_0");
		tfile_id.push_back("VIRAT_S_040100_03_1481_1738_0");
		tfile_id.push_back("VIRAT_S_040100_05_1330_1480_0");
		tfile_id.push_back("VIRAT_S_040103_00_262_530_0");
		tfile_id.push_back("VIRAT_S_040103_01_1040_1650_0");
		tfile_id.push_back("VIRAT_S_040103_03_400_795_0");
		tfile_id.push_back("VIRAT_S_040103_03_3490_3980_0");
		tfile_id.push_back("VIRAT_S_040103_05_1490_2050_0");
		tfile_id.push_back("VIRAT_S_040103_06_1750_2005_0");
		tfile_id.push_back("VIRAT_S_040103_07_900_1420_0");
		
		tfile_id.push_back("VIRAT_S_040104_01_1710_2150_0");
		tfile_id.push_back("VIRAT_S_040104_01_4490_4820_0");
		tfile_id.push_back("VIRAT_S_040104_01_5650_6050_0");
		// 13
	}
	else if(action=="approach"){
		tfile_id.push_back("VIRAT_S_040100_00_1505_1765_1");
		tfile_id.push_back("VIRAT_S_040100_01_1140_1425_1");
		tfile_id.push_back("VIRAT_S_040100_02_425_660_1");
		tfile_id.push_back("VIRAT_S_040100_04_125_390_1");
		tfile_id.push_back("VIRAT_S_040103_02_180_660_1");
		tfile_id.push_back("VIRAT_S_040103_03_2610_3140_1");
		tfile_id.push_back("VIRAT_S_040104_00_230_770_1");
		tfile_id.push_back("VIRAT_S_040104_00_310_910_1");
		tfile_id.push_back("VIRAT_S_040104_01_4850_5230_1");
		tfile_id.push_back("VIRAT_S_040104_01_4920_5230_1");
		// 10
	}
	else if(action=="approach2"){
		
		tfile_id.push_back("VIRAT_S_000004_6900_7800_1"); // GOOD: non-linear
		tfile_id.push_back("VIRAT_S_000001_9660_10170_1");	
		tfile_id.push_back("VIRAT_S_000002_3400_4200_1");
		tfile_id.push_back("VIRAT_S_000003_5340_5680_1");
		tfile_id.push_back("VIRAT_S_000003_17160_17700_1");
		tfile_id.push_back("VIRAT_S_000004_14540_15300_1");
		tfile_id.push_back("VIRAT_S_000005_780_1560_1"); // moving car, remove later
		tfile_id.push_back("VIRAT_S_000006_7490_7780_1");
		tfile_id.push_back("VIRAT_S_000006_15550_15790_1"); // this is a valid approach
		tfile_id.push_back("VIRAT_S_000008_1100_1814_1"); // replacement
		
		tfile_id.push_back("VIRAT_S_000000_9520_10100_1"); // GOOD: non-linear
		tfile_id.push_back("VIRAT_S_000007_1080_1680_1"); // linear long path
		tfile_id.push_back("VIRAT_S_000007_2525_3090_1"); // walk on side walk
		tfile_id.push_back("VIRAT_S_000007_5500_6130_1");
		tfile_id.push_back("VIRAT_S_000007_9980_10450_1");
		tfile_id.push_back("VIRAT_S_000007_14455_14910_1");
		tfile_id.push_back("VIRAT_S_000007_14590_15000_1"); // slight curve
		tfile_id.push_back("VIRAT_S_000008_1560_1990_1"); // slight curve
		tfile_id.push_back("VIRAT_S_000006_4110_4880_1"); // slow walk
		tfile_id.push_back("VIRAT_S_000007_9055_9500_1"); // hand shake, slight curve

		tfile_id.push_back("VIRAT_S_000005_5800_6400_1"); // partial?
		tfile_id.push_back("VIRAT_S_000006_15060_15730_1"); // hand off (avoids other car?) test
		
		// 22
	}
	else if(action=="depart2"){
		
		tfile_id.push_back("VIRAT_S_000005_6720_7200_0");	// typical path (train)
		tfile_id.push_back("VIRAT_S_000001_10240_10600_0");	// medium
		tfile_id.push_back("VIRAT_S_000001_17426_17800_0");	// straight line
		tfile_id.push_back("VIRAT_S_000002_4200_4910_0");	// long
		tfile_id.push_back("VIRAT_S_000004_7800_8700_0");	// GOOD test: missing obs
		tfile_id.push_back("VIRAT_S_000003_5680_6150_0");	// straight medium
		tfile_id.push_back("VIRAT_S_000007_9970_10350_0");	// train: slight short curve
		tfile_id.push_back("VIRAT_S_000004_15990_16600_0");	// clean curve
		tfile_id.push_back("VIRAT_S_000007_1930_2280_0");	// short 
		tfile_id.push_back("VIRAT_S_000003_12950_13235_0");	// too short sparse detection
		
		tfile_id.push_back("VIRAT_S_000003_17700_18500_0");	// GOOD test: bad tracker association
		tfile_id.push_back("VIRAT_S_000006_16100_16740_0");	// GOOD: test, sparse tracking		
		tfile_id.push_back("VIRAT_S_000006_5730_6000_0");	// good test: avoid car?
		tfile_id.push_back("VIRAT_S_000006_7160_7435_0");	// medium
		tfile_id.push_back("VIRAT_S_000006_16100_16560_0");
		tfile_id.push_back("VIRAT_S_000007_4830_5290_0");	// side walk
		tfile_id.push_back("VIRAT_S_000007_6670_7345_0");	// long
		tfile_id.push_back("VIRAT_S_000005_2750_3000_0");	// bad tracking, short path
		tfile_id.push_back("VIRAT_S_000007_11020_11690_0");
		tfile_id.push_back("VIRAT_S_000007_15350_15910_0");
		
		//20
		
	}
	else if(action=="walk2"){
		
		tfile_id.push_back("VIRAT_S_000005_12340_13370_2");	// diag, avoid car
		tfile_id.push_back("VIRAT_S_000000_4820_5865_2");	// side walk
		tfile_id.push_back("VIRAT_S_000000_17200_18350_2");	// Test sparse, miss detect
		tfile_id.push_back("VIRAT_S_000005_15100_15970_2");	// diag avoid car (this is a walk?)
		tfile_id.push_back("VIRAT_S_000000_12020_13200_2");	// side walk
		tfile_id.push_back("VIRAT_S_000003_7500_8435_2");	// shady
		tfile_id.push_back("VIRAT_S_000006_1945_2820_2");	// straigh diag
		tfile_id.push_back("VIRAT_S_000003_7900_8920_2");	// diaganal
		tfile_id.push_back("VIRAT_S_000001_5900_7050_2");	// side walk
		tfile_id.push_back("VIRAT_S_000007_2950_3985_2");	// unique, avoid car, side walk
		
		tfile_id.push_back("VIRAT_S_000008_4930_5800_2");	// partial diag
		tfile_id.push_back("VIRAT_S_000000_0_590_2");		// start mid (bad score!)		
		tfile_id.push_back("VIRAT_S_000003_19110_20200_2");	// Test sparse, miss detect
		tfile_id.push_back("VIRAT_S_000007_15070_16190_2");	// Test walk behind car, partial detect
	}
	else{
		if(job == "transfer"){
			cout << "ERROR: Invalid action name." << endl;
			return 0;
		}
	}
	
	// ===== DO NOT COMMENT OUT from here on ==== //
	cout << "number of files total:" << (float)file_id.size() << endl;
	trainSetSize = floor((float)file_id.size() * trainSizePercent/100.0);
	testSetSize = -1;
	s = action+"-"+run+"-";

	
	// ==== Adjust test set size ==== //
	if(testSetSize==-1){ 
		testSetSize = (int)file_id.size()-trainSetSize;
		if(job == "transfer") testSetSize = (int)tfile_id.size();
	}
	
	
	// ==== SANITY CHECK ==== //
	
	if (testSetSize<1) {
		cout << "ERROR: Test set size is less than 1." << endl;
		exit(1);
	}
	
	//if(action=="walk" && job!="tran" (testSetSize+trainSetSize > 14) ){
	//	cout << "ERROR: Total size too big.\n";
	//	exit(1);
	//}
	
	if( (run == "markovfeat" || run == "markovfull") && trainSetSize < 3 ){
		cout << "ERROR: Under-determined system. Increase number of training samples." << endl;
		exit(1);
	}
	
	cout << "Execute:" << s << endl;
	cout << "Train on:" << trainSetSize << endl;
	cout << "Test on:" << testSetSize << endl;
	cout << "Number of repititions:" << reps << endl;
	cout << "Append to training:" << append << endl;
	
	
	////////////////////////////
	//          HIOC          //
	////////////////////////////
	

	
	for(int i=rep_start;i<rep_start+reps;i++){		
		
		// ==== add training data ==== //
		vector<string> train_files;
		vector<int> train_id(file_id.size(),0);
		
		//int trainStartIndex = i*trainSetSize;
		int trainStartIndex = i*testSetSize;   // shift by testset size (20%)
		int trainEndIndex;
		for(int j=0;j<trainSetSize;j++){
			int k = (trainStartIndex+j)%file_id.size();
			train_files.push_back(file_id[k]);
			train_id[k]=1;
			trainEndIndex = k;
		}
		
		hioc.setTrainFiles(s,train_files,file_id);
		
		
		if(job=="train" || job== "both"){
			if(run =="full" || run=="feat" || run=="const") hioc.runTrainHIOC(2000,0.05);
			else if(run=="markovfeat") hioc.runTrainMarkovFeat();
			else if(run=="markovfull") hioc.runTrainMarkovFeat();
			else if(run=="markovmotion") hioc.runTrainMarkovMotion();
			else cout << "bad training name:" << run << endl;
		}
		
		
		// ==== add test data ==== //
		vector<string> test_files;
		if(job=="transfer"){
			int testStartIndex = 0; //11
			for(int j=0;j<testSetSize;j++){ // test on transfer set
				int k = (testStartIndex+j)%tfile_id.size();
				test_files.push_back(tfile_id[k]);
			}
		}
		else {
			int testStartIndex = (trainEndIndex+1)%file_id.size();
			for(int j=0;j<testSetSize;j++){ // test on remaining
				int k = (testStartIndex+j)%file_id.size();
				test_files.push_back(file_id[k]);
			}
		}
		
		hioc.setTestFiles(test_files);
		
		if(job=="test" || job== "both" || job=="transfer"){
		
			if(run=="feat" || run=="const" || run =="full") hioc.runTestHIOCgoals();
			else if(run=="markovfeat") hioc.runTestMarkovFeat();
			else if(run=="markovfull") hioc.runTestMarkovFeat();
			else if(run=="markovmotion") hioc.runTestMarkovMotion();
			else cout << "bad train name:" << run << endl;
		}
	}
    return 0;
}
