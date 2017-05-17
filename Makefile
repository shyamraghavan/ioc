ccp:
	g++ -O2 -lpthread -std=c++11 -o ccp.o ccp.cpp ccp_main.cpp `pkg-config --cflags --libs opencv`

ioc:
	g++ -O2 -lpthread -std=c++11 -o ioc.o ioc.cpp ioc_main.cpp `pkg-config --cflags --libs opencv`

qlearn:
	g++ -O2 -std=c++11 qlearn.cpp -o qlearn.o qlearn_main.cpp `pkg-config --cflags --libs opencv`

transfer:
	g++ -O3 -lpthread -std=c++11 -o transfer.o transfer.cpp transfer_main.cpp `pkg-config --cflags --libs opencv`

prep:
	cd prep && g++ -O2 -std=c++11 -o prep.o prepUMD.cpp prep.cpp `pkg-config --cflags --libs opencv`

hioc:
	g++ -O2 -std=c++11 trainHIOC.cpp -o trainHIOC.o hioc_main.cpp `pkg-config --cflags --libs opencv`
