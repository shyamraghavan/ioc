ccp:
	g++ -O2 -lpthread -std=c++11 ccp.cpp ccp_main.cpp `pkg-config --cflags --libs opencv`

ioc:
	g++ -O2 -lpthread -std=c++11 ioc.cpp ioc_main.cpp `pkg-config --cflags --libs opencv`

qlearn:
	g++ -O2 -std=c++11 qlearn.cpp qlearn_main.cpp `pkg-config --cflags --libs opencv`

transfer:
	g++ -O3 -lpthread -std=c++11 transfer.cpp transfer_main.cpp `pkg-config --cflags --libs opencv`

prep:
	cd prep && g++ -O2 -std=c++11 prepUMD.cpp prep.cpp `pkg-config --cflags --libs opencv`
