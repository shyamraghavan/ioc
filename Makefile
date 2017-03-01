all:
	g++ -O3 -lpthread -std=c++11 ccp.cpp main_1.cpp `pkg-config --cflags --libs opencv`
