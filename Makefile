all:
	g++ ccp.cpp main_1.cpp `pkg-config --cflags --libs opencv`
