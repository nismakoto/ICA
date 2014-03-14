#!/usr/bin/make -f
 
CXX = g++
CXXFLAGS = -O3 -Wall -Wextra -std=c++11
LDFLAGS = -lm
#LDFLAGS = -lncursesw
OBJS = ica
 
all: $(OBJS)
 
clean:
	$(RM) $(OBJS) core *y_data*.*
 
exercise18.cpp:
	$(CXX) $(CXXFLAGS) $< $(LDFLAGS) -o $@
