CXXFILES = src/app.cpp
INCLIBS = ./zstd/lib/libzstd.so ./x264/libx264.a
LIBS = -lpthread -lsqlite3 -lavcodec -lavformat -lavutil -lswscale 
CXXFLAGS = -O3 -std=c++17 -march=native -o nTracer_cdn -Wl,--no-as-needed -ldl

all:
	$(CXX) $(CXXFILES) $(INCLIBS) $(LIBS) $(CXXFLAGS)