app:
	g++ -std=c++17 -O3 -march=native src/app.cpp ./lib/zstd-1.5.5/lib/libzstd.so ./lib/x264/libx264.a -lpthread -lsqlite3 -o nTracer_cdn -Wl,--no-as-needed -ldl