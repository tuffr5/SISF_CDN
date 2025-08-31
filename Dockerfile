FROM ubuntu:24.04

ARG CDN_PORT=6000
ARG BUILD_THREAD=64

RUN apt update && \
    apt install -y \
        build-essential libboost-all-dev libsqlite3-dev libasio-dev nasm cmake \
        ffmpeg libswscale-dev libavutil-dev libavcodec-dev libavdevice-dev libavfilter-dev libavformat-dev \
        libavutil-dev libpostproc-dev libswresample-dev \
        libhdf5-dev libc6-dev

WORKDIR /app

COPY . .

RUN cd x264; make -j $BUILD_THREAD; cd ..
RUN cd zstd; make -j $BUILD_THREAD; cd ..
RUN cd ffmpeg_HDF5_filter; cmake .; make -j $BUILD_THREAD; cd ..

RUN cmake .; exit 0
RUN make -j $BUILD_THREAD

EXPOSE ${CDN_PORT}

CMD ["sh", "-c", "ls -lh /data/; ./nTracer_cdn 6000 /data/"]
