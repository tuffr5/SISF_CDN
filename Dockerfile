FROM ubuntu:24.04

ARG CDN_PORT=6000
ARG BUILD_THREAD=5

WORKDIR /app

COPY . .

RUN apt update
RUN apt install -y build-essential libboost-all-dev libsqlite3-dev libasio-dev nasm
RUN cd x264; make -j $BUILD_THREAD; cd ..
RUN cd zstd; make -j $BUILD_THREAD; cd ..

RUN make 

EXPOSE ${CDN_PORT}

CMD ["sh", "-c", "ls -lh /data/; ./nTracer_cdn 6000 /data/"]
