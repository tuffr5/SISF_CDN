FROM ubuntu:24.04

ARG CDN_PORT=6000

WORKDIR /app

COPY . .

RUN apt update
RUN apt install -y build-essential libboost-all-dev libsqlite3-dev libasio-dev nasm
RUN cd x264; make -j 5; cd ..
RUN cd zstd; make -j 5; cd ..

RUN make 

EXPOSE ${CDN_PORT}

CMD ["./nTracer_cdn", "$CDN_PORT", "/data/"]
