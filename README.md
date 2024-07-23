# Scalable Image Storage Format (SISF) CDN
A tool for remote access of SISF Files over HTTP.

## Endpoint Definitions
TODO

## C++ Libraries

1. Crow-cpp ([https://crowcpp.org/master/](https://crowcpp.org/master/))
   - see `src/crow.h` for more information.
2. C++ Subprocess from Arun Muralidharan ([https://github.com/arun11299/cpp-subprocess](https://github.com/arun11299/cpp-subprocess))
   - see `src/subprocess.hpp` for more information.
3. JSON for Modern C++ from Niels Lohmann ([https://github.com/nlohmann/json](https://github.com/nlohmann/json))
   - see `src/json.hpp` for more information.

## Local Setup

### Platform
This repository has been extensively tested under Ubuntu versions 22.04LTS and 24.04LTS. Before begining, ensure that the following packages/libraries are installed:
- `build-essential`
- `git`
- `libboost-all-dev`
- `libsqlite3-dev`
- `libasio-dev`
- `nasm`

### Git Submodules

There are two libraries which are imported using Git submodules which are required for building:
1. libzstd [https://github.com/facebook/zstd](https://github.com/facebook/zstd)
2. libx264 [https://www.videolan.org/developers/x264.html](https://www.videolan.org/developers/x264.html)

To import and build these dependencies, use the following shell commands:
```
git submodule init
git submodule update

cd x264
make -j 20
cd ..

cd zstd
make -j 20
cd ..
```

## Docker Setup

## Metadata Schema

### Introduction

The data storage strategy described here relies on two layers of segmentation

### Archive structure

```
(root)
├── meta
|   ├── chunk_0_0_0.0.1X.meta
|   ├── chunk_0_0_0.0.16X.meta
|   ├── chunk_0_0_1.0.1X.meta
|   ├── (...)
├── data
|   ├── chunk_0_0_0.0.1X.data
|   ├── chunk_0_0_1.0.16X.data
|   ├── chunk_0_0_1.0.1X.data
|   ├── (...)
├── metadata.bin
```

**Notes:**
- Individual parts of this structure can be symlinked to different file systems (e.g. tmpfs or a cache SSD)
- 

### Chunk naming scheme

```
chunk_0_0_0.0.1X.data
      | | | |  |
      ^-|-|-|--|-------- x location
        ^-|-|--|-------- y location
          ^-|--|-------- z location
            ^--|-------- channel
               ^-------- downsampling rate
```

### Contents of metadata.bin

```
[uint16_t version]
[uint16_t dtype]
[uint16_t channel_count]
[uint16_t mchunkx]
[uint16_t mchunky]
[uint16_t mchunkz]
[uint64_t resx]
[uint64_t resy]
[uint64_t resz]
[uint64_t sizex]
[uint64_t sizey]
[uint64_t sizez]
```

### Content of image.meta

```
[uint16_t version]
[uint16_t dtype]
[uint16_t channel_count]
[uint16_t compression_type]
[uint16_t chunkx]
[uint16_t chunky]
[uint16_t chunkz]
[uint64_t sizex]
[uint64_t sizey]
[uint64_t sizez]

[uint64_t cropstartx]
[uint64_t cropendx]
[uint64_t cropstarty]
[uint64_t cropendy]
[uint64_t cropstartz]
[uint64_t cropendz]

for i in range(count):
    [uint64_t offset]
    [uint32_t size]
```

### Parameter options

#### `dtype`
- `1 -> uint16`
- `2 -> uint8` (not implemented) 