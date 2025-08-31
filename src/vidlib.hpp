#include "subprocess.hpp"

#include <cmath>

#include <sstream>
#include <iostream>
#include <fstream>
#include <thread>
#include <mutex>
#include <vector>
#include <stdexcept>

#include "x264.h"

extern "C"
{
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libavutil/frame.h>
#include <libavutil/imgutils.h>
#include <libswscale/swscale.h>
#include <libavutil/avutil.h>
#include <libavutil/opt.h>

#include <ffmpeg_utils.h>
    size_t ffmpeg_native(unsigned flags, const unsigned int cd_values[], size_t buf_size, void **buf);
}

#define IMAGE_GAIN 10

std::mutex pthread_mutex;

std::string ffmpeg_location = "./ffmpeg";
std::string null_redirect = ""; // "2>/dev/null";
std::string encoder_name = "libx264";
std::string other_encode_settings = "";
std::string decode_params = "-framerate 24/1 -i - -framerate 24/1 -f rawvideo -pix_fmt gray -";

// std::string params = "-f rawvideo -pix_fmt gray16le -s 500x500 -i - -vcodec libx264 -f rawvideo -";
//-tune fastdecode
// std::string params = "-f rawvideo -framerate 24/1 -pix_fmt gray -s 128x128 -i - -vcodec libx264 -framerate 24/1 -f rawvideo -pix_fmt gray -";

const size_t target_framerate = 24;

typedef uint8_t pixtype;

void write_threaded(void *buffer, size_t buffer_size, subprocess::Popen *p)
{
    const size_t chunk_size = 4096;
    FILE *file = p->input();

    for (size_t current = 0; current < buffer_size; current += chunk_size)
    {
        size_t write_size = std::min(chunk_size, buffer_size - current);
        fwrite(((char *)buffer) + current, 1, write_size, file);
    }

    p->close_input();
}

void read_threaded(void **buffer_ret, size_t *buffer_size, FILE *file, size_t size_est)
{
    size_t current_size = size_est;
    void *buffer = malloc(current_size);

    size_t read_size = fread(buffer, 1, size_est, file);
    buffer = realloc(buffer, read_size);
    current_size = read_size;

    buffer_size[0] = current_size;
    buffer_ret[0] = buffer;
}

pixtype *uint16_to_pixtype(uint16_t *buffer, size_t len)
{
    pixtype *out = (pixtype *)calloc(len, sizeof(pixtype));

    double v;

    for (size_t i = 0; i < len; i++)
    {
        v = buffer[i]; // Read from input

        v *= IMAGE_GAIN; // Apply input gain
        v = sqrt(v);     // sqrt image to do 16->8bit conversion

        if (v <= 0)
            v = 0; // Clip lower
        if (v >= 255)
            v = 255; // Clip upper

        out[i] = (pixtype)(int)v;
    }

    return out;
}

uint16_t *pixtype_to_uint16(pixtype *buffer, size_t len)
{
    uint16_t *out = (uint16_t *)malloc(len * sizeof(uint16_t));

    double v;

    for (size_t i = 0; i < len; i++)
    {
        v = buffer[i]; // Read from input

        v *= v; // Apply x^2

        out[i] = (uint16_t)v;
    }

    return out;
}

std::pair<size_t, uint8_t *> uint16_to_pixtype_YUV420(uint16_t *buffer, size_t w, size_t h, size_t t)
{
    size_t page_offset = 3 * w * h * sizeof(uint8_t) / 2;
    size_t out_size = page_offset * t;

    uint8_t *out_buffer_r = (uint8_t *)malloc(out_size);
    uint8_t *out_buffer = out_buffer_r;

    double v;

    for (size_t f = 0; f < t; f++)
    {
        for (size_t y = 0; y < h; y++)
        {
            for (size_t x = 0; x < w; x++)
            {
                v = buffer[(f * h * w) + (y * w) + x];

                v *= IMAGE_GAIN; // Apply input gain
                v = sqrt(v);     // sqrt image to do 16->8bit conversion

                if (v <= 0)
                    v = 0; // Clip lower
                if (v >= 255)
                    v = 255; // Clip upper

                // Calculate the Y value
                out_buffer[(y * w) + x] = v; // Y (luminance)

                // Calculate the U and V values with reduced resolution (YUV 420)
                if (y % 2 == 0 && x % 2 == 0)
                {
                    int uIndex = (y / 2) * (w / 2) + (x / 2);
                    int vIndex = (y / 2) * (w / 2) + (x / 2);
                    out_buffer[w * h + uIndex] = 128;             // U (chrominance)
                    out_buffer[w * h + w * h / 4 + vIndex] = 128; // V (chrominance)
                }
            }
        }

        out_buffer += page_offset;
    }

    return {out_size, out_buffer_r};
}

uint16_t *pixtype_to_uint16_YUV420(pixtype *buffer, size_t w, size_t h, size_t t)
{
    size_t out_size = sizeof(uint16_t) * w * h * t;
    uint16_t *out_buffer = (uint16_t *)malloc(out_size);

    size_t page_offset = 3 * w * h * sizeof(uint8_t) / 2;

    double v;

    for (size_t f = 0; f < t; f++)
    {
        for (size_t y = 0; y < h; y++)
        {
            for (size_t x = 0; x < w; x++)
            {
                v = buffer[(page_offset * f) + (y * w) + x];

                v *= v;

                out_buffer[(f * h * w) + (y * w) + x] = (uint16_t)v;
            }
        }
    }

    return out_buffer;
}

// Custom I/O context for reading from memory
class FFmpegMemoryBuffer
{
public:
    const uint8_t *data;
    size_t size;
    size_t pos;

    FFmpegMemoryBuffer(const uint8_t *buffer, size_t bufferSize)
        : data(buffer), size(bufferSize), pos(0) {}
};

// Custom read function for memory buffer
static int memorybuffer_read_packet(void *opaque, uint8_t *buf, int buf_size)
{
    FFmpegMemoryBuffer *memBuffer = static_cast<FFmpegMemoryBuffer *>(opaque);

    // Calculate how much we can read
    int bytesToRead = std::min(buf_size, static_cast<int>(memBuffer->size - memBuffer->pos));

    if (bytesToRead <= 0)
    {
        return AVERROR_EOF;
    }

    // Copy data from memory buffer
    memcpy(buf, memBuffer->data + memBuffer->pos, bytesToRead);
    memBuffer->pos += bytesToRead;

    return bytesToRead;
}

pixtype *decode_stack_264(size_t sizex, size_t sizey, size_t sizez, void *buffer, size_t buffer_size)
{
    // Allocate output buffer
    uint8_t *out = (uint8_t *)calloc(sizex * sizey * sizez, sizeof(uint8_t));

    if (!buffer || buffer_size == 0)
    {
        std::cerr << "[H264Decode] Failed to load buffer" << std::endl;
        return (pixtype *)out;
    }

    // Create memory buffer context
    FFmpegMemoryBuffer memBuffer((const uint8_t *)buffer, buffer_size);

    // Allocate AVFormatContext
    AVFormatContext *formatContext = avformat_alloc_context();
    if (!formatContext)
    {
        std::cerr << "[H264Decode] Could not allocate format context" << std::endl;
        return (pixtype *)out;
    }

    // Create custom I/O context
    AVIOContext *ioContext = avio_alloc_context(
        static_cast<unsigned char *>(av_malloc(4096)), // Internal buffer
        4096,                                          // Buffer size
        0,                                             // Write flag (0 for read-only)
        &memBuffer,                                    // Opaque pointer
        memorybuffer_read_packet,                      // Read callback
        nullptr,                                       // Write callback (not needed)
        nullptr                                        // Seek callback (optional)
    );

    if (!ioContext)
    {
        std::cerr << "[H264Decode] Could not create I/O context" << std::endl;
        avformat_free_context(formatContext);
        return (pixtype *)out;
    }

    // Assign custom I/O context to format context
    formatContext->pb = ioContext;

    // Open input from memory buffer
    int ret = avformat_open_input(&formatContext, nullptr, nullptr, nullptr);
    if (ret < 0)
    {
        char errbuf[AV_ERROR_MAX_STRING_SIZE];
        av_strerror(ret, errbuf, AV_ERROR_MAX_STRING_SIZE);
        std::cerr << "[H264Decode] Could not open input: " << errbuf << std::endl;

        // Cleanup
        avio_context_free(&ioContext);
        avformat_free_context(formatContext);
        return (pixtype *)out;
    }

    // Retrieve stream information
    ret = avformat_find_stream_info(formatContext, nullptr);
    if (ret < 0)
    {
        std::cerr << "[H264Decode] Could not find stream information" << std::endl;

        // Cleanup
        avformat_close_input(&formatContext);
        return (pixtype *)out;
    }

    // Print some information about the media
    // av_dump_format(formatContext, 0, nullptr, 0);

    // Find video stream
    int video_stream_idx = -1;
    for (unsigned int i = 0; i < formatContext->nb_streams; ++i)
    {
        if (formatContext->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO)
        {
            video_stream_idx = i;
            break;
        }
    }

    if (video_stream_idx == -1)
    {
        std::cerr << "[H264Decode] Could not find video stream" << std::endl;
        return (pixtype *)out;
    }

    // Get codec parameters and codec context
    AVCodecParameters *codecpar = formatContext->streams[video_stream_idx]->codecpar;
    AVCodec *codec = (AVCodec *)avcodec_find_decoder(codecpar->codec_id);
    AVCodecContext *codec_ctx = avcodec_alloc_context3(codec);
    avcodec_parameters_to_context(codec_ctx, codecpar);

    // Open codec
    if (avcodec_open2(codec_ctx, codec, nullptr) < 0)
    {
        std::cerr << "[H264Decode] Could not open codec" << std::endl;
        return (pixtype *)out;
    }

    // Allocate frame and packet
    AVFrame *frame = av_frame_alloc();
    AVPacket packet;
    av_init_packet(&packet);

    size_t frame_cnt = 0;

    // Read frames from the video stream
    while (av_read_frame(formatContext, &packet) >= 0)
    {
        if (packet.stream_index == video_stream_idx)
        {
            // Decode video frame
            int ret = avcodec_send_packet(codec_ctx, &packet);
            if (ret < 0)
            {
                std::cerr << "[H264Decode] Error sending packet for decoding" << std::endl;
                break;
            }

            while (ret >= 0)
            {
                ret = avcodec_receive_frame(codec_ctx, frame);
                if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF)
                {
                    break;
                }
                else if (ret < 0)
                {
                    std::cerr << "[H264Decode] Error during decoding" << std::endl;
                    break;
                }

                if (frame_cnt < sizez)
                { // Ignore z buffer
                    for (size_t x = 0; x < sizex; x++)
                    {
                        for (size_t y = 0; y < sizey; y++)
                        {
                            const size_t in_offset = (x * frame->linesize[0]) + y;
                            const size_t out_offset = (x * sizey * sizez) + (y * sizez) + frame_cnt;

                            out[out_offset] = frame->data[0][in_offset];
                        }
                    }
                }

                frame_cnt++;
            }
        }
    }

    while (true)
    {
        avcodec_send_packet(codec_ctx, nullptr);
        int ret = avcodec_receive_frame(codec_ctx, frame);
        if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF)
        {
            break;
        }
        else if (ret < 0)
        {
            std::cerr << "[H264Decode] Error during decoding" << std::endl;
            break;
        }

        if (frame_cnt < sizez)
        { // Ignore z buffer
            for (size_t x = 0; x < sizex; x++)
            {
                for (size_t y = 0; y < sizey; y++)
                {
                    const size_t in_offset = (x * frame->linesize[0]) + y;
                    const size_t out_offset = (x * sizey * sizez) + (y * sizez) + frame_cnt;

                    out[out_offset] = frame->data[0][in_offset];
                }
            }
        }

        frame_cnt++;
    }

    av_packet_unref(&packet);

    // Cleanup
    avformat_close_input(&formatContext);

    return (pixtype *)out;
}

pixtype *decode_stack_AV1(size_t sizex, size_t sizey, size_t sizez, void *buffer, size_t buffer_size)
{
    // Allocate output buffer
    uint8_t *out = (uint8_t *)calloc(sizex * sizey * sizez, sizeof(uint8_t));

    if (!buffer || buffer_size == 0)
    {
        std::cerr << "[AV1Decode] Failed to load buffer" << std::endl;
        return (pixtype *)out;
    }

    // Create memory buffer context
    FFmpegMemoryBuffer memBuffer((const uint8_t *)buffer, buffer_size);

    // Allocate AVFormatContext
    AVFormatContext *formatContext = avformat_alloc_context();
    if (!formatContext)
    {
        std::cerr << "[AV1Decode] Could not allocate format context" << std::endl;
        return (pixtype *)out;
    }

    // Create custom I/O context
    const size_t av_context_buffer_size = 4096;
    AVIOContext *ioContext = avio_alloc_context(
        static_cast<unsigned char *>(av_malloc(av_context_buffer_size)), // Internal buffer
        av_context_buffer_size,                                          // Buffer size
        0,                                                               // Write flag (0 for read-only)
        &memBuffer,                                                      // Opaque pointer
        memorybuffer_read_packet,                                        // Read callback
        nullptr,                                                         // Write callback (not needed)
        nullptr                                                          // Seek callback (optional)
    );

    if (!ioContext)
    {
        std::cerr << "[AV1Decode] Could not create I/O context" << std::endl;
        avformat_free_context(formatContext);
        return (pixtype *)out;
    }

    // Assign custom I/O context to format context
    formatContext->pb = ioContext;

    // Open input from memory buffer
    int ret = avformat_open_input(&formatContext, nullptr, nullptr, nullptr);
    if (ret < 0)
    {
        char errbuf[AV_ERROR_MAX_STRING_SIZE];
        av_strerror(ret, errbuf, AV_ERROR_MAX_STRING_SIZE);
        std::cerr << "[AV1Decode] Could not open input: " << errbuf << std::endl;

        // Cleanup
        avio_context_free(&ioContext);
        avformat_free_context(formatContext);
        return (pixtype *)out;
    }

    // Retrieve stream information
    ret = avformat_find_stream_info(formatContext, nullptr);
    if (ret < 0)
    {
        std::cerr << "[AV1Decode] Could not find stream information" << std::endl;

        // Cleanup
        avformat_close_input(&formatContext);
        return (pixtype *)out;
    }

    // Print some information about the media
    // av_dump_format(formatContext, 0, nullptr, 0);

    // Find video stream
    int video_stream_idx = -1;
    for (unsigned int i = 0; i < formatContext->nb_streams; ++i)
    {
        if (formatContext->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO)
        {
            video_stream_idx = i;
            break;
        }
    }

    if (video_stream_idx == -1)
    {
        std::cerr << "[AV1Decode] Could not find video stream" << std::endl;
        return (pixtype *)out;
    }

    // Get codec parameters and codec context
    AVCodecParameters *codecpar = formatContext->streams[video_stream_idx]->codecpar;
    AVCodec *codec = (AVCodec *)avcodec_find_decoder(codecpar->codec_id);
    AVCodecContext *codec_ctx = avcodec_alloc_context3(codec);
    avcodec_parameters_to_context(codec_ctx, codecpar);

    // Open codec
    if (avcodec_open2(codec_ctx, codec, nullptr) < 0)
    {
        std::cerr << "[AV1Decode] Could not open codec" << std::endl;
        return (pixtype *)out;
    }

    // Allocate frame and packet
    AVFrame *frame = av_frame_alloc();
    AVPacket packet;
    av_init_packet(&packet);

    size_t frame_cnt = 0;

    // Read frames from the video stream
    while (av_read_frame(formatContext, &packet) >= 0)
    {
        if (packet.stream_index == video_stream_idx)
        {
            // Decode video frame
            int ret = avcodec_send_packet(codec_ctx, &packet);
            if (ret < 0)
            {
                std::cerr << "[AV1Decode] Error sending packet for decoding" << std::endl;
                break;
            }

            while (ret >= 0)
            {
                ret = avcodec_receive_frame(codec_ctx, frame);
                if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF)
                {
                    break;
                }
                else if (ret < 0)
                {
                    std::cerr << "[AV1Decode] Error during decoding" << std::endl;
                    break;
                }

                if (frame_cnt < sizez)
                { // Ignore z buffer
                    for (size_t x = 0; x < sizex; x++)
                    {
                        for (size_t y = 0; y < sizey; y++)
                        {
                            const size_t in_offset = (x * frame->linesize[0]) + y;
                            const size_t out_offset = (x * sizey * sizez) + (y * sizez) + frame_cnt;

                            out[out_offset] = frame->data[0][in_offset];
                        }
                    }
                }

                frame_cnt++;
            }
        }
    }

    while (true)
    {
        avcodec_send_packet(codec_ctx, nullptr);
        int ret = avcodec_receive_frame(codec_ctx, frame);
        if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF)
        {
            break;
        }
        else if (ret < 0)
        {
            std::cerr << "[AV1Decode] Error during decoding" << std::endl;
            break;
        }

        if (frame_cnt < sizez)
        { // Ignore z buffer
            for (size_t x = 0; x < sizex; x++)
            {
                for (size_t y = 0; y < sizey; y++)
                {
                    const size_t in_offset = (x * frame->linesize[0]) + y;
                    const size_t out_offset = (x * sizey * sizez) + (y * sizez) + frame_cnt;

                    out[out_offset] = frame->data[0][in_offset];
                }
            }
        }

        frame_cnt++;
    }

    av_packet_unref(&packet);

    // Cleanup
    avformat_close_input(&formatContext);

    return (pixtype *)out;
}

pixtype *decode_stack_native(void *buffer, size_t buffer_size)
{
    void *out;

    size_t offset = 0;
    const uint8_t* byte_buffer = static_cast<const uint8_t*>(buffer);

    uint32_t metadata_size = *((uint32_t *)(byte_buffer + offset));
    offset += sizeof(uint32_t);
    uint32_t version = *((uint32_t *)(byte_buffer + offset));
    offset += sizeof(uint32_t);

    // TODO Check version match

    uint32_t enc_id = *((uint32_t *)(byte_buffer + offset));
    offset += sizeof(uint32_t);
    uint32_t dec_id = *((uint32_t *)(byte_buffer + offset));
    offset += sizeof(uint32_t);
    uint32_t width = *((uint32_t *)(byte_buffer + offset));
    offset += sizeof(uint32_t);
    uint32_t height = *((uint32_t *)(byte_buffer + offset));
    offset += sizeof(uint32_t);
    uint32_t depth = *((uint32_t *)(byte_buffer + offset));
    offset += sizeof(uint32_t);
    uint32_t bit_mode = *((uint32_t *)(byte_buffer + offset));
    offset += sizeof(uint32_t);
    uint32_t preset_id = *((uint32_t *)(byte_buffer + offset));
    offset += sizeof(uint32_t);
    uint32_t tune_id = *((uint32_t *)(byte_buffer + offset));
    offset += sizeof(uint32_t);
    uint32_t crf = *((uint32_t *)(byte_buffer + offset));
    offset += sizeof(uint32_t);
    uint32_t film_grain = *((uint32_t *)(byte_buffer + offset));
    offset += sizeof(uint32_t);
    uint32_t stored_gpu_id = *((uint32_t *)(byte_buffer + offset));
    offset += sizeof(uint32_t);

    uint64_t compressed_size = *((uint64_t *)(byte_buffer + offset));
    offset += sizeof(uint64_t);

    const unsigned int cd_values[11] = {
        enc_id,
        dec_id,
        width,
        height,
        depth,
        bit_mode,
        preset_id,
        tune_id,
        crf,
        film_grain,
        0 // TODO Fix
    };

    size_t buffer_leftover = buffer_size - offset;
    out = malloc(buffer_leftover);

    // TODO check if buffer_leftover == cd_values->compressed_size

    memcpy(out, byte_buffer + offset, buffer_leftover);

    size_t outsize = ffmpeg_native(!FFMPEG_FLAG_COMPRESS, cd_values, compressed_size, &out);

    return (pixtype *)out;
}
