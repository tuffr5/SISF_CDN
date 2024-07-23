#include "subprocess.hpp"

#include <cmath>

#include <sstream>
#include <iostream>
#include <thread>
#include <mutex>

#include "x264.h"

#define IMAGE_GAIN 10

std::mutex pthread_mutex;

std::string ffmpeg_location = "/home/loganaw/test/nTracer2_cdn3/ffmpeg-6.0-amd64-static/ffmpeg";
// std::string ffmpeg_location = "/Users/loganaw/Downloads/ffmpeg";
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

pixtype *decode_stack(size_t w, size_t h, size_t t, void *buffer, size_t buffer_size)
{
    const size_t frame_size = w * h * sizeof(pixtype);
    const size_t stack_size = frame_size * t;

    std::string cmd = ffmpeg_location + " " + decode_params;

    // Use subprocess
    subprocess::Popen *p = new subprocess::Popen(
        {"/bin/sh", "-c", cmd},
        subprocess::output{subprocess::PIPE},
        subprocess::input{subprocess::PIPE},
        subprocess::error{subprocess::PIPE},
        subprocess::bufsize{(int) 256 * 256 * 256 * 2});

    //std::cerr << "GOT " << buffer_size << std::endl;

    //{
    //    std::ofstream file("test.h256", std::ios::binary);
    //    file.write((char *)buffer, buffer_size);
    //    file.close();
    //}

/*
    const size_t send_chunk_size = 1000;
    for(size_t i = 0; i < buffer_size; i+=send_chunk_size) {
        char * buffer_offset = (char *) buffer;
        buffer_offset += i;

        std::cerr << "PREPPING " << i << std::endl;

        size_t tosend = std::min(send_chunk_size, buffer_size - i);

        size_t s = p->send(buffer_offset, tosend);

        std::cerr << "SENTT " << s << std::endl;
    }
    */

    //size_t s = p->send((char *)buffer, buffer_size);
    //std::cerr << "SENT " << s << std::endl;

    auto buf = p->communicate((char *) buffer, buffer_size).first.buf;

    //auto buf = p->communicate().first.buf;

    size_t buf_size = buf.size();
    //std::cerr << "Got " << buf_size << std::endl;
    void *out = malloc(buf_size);
    memcpy(out, buf.data(), buf_size);

    if (buf_size != stack_size)
        ;

    delete p;

    return (pixtype *)out;
}

std::string SVT_AV1_location_dec = "/home/loganaw/SVT-AV1/Bin/Release/SvtAv1DecApp";
std::pair<size_t, pixtype *> decode_stack_AV1_decapp(size_t w, size_t h, size_t t, void *buffer, size_t buffer_size)
{
    const size_t frame_size = w * h * sizeof(pixtype);
    const size_t stack_size = frame_size * t;

    // std::string cmd = ffmpeg_location + " " + decode_params + " " + null_redirect;

    std::string cmd = SVT_AV1_location_dec + " -i stdin -o stdout";

    // Use subprocess
    subprocess::Popen *p = new subprocess::Popen({"/bin/sh", "-c", cmd},
                                                 subprocess::output{subprocess::PIPE}, subprocess::input{subprocess::PIPE});

    p->send((char *)buffer, buffer_size);

    auto buf = p->communicate().first.buf;

    size_t buf_size = buf.size();
    // std::cout << "Got " << buf_size << std::endl;
    void *out = malloc(buf_size);
    memcpy(out, buf.data(), buf_size);

    if (buf_size != stack_size)
        ;

    delete p;

    return {buf_size, (pixtype *)out};
}

std::pair<size_t, void *> encode_stack(size_t w, size_t h, size_t t, pixtype *stack)
{
    size_t frame_size = w * h * sizeof(pixtype);
    size_t stack_size = frame_size * t;

    std::stringstream cmd_build;
    cmd_build << ffmpeg_location << " "
              << "-f rawvideo -framerate " << target_framerate << "/1 -pix_fmt gray -s " << w << "x" << h
              << " -i - -vcodec " << encoder_name << " -framerate " << target_framerate << "/1 " << other_encode_settings
              << " -f rawvideo -pix_fmt gray -" << null_redirect;

    // Use subprocess
    subprocess::Popen *p = new subprocess::Popen(
        {"/bin/sh", "-c", cmd_build.str()},
        //{"cat"},
        subprocess::output{subprocess::PIPE},
        subprocess::input{subprocess::PIPE},
        subprocess::error{subprocess::PIPE},
        subprocess::close_fds{false},
        subprocess::bufsize{(int) 256 * 256 * 256 * 2},
        subprocess::shell{false});

    auto [outb, errb] = p->communicate((char *)stack, stack_size);

    size_t buf_size = outb.buf.size();
    void *out = malloc(buf_size);
    memcpy(out, outb.buf.data(), buf_size);

    p->wait();
    p->kill();

    delete p;

    return {buf_size, out};
}

std::string SVT_AV1_location = "/home/loganaw/SVT-AV1/Bin/Release/SvtAv1EncApp";
std::pair<size_t, void *> encode_stack_AV1_encapp(size_t w, size_t h, size_t t, pixtype *stack, size_t stack_size_in)
{
    std::stringstream cmd_build;
    cmd_build << SVT_AV1_location << " "
            << "--crf 20 "
            << "--lp 4 "
            << "-i - "
            << "-w " << w << " "
            << "-h " << h << " "
            << "--fps " << target_framerate << " "
            << "-b -";

    // Use subprocess
    subprocess::Popen *p = new subprocess::Popen(
        {"/bin/sh", "-c", cmd_build.str()},
        subprocess::output{subprocess::PIPE},
        subprocess::input{subprocess::PIPE},
        subprocess::error{subprocess::PIPE},
        subprocess::close_fds{false},
        subprocess::bufsize{(int) (w * h * t * 2)},
        subprocess::shell{false});

    auto [outb, errb] = p->communicate((char *)stack, stack_size_in);

    size_t buf_size = outb.buf.size();
    void *out = malloc(buf_size);
    memcpy(out, outb.buf.data(), buf_size);

    p->wait();
    p->kill();

    delete p;

    return {buf_size, out};
}

std::pair<size_t, void *> encode_stack_x264(size_t w_in, size_t h_in, size_t t_in, pixtype *buffer_in)
{
    const int thread_count = 2;
    const int show_stderr = X264_LOG_NONE; // X264_LOG_INFO

    x264_param_t param;
    x264_picture_t pic;
    x264_picture_t pic_out;
    x264_t *h;

    int i_frame = 0;
    int i_frame_size;
    x264_nal_t *nal;
    int i_nal;

    /* Get default params for preset/tuning */
    if (x264_param_default_preset(&param, "slower", NULL) < 0)
    {
        std::cerr << "Failed to set preset!" << std::endl;
        // return -1;
    }

    /* Configure non-default params */
    param.i_threads = thread_count;
    param.i_lookahead_threads = thread_count;
    param.i_bitdepth = 8;
    param.i_csp = X264_CSP_I400;
    param.i_width = w_in;
    param.i_height = h_in;
    param.i_fps_num = 24;
    param.i_fps_den = 1;
    param.b_vfr_input = 0;
    param.b_repeat_headers = 1;
    param.b_annexb = 1;
    param.b_deterministic = 1;
    param.i_log_level = show_stderr;

    param.rc.i_rc_method = X264_RC_CRF;
    int crf_qp_val = 10;
    param.rc.i_qp_constant = crf_qp_val;
    param.rc.i_qp_min = crf_qp_val;
    param.rc.i_qp_max = crf_qp_val;

    /* Apply profile restrictions. */
    if (x264_param_apply_profile(&param, "high") < 0)
    {
        // return -1;
    }

    if (x264_picture_alloc(&pic, param.i_csp, param.i_width, param.i_height) < 0)
    {
        // return -1;
    }

    h = x264_encoder_open(&param);
    if (!h)
    {
        x264_picture_clean(&pic);
        // return -1;
    }

    int luma_size = param.i_width * param.i_height;

    char *out = (char *)malloc(luma_size * t_in);
    size_t out_offset = 0;

    /* Encode frames */
    for (size_t i_frame = 0; i_frame < t_in; i_frame++)
    {
        /* Read input frame */
        // memset(pic.img.plane[0], (int)i_frame, luma_size);
        memcpy(pic.img.plane[0], buffer_in + (luma_size * i_frame), luma_size);

        pic.i_pts = i_frame;
        i_frame_size = x264_encoder_encode(h, &nal, &i_nal, &pic, &pic_out);
        if (i_frame_size < 0)
        {
            // return kill_stream(h, &pic);
        }
        else if (i_frame_size)
        {
            memcpy(out + out_offset, nal->p_payload, i_frame_size);
            out_offset += i_frame_size;
        }
    }

    /* Flush delayed frames */
    while (x264_encoder_delayed_frames(h))
    {
        i_frame_size = x264_encoder_encode(h, &nal, &i_nal, NULL, &pic_out);
        if (i_frame_size < 0)
        {
            // return kill_stream(h, &pic);
        }
        else if (i_frame_size)
        {
            memcpy(out + out_offset, nal->p_payload, i_frame_size);
            out_offset += i_frame_size;
        }
    }

    out = (char *)realloc(out, out_offset);

    x264_encoder_close(h);
    x264_picture_clean(&pic);

    return {out_offset, out};
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
