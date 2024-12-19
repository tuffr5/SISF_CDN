#include <glob.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>

#include <iostream>
#include <sstream>
#include <fstream>
#include <vector>
#include <tuple>
#include <string>
#include <algorithm>
#include <queue>
#include <deque>
#include <mutex>
#include <map>
#include <chrono>

#include "vidlib.hpp"

#include "../zstd/lib/zstd.h"

#define CHUNK_TIMER 0
#define DEBUG_SLICING 0

size_t mchunk_uuid = 0;
std::mutex mchunk_uuid_mutex;

struct global_chunk_line
{
    size_t mchunk;
    size_t chunk;
    uint16_t *ptr;
};

std::chrono::duration cache_lock_timeout = std::chrono::milliseconds(10);

std::timed_mutex global_chunk_cache_mutex;
size_t global_cache_size = 5000;
global_chunk_line *global_chunk_cache = (global_chunk_line *)calloc(global_cache_size, sizeof(global_chunk_line));
size_t global_chunk_cache_last = 0;

// https://stackoverflow.com/questions/8401777/simple-glob-in-c-on-unix-system
std::vector<std::string> glob_tool(const std::string &pattern)
{
    std::vector<std::string> filenames;

    // glob struct resides on the stack
    glob_t glob_result;
    memset(&glob_result, 0, sizeof(glob_result));

    // do the glob operation
    const int glob_flag = GLOB_NOSORT | GLOB_TILDE;
    int return_value = glob(pattern.c_str(), glob_flag, NULL, &glob_result);

    if (return_value == GLOB_NOMATCH)
    {
        // Return nothing if no results found
        globfree(&glob_result);
        return filenames;
    }

    if (return_value != 0)
    {
        // Fail if another error code is found
        globfree(&glob_result);
        std::stringstream ss;
        ss << "glob() failed with return_value " << return_value << std::endl;
        throw std::runtime_error(ss.str());
    }

    // collect all the filenames into a std::list<std::string>
    for (size_t i = 0; i < glob_result.gl_pathc; ++i)
    {
        filenames.push_back(std::string(glob_result.gl_pathv[i]));
    }

    std::sort(filenames.begin(), filenames.end());

    globfree(&glob_result);
    return filenames;
}

struct metadata_entry
{
    uint64_t offset;
    uint32_t size;
};

class packed_reader
{
private:
    const size_t entry_file_line_size = 8 + 4;

public:
    bool found;
    uint16_t channel_count;
    uint16_t dtype, version;
    uint16_t compression_type;
    uint16_t chunkx, chunky, chunkz;
    uint64_t sizex, sizey, sizez;
    uint64_t countx, county, countz;

    uint64_t cropstartx, cropstarty, cropstartz;
    uint64_t cropendx, cropendy, cropendz;

    size_t max_chunk_size;
    size_t header_size;

    size_t this_mchunk_id;

    std::string meta_fname, data_fname;

    packed_reader(size_t chunk_id, std::string metadata_fname_in, std::string data_fname_in)
    {
        this_mchunk_id = chunk_id;
        meta_fname = metadata_fname_in;
        data_fname = data_fname_in;

        std::ifstream file(meta_fname, std::ios::in | std::ios::binary);

        //if (file.fail())
        //{
        //    std::cerr << "Fopen failed (chunk metadata)" << std::endl;
        //    found = false
        //    return;
        //}

        found = true;

        file.read((char *)&version, sizeof(uint16_t));
        file.read((char *)&dtype, sizeof(uint16_t));
        file.read((char *)&channel_count, sizeof(uint16_t));
        file.read((char *)&compression_type, sizeof(uint16_t));

        file.read((char *)&chunkx, sizeof(uint16_t));
        file.read((char *)&chunky, sizeof(uint16_t));
        file.read((char *)&chunkz, sizeof(uint16_t));
        file.read((char *)&sizex, sizeof(uint64_t));
        file.read((char *)&sizey, sizeof(uint64_t));
        file.read((char *)&sizez, sizeof(uint64_t));

        file.read((char *)&cropstartx, sizeof(uint64_t));
        file.read((char *)&cropendx, sizeof(uint64_t));
        file.read((char *)&cropstarty, sizeof(uint64_t));
        file.read((char *)&cropendy, sizeof(uint64_t));
        file.read((char *)&cropstartz, sizeof(uint64_t));
        file.read((char *)&cropendz, sizeof(uint64_t));

        countx = (sizex + ((size_t)chunkx) - 1) / ((size_t)chunkx);
        county = (sizey + ((size_t)chunky) - 1) / ((size_t)chunky);
        countz = (sizez + ((size_t)chunkz) - 1) / ((size_t)chunkz);

        max_chunk_size = chunkx * chunky * chunkz * sizeof(uint16_t);
        max_chunk_size *= channel_count;

        header_size = file.tellg();
        file.close();
    }

    ~packed_reader()
    {
    }

    size_t find_index(size_t x, size_t y, size_t z)
    {
        size_t ix = x / chunkx;
        size_t iy = y / chunky;
        size_t iz = z / chunkz;

        return (ix * countz * county) + (iy * countz) + iz;
    }

    metadata_entry *load_meta_entry(size_t id)
    {
        metadata_entry *out = (metadata_entry *)malloc(sizeof(metadata_entry));
        out->offset = 0;
        out->size = 0;

        const size_t offset = header_size + (entry_file_line_size * id);

        const size_t retry_count = 5;
        for (size_t i = 0; i < retry_count; i++)
        {
            std::ifstream file(meta_fname, std::ios::in | std::ios::binary);

            if (file.fail())
            {
                std::cerr << "Fopen failed (metadata)" << std::endl;
                continue;
            }

            file.seekg(offset);
            file.read((char *)&(out->offset), sizeof(uint64_t));
            file.read((char *)&(out->size), sizeof(uint32_t));
            file.close();
            break;
        }

        return out;
    }

    std::mutex chunk_cache_mutex;
    std::deque<std::tuple<size_t, uint16_t *>> chunk_cache;

    uint16_t *load_chunk(size_t id)
    {
        uint16_t *out = (uint16_t *)calloc(max_chunk_size, 1);
        metadata_entry *sel = load_meta_entry(id);

        if (sel->size == 0)
        {
            // Failed to read metadata (impossible for chunk size to be 0)
            free(sel);
            return out;
        }

        uint16_t *from_cache = 0;

        if (global_chunk_cache_mutex.try_lock_for(cache_lock_timeout))
        {
            for (size_t i = 0; i < global_cache_size; i++)
            {
                if (global_chunk_cache[i].chunk == id)
                {
                    if (global_chunk_cache[i].mchunk == this_mchunk_id)
                    {
                        from_cache = global_chunk_cache[i].ptr;
                        break;
                    }
                }
            }

            // Copy from cache
            if (from_cache != 0)
            {
                memcpy((void *)out, (void *)from_cache, max_chunk_size);
            }

            global_chunk_cache_mutex.unlock();
        }

        // Either from_cache has the chunk, or was not in cache, or failed to get lock

        if (from_cache == 0)
        {
            // Read from file
            size_t buffer_size = sel->size;
            uint16_t *read_buffer = (uint16_t *)malloc(buffer_size);

            const size_t retry_count = 10;
            for (size_t i = 0; i < retry_count; i++)
            {
                std::ifstream file(data_fname, std::ios::in | std::ios::binary);

                if (file.fail())
                {
                    std::cerr << "Fopen failed" << std::endl;
                    continue;
                }

                file.seekg(sel->offset);
                file.read((char *)read_buffer, sel->size);
                file.close();
                break;
            }

            // Decompress
            size_t decomp_size;
            char *read_decomp_buffer;
            pixtype *read_decomp_buffer_pt;

            // 1 -> zstd
            // 2 -> 264
            // 3 -> AV1
            switch (compression_type)
            {
            case 1:
                // Decompress with ZSTD
                read_decomp_buffer = (char *)calloc(max_chunk_size, 1);
                decomp_size = ZSTD_decompress(read_decomp_buffer, max_chunk_size, read_buffer, sel->size);
                break;

            case 2:
                // Decompress with vidlib
                read_decomp_buffer_pt = decode_stack_264(chunkx, chunky, chunkz, read_buffer, sel->size);
                read_decomp_buffer = (char *)pixtype_to_uint16(read_decomp_buffer_pt, chunkx * chunky * chunkz);
                free(read_decomp_buffer_pt);
                break;

            case 3:
                // Decompress with vidlib 2
                read_decomp_buffer_pt = decode_stack_AV1(chunkx, chunky, chunkz, read_buffer, sel->size);
                // read_decomp_buffer = (char *)pixtype_to_uint16_YUV420(read_decomp_buffer_pt, chunkx, chunky, chunkz);
                read_decomp_buffer = (char *)pixtype_to_uint16(read_decomp_buffer_pt, chunkx * chunky * chunkz);
                free(read_decomp_buffer_pt);
                break;
            }

            free(read_buffer);

            // Copy result
            memcpy((void *)out, (void *)read_decomp_buffer, max_chunk_size);

            if (global_chunk_cache_mutex.try_lock_for(cache_lock_timeout))
            {
                if (global_chunk_cache[global_chunk_cache_last].ptr != 0)
                    free(global_chunk_cache[global_chunk_cache_last].ptr);

                global_chunk_cache[global_chunk_cache_last].chunk = (size_t)id;
                global_chunk_cache[global_chunk_cache_last].mchunk = (size_t)this_mchunk_id;
                global_chunk_cache[global_chunk_cache_last].ptr = (uint16_t *)read_decomp_buffer;

                global_chunk_cache_last++;
                if (global_chunk_cache_last == global_cache_size)
                {
                    global_chunk_cache_last = 0;
                }

                global_chunk_cache_mutex.unlock();
            }
            else
            {
                free(read_decomp_buffer);
            }
        }

        free(sel);

        return out;
    }

    uint16_t read_pixel(size_t i, size_t j, size_t k)
    {
        const size_t xmin = ((size_t)chunkx) * (i / ((size_t)chunkx));
        const size_t xmax = std::min((size_t)xmin + chunkx, (size_t)sizex);
        const size_t xsize = xmax - xmin;

        const size_t ymin = ((size_t)chunky) * (j / ((size_t)chunky));
        const size_t ymax = std::min((size_t)ymin + chunky, (size_t)sizey);
        const size_t ysize = ymax - ymin;

        const size_t zmin = ((size_t)chunkz) * (k / ((size_t)chunkz));
        const size_t zmax = std::min((size_t)zmin + chunkz, (size_t)sizez);
        const size_t zsize = zmax - zmin;

        //                   X                              Y                    Z
        const size_t coffset = ((i - xmin) * ysize * zsize) + ((j - ymin) * zsize) + (k - zmin);

        const size_t chunk_id = find_index(i, j, k);

        uint16_t *chunk = load_chunk(chunk_id);

        const uint16_t out = chunk[coffset];

        free(chunk);

        return out;
    }

    void print_info()
    {
        std::cout << "-----------------------------------" << std::endl;
        std::cout << "Files: " << meta_fname << ", " << data_fname << std::endl;
        std::cout << "dtype = " << dtype << std::endl;
        std::cout << "Chunks: " << chunkx << ", " << chunky << ", " << chunkz << std::endl;
        std::cout << "Size: " << sizex << ", " << sizey << ", " << sizez << std::endl;
        std::cout << "Tile Counts: " << countx << ", " << county << ", " << countz << std::endl;
        std::cout << "-----------------------------------" << std::endl;
    }
};

class archive_reader
{
public:
    std::string fname; // "./example_dset"
    uint16_t channel_count;
    uint16_t archive_version; // 1 == current
    uint16_t dtype;
    uint16_t mchunkx, mchunky, mchunkz;
    uint64_t resx, resy, resz;
    uint64_t sizex, sizey, sizez;
    uint64_t mcountx, mcounty, mcountz;

    std::vector<size_t> scales;

    archive_reader(std::string name_in)
    {
        fname = name_in;
        load_metadata();
    }

    ~archive_reader() {}

    void load_metadata()
    {
        std::ifstream file(fname + "/metadata.bin", std::ios::in | std::ios::binary);

        if (file.fail())
        {
            ;
        }

        file.read((char *)&archive_version, sizeof(uint16_t));
        file.read((char *)&dtype, sizeof(uint16_t));
        file.read((char *)&channel_count, sizeof(uint16_t));

        file.read((char *)&mchunkx, sizeof(uint16_t));
        file.read((char *)&mchunky, sizeof(uint16_t));
        file.read((char *)&mchunkz, sizeof(uint16_t));
        file.read((char *)&resx, sizeof(uint64_t));
        file.read((char *)&resy, sizeof(uint64_t));
        file.read((char *)&resz, sizeof(uint64_t));
        file.read((char *)&sizex, sizeof(uint64_t));
        file.read((char *)&sizey, sizeof(uint64_t));
        file.read((char *)&sizez, sizeof(uint64_t));

        mcountx = (sizex + mchunkx - 1) / mchunkx;
        mcounty = (sizey + mchunky - 1) / mchunky;
        mcountz = (sizez + mchunkz - 1) / mchunkz;

        // Find resolution tiers
        std::vector<std::string> fnames = glob_tool(std::string(fname + "/meta/*.meta"));

        for (std::vector<std::string>::iterator i = fnames.begin(); i != fnames.end(); i++)
        {
            size_t loc1 = i->find_last_of("/");
            size_t loc2 = i->find_last_of('.');

            const size_t label_offset = 6; // "chunk_"

            std::string ii = std::string(i->c_str() + loc1 + label_offset + 1, i->c_str() + loc2);

            std::string scale_label = ii.substr(ii.find_last_of('.') + 1);
            const size_t scale = stoi(scale_label);
            const size_t cnt = std::count(scales.begin(), scales.end(), scale);

            if (cnt == 0)
            {
                scales.push_back(scale);
            }
        }

        std::sort(scales.begin(), scales.end());
    }

    std::tuple<size_t, size_t, size_t> get_size(size_t scale)
    {
        size_t size_x_out = sizex / scale;
        size_t size_y_out = sizey / scale;
        size_t size_z_out = sizez / scale;

        // Start with the size of one chunk
        double dilation_x = this->mchunkx;
        double dilation_y = this->mchunky;
        double dilation_z = this->mchunkz;

        // Divide by scale
        dilation_x /= scale;
        dilation_y /= scale;
        dilation_z /= scale;

        // Isolate fractional part of dilation
        dilation_x -= std::floor(dilation_x);
        dilation_y -= std::floor(dilation_y);
        dilation_z -= std::floor(dilation_z);

        // Scale dilation by number of tiles
        dilation_x *= this->mcountx;
        dilation_y *= this->mcounty;
        dilation_z *= this->mcountz;

        // round up
        dilation_x = std::ceil(dilation_x);
        dilation_y = std::ceil(dilation_y);
        dilation_z = std::ceil(dilation_z);

        size_x_out -= dilation_x;
        size_y_out -= dilation_y;
        size_z_out -= dilation_z;

        // std::cout << "Scale: " << scale << " " << dilation_x << " " << dilation_y << " " << dilation_z << std::endl;

        return std::make_tuple(size_x_out, size_y_out, size_z_out);
    }

    std::tuple<size_t, size_t, size_t> get_res(size_t scale)
    {
        std::tuple<size_t, size_t, size_t> size = this->get_size(scale);

        double resx_out = resx;
        double resy_out = resy;
        double resz_out = resz;

        resx_out *= sizex;
        resy_out *= sizey;
        resz_out *= sizez;

        resx_out /= std::get<0>(size);
        resy_out /= std::get<1>(size);
        resz_out /= std::get<2>(size);

        return std::make_tuple((size_t)resx_out, (size_t)resy_out, (size_t)resz_out);
    }

    std::tuple<size_t, size_t, size_t> inline find_index(size_t x, size_t y, size_t z)
    {
        size_t ix = x / mchunkx;
        size_t iy = y / mchunky;
        size_t iz = z / mchunkz;

        return std::make_tuple(ix, iy, iz);
    }

    size_t inline pixel_size()
    {
        return sizeof(uint16_t);
    }

    std::map<std::tuple<size_t, size_t, size_t, size_t, size_t>, packed_reader *> mchunk_buffer;
    std::mutex mchunk_buffer_mutex;
    packed_reader *get_mchunk(size_t scale, size_t channel, size_t i, size_t j, size_t k)
    {
        std::tuple<size_t, size_t, size_t, size_t, size_t> id_tuple = std::make_tuple(scale, channel, i, j, k);

        mchunk_buffer_mutex.lock();
        packed_reader *out = mchunk_buffer[id_tuple];

        if (out == 0)
        {
            std::stringstream ss;
            ss << "chunk_" << i << '_' << j << '_' << k << '.' << channel << '.' << scale << 'X';
            const std::string chunk_root = ss.str();

            const std::string chunk_meta_name = fname + "/meta/" + chunk_root + ".meta";
            const std::string chunk_data_name = fname + "/data/" + chunk_root + ".data";

            mchunk_uuid_mutex.lock();
            size_t random_id = mchunk_uuid;
            mchunk_uuid++;
            mchunk_uuid_mutex.unlock();

            out = new packed_reader(random_id, chunk_meta_name, chunk_data_name);

            mchunk_buffer[id_tuple] = out;
        }
        mchunk_buffer_mutex.unlock();

        return out;
    }

    uint16_t *load_region(
        size_t scale,
        size_t xs, size_t xe,
        size_t ys, size_t ye,
        size_t zs, size_t ze)
    {
        std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

        // Calculate size of output
        const size_t osizex = xe - xs;
        const size_t osizey = ye - ys;
        const size_t osizez = ze - zs;
        const size_t buffer_size = osizex * osizey * osizez * sizeof(uint16_t) * channel_count;

        // Allocate buffer for output
        uint16_t *out_buffer = (uint16_t *)malloc(buffer_size);

        // Define map for storing already decompressed chunks
        std::map<std::tuple<size_t, size_t, size_t, size_t, size_t>, uint16_t *> chunk_cache;

        // Scaled metachunk size
        const size_t mcx = mchunkx / scale;
        const size_t mcy = mchunky / scale;
        const size_t mcz = mchunkz / scale;

        // Variables to store chunk reader and data (shared in loop)
        packed_reader *chunk_reader = nullptr;
        std::tuple<size_t, size_t, size_t, size_t, size_t> *chunk_identifier = nullptr;
        size_t sub_chunk_id;
        uint16_t *chunk;

        // Variables for tracking the last chunks that were used
        size_t last_x, last_y, last_z, last_sub, last_c;
        size_t cxmin, cxmax, cxsize;
        size_t cymin, cymax, cysize;
        size_t czmin, czmax, czsize;

        for (size_t c = 0; c < channel_count; c++)
        {
            for (size_t i = xs; i < xe; i++)
            {
                const size_t xmin = mcx * (i / mcx);                             // lower bound of mchunk
                const size_t xmax = std::min((size_t)xmin + mcx, (size_t)sizex); // upper bound of mchunk
                const size_t xsize = xmax - xmin;                                // size of mchunk
                const size_t chunk_id_x = i / ((size_t)mcx);                     // mchunk x id
                const size_t x_in_chunk = i - xmin;                              // x displacement inside chunk

                for (size_t j = ys; j < ye; j++)
                {
                    const size_t ymin = mcy * (j / mcy);
                    const size_t ymax = std::min((size_t)ymin + mcy, (size_t)sizey);
                    const size_t ysize = ymax - ymin;
                    const size_t chunk_id_y = j / ((size_t)mcy);
                    const size_t y_in_chunk = j - ymin;

                    for (size_t k = zs; k < ze; k++)
                    {
                        const size_t zmin = mcz * (k / mcz);
                        const size_t zmax = std::min((size_t)zmin + mcz, (size_t)sizez);
                        const size_t zsize = zmax - zmin;
                        const size_t chunk_id_z = k / ((size_t)mcz);
                        const size_t z_in_chunk = k - zmin;

                        bool force = false;
                        if (chunk_reader == nullptr ||
                            chunk_identifier == nullptr ||
                            last_x != chunk_id_x ||
                            last_y != chunk_id_y ||
                            last_z != chunk_id_z ||
                            last_c != c)
                        {
                            force = true;
                            chunk_reader = get_mchunk(scale, c, chunk_id_x, chunk_id_y, chunk_id_z);

                            last_x = chunk_id_x;
                            last_y = chunk_id_y;
                            last_z = chunk_id_z;
                        }

                        // Shift ranges for cropping
                        const size_t x_in_chunk_offset = x_in_chunk + chunk_reader->cropstartx;
                        const size_t y_in_chunk_offset = y_in_chunk + chunk_reader->cropstarty;
                        const size_t z_in_chunk_offset = z_in_chunk + chunk_reader->cropstartz;

                        // Find sub chunk id from coordinates
                        sub_chunk_id = chunk_reader->find_index(x_in_chunk_offset, y_in_chunk_offset, z_in_chunk_offset);

                        // Only perform this step if there has been a change in chunk
                        if (force ||
                            last_sub != sub_chunk_id)
                        {
                            // Replace the chunk id with the new one
                            if (chunk_identifier != nullptr)
                            {
                                delete chunk_identifier;
                            }
                            chunk_identifier = new std::tuple(c, chunk_id_x, chunk_id_y, chunk_id_z, sub_chunk_id);

                            // Check if the chunk is in the tmp cache
                            chunk = chunk_cache[*chunk_identifier];
                            if (chunk == 0)
                            {
                                chunk = chunk_reader->load_chunk(sub_chunk_id);
                                chunk_cache[*chunk_identifier] = chunk;
                            }

                            // Find the start/stop coordinates of this chunk
                            cxmin = ((size_t)chunk_reader->chunkx) * (x_in_chunk_offset / ((size_t)chunk_reader->chunkx)); // Minimum value of the chunk
                            cxmax = std::min((size_t)cxmin + chunk_reader->chunkx, (size_t)chunk_reader->sizex);           // Maximum value of the chunk
                            cxsize = cxmax - cxmin;                                                                        // Size of the chunk

                            cymin = ((size_t)chunk_reader->chunky) * (y_in_chunk_offset / ((size_t)chunk_reader->chunky));
                            cymax = std::min((size_t)cymin + chunk_reader->chunky, (size_t)chunk_reader->sizey);
                            cysize = cymax - cymin;

                            czmin = ((size_t)chunk_reader->chunkz) * (z_in_chunk_offset / ((size_t)chunk_reader->chunkz));
                            czmax = std::min((size_t)czmin + chunk_reader->chunkz, (size_t)chunk_reader->sizez);
                            czsize = czmax - czmin;

                            // Store this ID as the most recent chunk
                            last_sub = sub_chunk_id;
                            last_c = c;
                        }

                        // Calculate the coordinates of the input and output inside their respective buffers
                        const size_t coffset = ((x_in_chunk_offset - cxmin) * cysize * czsize) + // X
                                               ((y_in_chunk_offset - cymin) * czsize) +          // Y
                                               (z_in_chunk_offset - czmin);                      // Z

                        const size_t ooffset = (c * osizey * osizex * osizez) + // C
                                               ((k - zs) * osizey * osizex) +   // Z
                                               ((j - ys) * osizex) +            // Y
                                               ((i - xs));                      // X

                        out_buffer[ooffset] = chunk[coffset];
                    }
                }
            }
        }

        if (chunk_identifier != nullptr)
        {
            delete chunk_identifier;
        }

        for (auto it = chunk_cache.begin(); it != chunk_cache.end(); it++)
        {
            free(it->second);
        }

        if (CHUNK_TIMER)
        {
            std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
            size_t dt = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
            std::cout << "Time difference = " << dt << " [us]" << std::endl;
        }

        return out_buffer;
    }

    void print_info()
    {
        std::cout << "-----------------------------------" << std::endl;
        std::cout << "File: " << fname << std::endl;
        std::cout << "dtype = " << dtype << "; version = " << archive_version << ";" << std::endl;
        std::cout << "Chunks: " << mchunkx << ", " << mchunky << ", " << mchunkz << std::endl;
        std::cout << "Size: " << sizex << ", " << sizey << ", " << sizez << std::endl;
        std::cout << "Tile Counts: " << mcountx << ", " << mcounty << ", " << mcountz << std::endl;
        std::cout << "Scales: ";
        for (size_t n : scales)
            std::cout << n << ' ';
        std::cout << "Channels = " << channel_count << std::endl;
        std::cout << std::endl;
        std::cout << "-----------------------------------" << std::endl;
    }
};