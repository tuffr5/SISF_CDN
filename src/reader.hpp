#include <glob.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>

#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

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
#include <future>
#include <map>
#include <chrono>
#include <set>

#include <nlohmann/json.hpp>

#include "vidlib.hpp"

#include "../zstd/lib/zstd.h"

#include "absl/flags/flag.h"
#include "absl/flags/marshalling.h"
#include "absl/status/status.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_split.h"
#include <half.hpp>

#include "tensorstore/array.h"
#include "tensorstore/context.h"
#include "tensorstore/contiguous_layout.h"
#include "tensorstore/data_type.h"
#include "examples/data_type_invoker.h"
#include "absl/flags/parse.h"
#include "tensorstore/index.h"
#include "tensorstore/index_space/dim_expression.h"
#include "tensorstore/index_space/index_transform.h"
#include "tensorstore/index_space/transformed_array.h"
#include "tensorstore/index_space/index_transform_builder.h"
#include "tensorstore/open.h"
#include "tensorstore/read_write_options.h"
#include "tensorstore/open_mode.h"
#include "tensorstore/progress.h"
#include "tensorstore/rank.h"
#include "tensorstore/spec.h"
#include "tensorstore/tensorstore.h"
#include "tensorstore/util/future.h"
#include "tensorstore/util/iterate_over_index_range.h"
#include "tensorstore/util/json_absl_flag.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/span.h"
#include "tensorstore/util/status.h"
#include "tensorstore/util/str_cat.h"
#include "tensorstore/util/utf8_string.h"

#define CHUNK_TIMER 0
#define DEBUG_SLICING 0
#define IO_RETRY_COUNT 5

size_t mchunk_uuid = 0;
std::mutex mchunk_uuid_mutex;

struct global_chunk_line
{
    size_t mchunk;
    size_t chunk;
    uint16_t *ptr;
};

enum ArchiveType
{
    SISF_JSON,
    SISF,
    ZARR,
    DESCRIPTOR
};

using json = nlohmann::json;

std::chrono::duration cache_lock_timeout = std::chrono::milliseconds(10);

std::timed_mutex global_chunk_cache_mutex;
size_t global_cache_size = 1000;
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
    const size_t header_size_expected = sizeof(uint16_t) * 7 + sizeof(uint64_t) * 9;

public:
    bool is_valid;
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

    // mmap for data file - eliminates file open/close overhead
    void *mmap_data_ptr = nullptr;
    size_t mmap_data_size = 0;
    int mmap_data_fd = -1;

    packed_reader(size_t chunk_id, std::string metadata_fname_in, std::string data_fname_in)
    {
        is_valid = false;
        this_mchunk_id = chunk_id;
        meta_fname = metadata_fname_in;
        data_fname = data_fname_in;

        std::ifstream file(meta_fname, std::ios::in | std::ios::binary);

        if (file.fail())
        {
            std::cerr << "Fopen failed (chunk metadata)" << std::endl;
            return;
        }

        std::streamsize bytes_read = 0;
        file.read((char *)&version, sizeof(uint16_t));
        bytes_read += file.gcount();
        file.read((char *)&dtype, sizeof(uint16_t));
        bytes_read += file.gcount();
        file.read((char *)&channel_count, sizeof(uint16_t));
        bytes_read += file.gcount();
        file.read((char *)&compression_type, sizeof(uint16_t));
        bytes_read += file.gcount();

        file.read((char *)&chunkx, sizeof(uint16_t));
        bytes_read += file.gcount();
        file.read((char *)&chunky, sizeof(uint16_t));
        bytes_read += file.gcount();
        file.read((char *)&chunkz, sizeof(uint16_t));
        bytes_read += file.gcount();
        file.read((char *)&sizex, sizeof(uint64_t));
        bytes_read += file.gcount();
        file.read((char *)&sizey, sizeof(uint64_t));
        bytes_read += file.gcount();
        file.read((char *)&sizez, sizeof(uint64_t));
        bytes_read += file.gcount();

        file.read((char *)&cropstartx, sizeof(uint64_t));
        bytes_read += file.gcount();
        file.read((char *)&cropendx, sizeof(uint64_t));
        bytes_read += file.gcount();
        file.read((char *)&cropstarty, sizeof(uint64_t));
        bytes_read += file.gcount();
        file.read((char *)&cropendy, sizeof(uint64_t));
        bytes_read += file.gcount();
        file.read((char *)&cropstartz, sizeof(uint64_t));
        bytes_read += file.gcount();
        file.read((char *)&cropendz, sizeof(uint64_t));
        bytes_read += file.gcount();

        header_size = file.tellg();
        file.close();

        if (header_size != header_size_expected || bytes_read != header_size_expected)
        {
            std::cerr << "Metadata read failed (short read)" << std::endl;
            return;
        }

        countx = (sizex + ((size_t)chunkx) - 1) / ((size_t)chunkx);
        county = (sizey + ((size_t)chunky) - 1) / ((size_t)chunky);
        countz = (sizez + ((size_t)chunkz) - 1) / ((size_t)chunkz);

        max_chunk_size = channel_count * chunkx * chunky * chunkz * sizeof(uint16_t);

        // Set up mmap for data file
        mmap_data_fd = open(data_fname.c_str(), O_RDONLY);
        if (mmap_data_fd != -1)
        {
            struct stat st;
            if (fstat(mmap_data_fd, &st) == 0)
            {
                mmap_data_size = st.st_size;
                mmap_data_ptr = mmap(nullptr, mmap_data_size, PROT_READ, MAP_PRIVATE, mmap_data_fd, 0);
                if (mmap_data_ptr == MAP_FAILED)
                {
                    std::cerr << "mmap failed for data file, falling back to ifstream" << std::endl;
                    mmap_data_ptr = nullptr;
                    close(mmap_data_fd);
                    mmap_data_fd = -1;
                }
                else
                {
                    // Advise kernel for random access pattern
                    madvise(mmap_data_ptr, mmap_data_size, MADV_RANDOM);
                }
            }
            else
            {
                close(mmap_data_fd);
                mmap_data_fd = -1;
            }
        }

        is_valid = true;
    }

    ~packed_reader()
    {
        // Clean up mmap
        if (mmap_data_ptr != nullptr && mmap_data_ptr != MAP_FAILED)
        {
            munmap(mmap_data_ptr, mmap_data_size);
        }
        if (mmap_data_fd != -1)
        {
            close(mmap_data_fd);
        }
    }

    // Call this after data file is updated (e.g., after patch endpoint)
    // Thread-safe: blocks reads briefly during remap
    std::mutex mmap_refresh_mutex;

    void refresh_mmap()
    {
        std::lock_guard<std::mutex> lock(mmap_refresh_mutex);

        // Unmap old mapping
        if (mmap_data_ptr != nullptr && mmap_data_ptr != MAP_FAILED)
        {
            munmap(mmap_data_ptr, mmap_data_size);
            mmap_data_ptr = nullptr;
        }
        if (mmap_data_fd != -1)
        {
            close(mmap_data_fd);
            mmap_data_fd = -1;
        }

        // Remap the file (may have new size)
        mmap_data_fd = open(data_fname.c_str(), O_RDONLY);
        if (mmap_data_fd != -1)
        {
            struct stat st;
            if (fstat(mmap_data_fd, &st) == 0)
            {
                mmap_data_size = st.st_size;
                mmap_data_ptr = mmap(nullptr, mmap_data_size, PROT_READ, MAP_PRIVATE, mmap_data_fd, 0);
                if (mmap_data_ptr == MAP_FAILED)
                {
                    std::cerr << "mmap refresh failed, falling back to ifstream" << std::endl;
                    mmap_data_ptr = nullptr;
                    close(mmap_data_fd);
                    mmap_data_fd = -1;
                }
                else
                {
                    madvise(mmap_data_ptr, mmap_data_size, MADV_RANDOM);
                }
            }
            else
            {
                close(mmap_data_fd);
                mmap_data_fd = -1;
            }
        }
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

        for (size_t i = 0; i < IO_RETRY_COUNT; i++)
        {
            std::ifstream file(meta_fname, std::ios::in | std::ios::binary);

            if (file.fail())
            {
                std::cerr << "Fopen failed (metadata)" << std::endl;
                continue;
            }

            file.seekg(offset);
            file.read((char *)&(out->offset), sizeof(uint64_t));
            std::streamsize bytes_read = file.gcount();
            file.read((char *)&(out->size), sizeof(uint32_t));
            bytes_read += file.gcount();
            file.close();

            if (bytes_read != sizeof(uint32_t) + sizeof(uint64_t))
            {
                std::cerr << "Metadata read failed (short read)" << std::endl;
                continue;
            }

            break;
        }

        return out;
    }

    void replace_meta_entry(size_t id, metadata_entry *new_entry)
    {
        const size_t offset = header_size + (entry_file_line_size * id);

        for (size_t i = 0; i < IO_RETRY_COUNT; i++)
        {
            std::fstream file(meta_fname, std::ios::in | std::ios::out | std::ios::binary);

            if (file.fail())
            {
                std::cerr << "Fopen failed (metadata write)" << std::endl;
                continue;
            }

            file.seekp(offset);
            file.write((char *)&(new_entry->offset), sizeof(uint64_t));
            file.write((char *)&(new_entry->size), sizeof(uint32_t));
            file.close();
            break;
        }
    }

    std::mutex chunk_cache_mutex;
    std::deque<std::tuple<size_t, uint16_t *>> chunk_cache;

    // Benchmark stats for load_chunk (accumulated per load_region call)
    mutable size_t bench_chunk_cache_hits = 0;
    mutable size_t bench_chunk_cache_misses = 0;
    mutable size_t bench_chunk_io_us = 0;
    mutable size_t bench_chunk_decomp_us = 0;
    mutable size_t bench_chunk_cache_us = 0;

    void reset_chunk_bench() const {
        bench_chunk_cache_hits = 0;
        bench_chunk_cache_misses = 0;
        bench_chunk_io_us = 0;
        bench_chunk_decomp_us = 0;
        bench_chunk_cache_us = 0;
    }

    uint16_t *load_chunk(size_t id, size_t sizex, size_t sizey, size_t sizez)
    {
        auto bench_start = std::chrono::steady_clock::now();

        const size_t out_buffer_size = sizex * sizey * sizez * sizeof(uint16_t);
        uint16_t *out = (uint16_t *)calloc(out_buffer_size, 1);
        metadata_entry *sel = load_meta_entry(id);

        if (sel->size == 0)
        {
            // Failed to read metadata (impossible for chunk size to be 0)
            free(sel);
            return out;
        }

        uint16_t *from_cache = 0;

        auto cache_start = std::chrono::steady_clock::now();
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
                memcpy((void *)out, (void *)from_cache, out_buffer_size);
                bench_chunk_cache_hits++;
            }

            global_chunk_cache_mutex.unlock();
        }
        bench_chunk_cache_us += std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now() - cache_start).count();

        // Either from_cache has the chunk, or was not in cache, or failed to get lock

        if (from_cache == 0)
        {
            bench_chunk_cache_misses++;

            // Read from file (use mmap if available, fallback to ifstream)
            auto io_start = std::chrono::steady_clock::now();
            size_t buffer_size = sel->size;
            uint16_t *read_buffer = (uint16_t *)malloc(buffer_size);

            bool read_failed = true;

            // Try mmap first (much faster - no syscall per read)
            // Lock briefly to prevent reading during refresh_mmap()
            {
                std::lock_guard<std::mutex> lock(mmap_refresh_mutex);
                if (mmap_data_ptr != nullptr && sel->offset + sel->size <= mmap_data_size)
                {
                    memcpy(read_buffer, (char *)mmap_data_ptr + sel->offset, sel->size);
                    read_failed = false;
                }
            }

            // Fallback to ifstream if mmap not available or failed
            if (read_failed)
            {
                for (size_t i = 0; i < IO_RETRY_COUNT; i++)
                {
                    std::ifstream file(data_fname, std::ios::in | std::ios::binary);

                    if (file.fail())
                    {
                        // std::cerr << "Fopen failed" << std::endl;
                        continue;
                    }

                    file.seekg(sel->offset);
                    file.read((char *)read_buffer, sel->size);
                    std::streamsize bytes_read = file.gcount();
                    file.close();
                    if (bytes_read != sel->size)
                    {
                        // std::cerr << "Read failed (short read)" << std::endl;
                        continue;
                    }
                    read_failed = false;
                    break;
                }
            }
            bench_chunk_io_us += std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::steady_clock::now() - io_start).count();

            // Check for read failure
            if (read_failed)
            {
                // std::cerr << "Read failed (max retries)" << std::endl;
                free(read_buffer);
                free(sel);
                return out;
            }

            // Decompress
            auto decomp_start = std::chrono::steady_clock::now();
            size_t decomp_size;
            char *read_decomp_buffer;
            pixtype *read_decomp_buffer_pt;

            uint32_t height, width, depth = 0;

            // 1 -> zstd
            // 2 -> 264
            // 3 -> AV1
            switch (compression_type)
            {
            case 1:
                // Decompress with ZSTD
                read_decomp_buffer = (char *)calloc(out_buffer_size, 1);
                decomp_size = ZSTD_decompress(read_decomp_buffer, out_buffer_size, read_buffer, sel->size);
                break;

            case 2:
            case 3:
                // Decompress with vidlib 2
                // read_decomp_buffer_pt = decode_stack_AV1(sizex, sizey, sizez, read_buffer, sel->size);
                auto decode_result = decode_stack_native(read_buffer, sel->size);

                read_decomp_buffer_pt = std::get<0>(decode_result);
                decomp_size = std::get<1>(decode_result);

                width = std::get<0>(std::get<2>(decode_result));
                height = std::get<1>(std::get<2>(decode_result));
                depth = std::get<2>(std::get<2>(decode_result));

                if (std::get<3>(decode_result) == sizeof(uint8_t))
                {
                    read_decomp_buffer = (char *)uint8_to_uint16_crop(read_decomp_buffer_pt, decomp_size, width, height, depth, sizex, sizey, sizez);
                    decomp_size = sizex * sizey * sizez * sizeof(uint16_t);
                    free(read_decomp_buffer_pt);
                }
                else if (std::get<3>(decode_result) == sizeof(uint16_t))
                {
                    read_decomp_buffer = (char *)uint16_to_uint16_crop((uint16_t *)read_decomp_buffer_pt, decomp_size, width, height, depth, sizex, sizey, sizez);
                    decomp_size = sizex * sizey * sizez * sizeof(uint16_t);
                    free(read_decomp_buffer_pt);
                }
                else
                {
                    std::cerr << "decode_stack_native returned unexpected pixel size" << std::endl;
                }

                break;
            }
            bench_chunk_decomp_us += std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::steady_clock::now() - decomp_start).count();

            free(read_buffer);

            // Copy result
            memcpy((void *)out, (void *)read_decomp_buffer, decomp_size);

            if (global_chunk_cache_mutex.try_lock_for(cache_lock_timeout))
            {
                global_chunk_line *cache_line = global_chunk_cache + global_chunk_cache_last;

                if (cache_line->ptr != 0)
                {
                    free(cache_line->ptr);
                }

                cache_line->chunk = (size_t)id;
                cache_line->mchunk = (size_t)this_mchunk_id;
                cache_line->ptr = (uint16_t *)read_decomp_buffer;

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

    void overwrite_chunk(size_t id, uint16_t *data, size_t data_size)
    {
        // compress data using ZSTD
        size_t compressed_size = ZSTD_compressBound(data_size);
        void *compressed_data = malloc(compressed_size);

        compressed_size = ZSTD_compress(compressed_data, compressed_size, data, data_size, 5);
        if (ZSTD_isError(compressed_size))
        {
            std::cerr << "ZSTD_compress failed" << std::endl;
            free(compressed_data);
            return;
        }

        {
            global_chunk_cache_mutex.lock();

            // Write to file
            std::fstream file(data_fname, std::ios::in | std::ios::out | std::ios::binary);
            if (file.fail())
            {
                std::cerr << "Fopen failed (write)" << std::endl;
                free(compressed_data);
                return;
            }

            file.seekp(0, std::ios::end);
            size_t new_offset = file.tellp();
            // file.seekp(sel->offset);
            file.write((char *)compressed_data, compressed_size);
            file.close();

            metadata_entry *new_entry = (metadata_entry *)malloc(sizeof(metadata_entry));
            new_entry->offset = new_offset;
            new_entry->size = compressed_size;

            replace_meta_entry(id, new_entry);
            free(new_entry);

            // Delete the prexisting values in the cache
            for (size_t i = 0; i < global_cache_size; i++)
            {
                if (global_chunk_cache[i].chunk == id)
                {
                    if (global_chunk_cache[i].mchunk == this_mchunk_id)
                    {
                        if (global_chunk_cache[i].ptr != 0)
                            free(global_chunk_cache[i].ptr);
                        global_chunk_cache[i].ptr = 0;
                    }
                }
            }

            global_chunk_cache_mutex.unlock();
        }

        free(compressed_data);

        // Refresh mmap to see the new data (file has grown)
        refresh_mmap();
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

        uint16_t *chunk = load_chunk(chunk_id, xsize, ysize, zsize);

        const uint16_t out = chunk[coffset];

        free(chunk);

        return out;
    }

    void print_info()
    {
        /*
        std::cout << "-----------------------------------" << std::endl;
        std::cout << "Files: " << meta_fname << ", " << data_fname << std::endl;
        std::cout << "dtype = " << dtype << std::endl;
        std::cout << "Chunks: " << chunkx << ", " << chunky << ", " << chunkz << std::endl;
        std::cout << "Size: " << sizex << ", " << sizey << ", " << sizez << std::endl;
        std::cout << "Tile Counts: " << countx << ", " << county << ", " << countz << std::endl;
        std::cout << "-----------------------------------" << std::endl;
        */

        std::cout << '[' << data_fname << "] " << " dtype=" << dtype << " chunks=(" << chunkx << ','
                  << chunky << ',' << chunkz << ") size=(" << sizex << ',' << sizey << ',' << sizez << ") "
                  << "tile_counts=(" << countx << ',' << county << ',' << countz << ")" << std::endl;
    }
};

class descriptor_layer
{
public:
    std::string source_name;
    uint16_t source_channel = 0;
    uint16_t target_channel = 0;

    size_t sizex = 0;
    size_t sizey = 0;
    size_t sizez = 0;

    // TODO, not currently used
    int64_t ioffsetx = 0;
    int64_t ioffsety = 0;
    int64_t ioffsetz = 0;

    int64_t ooffsetx = 0;
    int64_t ooffsety = 0;
    int64_t ooffsetz = 0;

    bool invertx = false;
    bool inverty = false;
    bool invertz = false;

    descriptor_layer()
    {
    }
};

class archive_reader
{
public:
    bool is_protected = false;
    bool is_valid = true;
    bool metadata_json = false;

    std::string fname; // "./example_dset"
    uint16_t channel_count;
    uint16_t archive_version; // 1 == current
    uint16_t dtype;
    uint16_t mchunkx, mchunky, mchunkz;
    uint64_t resx, resy, resz;
    uint64_t sizex, sizey, sizez;
    uint64_t mcountx, mcounty, mcountz;

    std::vector<size_t> scales;

    std::vector<descriptor_layer *> descriptor_layers;
    std::map<uint16_t, uint16_t> descriptor_channel_map;
    int64_t descriptor_x_origin;
    int64_t descriptor_y_origin;
    int64_t descriptor_z_origin;

    ArchiveType type;

    std::unordered_map<std::string, archive_reader *> *parent_archive_inventory;

    archive_reader(std::string name_in, enum ArchiveType type_in, std::unordered_map<std::string, archive_reader *> *par_in = nullptr)
    {
        fname = name_in;
        type = type_in;
        parent_archive_inventory = par_in;

        metadata_json = false;
        if (type == SISF_JSON)
        {
            // SISF_JSON is parsed the same other than metadata, set flag and type
            metadata_json = true;
            type = SISF;
        }

        switch (type)
        {
        case SISF:
            load_metadata_sisf();
            break;
        case ZARR:
            load_metadata_zarr();
            break;
        case DESCRIPTOR:
            load_metadata_descriptor();
            break;
        }

        load_protection();
    }

    ~archive_reader() {}

    // Return true if the contents of filters allows access to this dataset
    bool verify_protection(std::vector<std::pair<std::string, std::string>> filters)
    {
        if (!this->is_protected)
        {
            return true;
        }

        std::string token_in = "";
        for (auto i : filters)
        {
            if (i.first == "token")
            {
                token_in = i.second;
            }
        }

        if (token_in.size() == 0)
        {
            return false;
        }

        std::ifstream access_file(fname + "/.sisf_access");

        std::string line;
        while (std::getline(access_file, line))
        {
            if (line.size() == 0)
            {
                continue;
            }

            if (line == token_in)
                return true;
        }

        return false;
    }

    void load_protection()
    {
        std::vector<std::string> fnames = glob_tool(std::string(fname + "/.sisf_access"));

        if (fnames.size() > 0)
        {
            this->is_protected = true;
        }
        else
        {
            this->is_protected = false;
        }
    }

    void load_metadata_zarr()
    {
        tensorstore::Context context = tensorstore::Context::Default();
        auto store_future = tensorstore::Open({{"driver", "zarr3"},
                                               {"kvstore", {{"driver", "file"}, {"path", fname}}}},
                                              context, tensorstore::OpenMode::open, tensorstore::ReadWriteMode::read);

        auto store_result = store_future.result();

        if (!store_result.ok())
        {
            std::cerr << "Error opening TensorStore: " << store_result.status() << std::endl;
        }

        auto store = std::move(store_result.value());

        auto domain = store.domain();
        auto shape = domain.shape();
        auto labels = domain.labels();

        sizex = 0;
        sizey = 0;
        sizez = 0;
        channel_count = 0;

        size_t i = 0;
        for (const auto &dim : shape)
        {
            switch (i)
            {
            case 0: // x
                sizex = dim;
                break;
            case 1: // y
                sizey = dim;
                break;
            case 2: // z
                sizez = dim;
                break;
            case 3: // c
                channel_count = dim;
                break;
            }

            i++;
        }

        if (sizex == 0 || sizey == 0 || sizez == 0 || channel_count == 0)
        {
            std::cerr << "Invalid rank in Zarr dataset [" << fname << "]. Found i=" << i << std::endl;
        }

        archive_version = 0;
        dtype = 1;

        // TODO Load res from ts
        resx = 100;
        resy = 100;
        resz = 100;

        auto dim_units_result = store.dimension_units();

        if (!dim_units_result.ok())
        {
            std::cerr << "Error reading dimension_units from TensorStore: " << store_result.status() << std::endl;
        }
        else
        {
            auto dim_units = dim_units_result.value();

            for (size_t i = 0; i < dim_units.size(); i++)
            {
                if (dim_units[i].has_value())
                {
                    tensorstore::Unit u = dim_units[i].value();

                    // TODO Verify that unit is nm and scale properly

                    switch (i)
                    {
                    case 0: // x
                        resx = u.multiplier;
                        break;
                    case 1: // y
                        resy = u.multiplier;
                        break;
                    case 2: // z
                        resz = u.multiplier;
                        break;
                    case 3: // c
                        // Color does not have a unit
                        break;
                    }

                    // std::cout << "Name: " << labels[i] << '\t' << u.to_string() << '\t' << u.base_unit << '\t' << u.multiplier << std::endl;
                }
            }
        }

        mchunkx = sizex;
        mchunky = sizey;
        mchunkz = sizez;

        mcountx = 1;
        mcounty = 1;
        mcountz = 1;

        scales.push_back(1);
    }

    void load_metadata_sisf()
    {
        if (metadata_json)
        {
            std::ifstream inputFile(fname + "/metadata.json");
            if (!inputFile)
            {
                ; // TODO error handling
            }

            json jsonData;
            inputFile >> jsonData; // Read JSON data from file

            archive_version = 2;
            dtype = 1;
            channel_count = jsonData["channel_count"];

            mchunkx = jsonData["xchunk"];
            mchunky = jsonData["ychunk"];
            mchunkz = jsonData["zchunk"];

            sizex = jsonData["xsize"];
            sizey = jsonData["ysize"];
            sizez = jsonData["zsize"];

            resx = jsonData["xres"];
            resy = jsonData["yres"];
            resz = jsonData["zres"];

            // std::cout << "Read JSON from file: " << jsonData.dump(4) << std::endl;
        }
        else
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
        }

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

    void load_metadata_descriptor()
    {
        std::ifstream inputFile(fname + "/descriptor.json");
        if (!inputFile)
        {
            ; // TODO error handling
        }

        json jsonData;
        inputFile >> jsonData; // Read JSON data from file

        scales.push_back(1);

        mcountx = 0;
        mcounty = 0;
        mcountz = 0;

        mchunkx = 0;
        mchunky = 0;
        mchunkz = 0;

        archive_version = 2;
        dtype = 1;

        resx = jsonData["xres"];
        resy = jsonData["yres"];
        resz = jsonData["zres"];

        json layers = jsonData["layers"];

        channel_count = 0;
        if (layers.is_array())
        {
            for (const auto &element : layers)
            {
                descriptor_layer *layer = new descriptor_layer();

                layer->source_name = element["source"];
                layer->source_channel = element["source_channel"];
                layer->target_channel = element["target_channel"];

                json source_size = element["source_size"];
                if (source_size.is_array())
                {
                    size_t i = 0;

                    for (const auto &a : source_size)
                    {
                        int s = a.get<int>();

                        switch (i)
                        {
                        case 0:
                            layer->sizex = s;
                            break;
                        case 1:
                            layer->sizey = s;
                            break;
                        case 2:
                            layer->sizez = s;
                            break;
                        default:
                            break;
                        }

                        i++;
                    }

                    if (i != 3)
                    {
                        throw std::runtime_error("invalid source_size size");
                    }
                }
                else
                {
                    throw std::runtime_error("source_size should be an array");
                }

                json out_size = element["target_offset"];
                if (out_size.is_array())
                {
                    size_t i = 0;

                    for (const auto &a : out_size)
                    {
                        int s = a.get<int>();

                        switch (i)
                        {
                        case 0:
                            layer->ooffsetx = s;
                            break;
                        case 1:
                            layer->ooffsety = s;
                            break;
                        case 2:
                            layer->ooffsetz = s;
                            break;
                        default:
                            break;
                        }

                        i++;
                    }

                    if (i != 3)
                    {
                        throw std::runtime_error("invalid out_size size");
                    }
                }
                else
                {
                    throw std::runtime_error("out_size should be an array");
                }

                descriptor_layers.push_back(layer);
            }
        }
        else
        {
            throw std::runtime_error("layer is not an array");
        }

        int64_t minx = 0; // std::numeric_limits<int64_t>::max();
        int64_t maxx = std::numeric_limits<int64_t>::min();

        int64_t miny = 0; // std::numeric_limits<int64_t>::max();
        int64_t maxy = std::numeric_limits<int64_t>::min();

        int64_t minz = 0; // std::numeric_limits<int64_t>::max();
        int64_t maxz = std::numeric_limits<int64_t>::min();

        std::set<uint16_t> found_channels;
        for (descriptor_layer *l : descriptor_layers)
        {
            uint16_t tc = l->target_channel;
            found_channels.insert(tc);
            if (descriptor_channel_map.count(tc) == 0)
            {
                descriptor_channel_map[tc] = found_channels.size() - 1;
            }

            minx = std::min(minx, l->ooffsetx);
            miny = std::min(miny, l->ooffsety);
            minz = std::min(minz, l->ooffsetz);

            maxx = std::max(maxx, static_cast<int64_t>(l->sizex) + l->ooffsetx);
            maxy = std::max(maxy, static_cast<int64_t>(l->sizey) + l->ooffsety);
            maxz = std::max(maxz, static_cast<int64_t>(l->sizez) + l->ooffsetz);
        }

        sizex = maxx - minx;
        sizey = maxy - miny;
        sizez = maxz - minz;

        descriptor_x_origin = minx;
        descriptor_y_origin = miny;
        descriptor_z_origin = minz;

        channel_count = found_channels.size();
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

    bool contains_scale(size_t scale)
    {
        return std::find(scales.begin(), scales.end(), scale) != scales.end();
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

        if (out == 0 || out == nullptr)
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

            if (!out->is_valid)
            {
                delete out;
                out = nullptr;
            }

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

        // Benchmark timing variables
        size_t bench_get_mchunk_us = 0;
        size_t bench_load_chunk_us = 0;
        size_t bench_voxel_copy_us = 0;
        size_t bench_local_cache_us = 0;
        size_t bench_chunk_loads = 0;
        size_t bench_local_cache_hits = 0;
        size_t bench_voxel_count = 0;

        // Calculate size of output
        const size_t osizex = xe - xs;
        const size_t osizey = ye - ys;
        const size_t osizez = ze - zs;
        const size_t buffer_size = osizex * osizey * osizez * sizeof(uint16_t) * channel_count;

        // Allocate buffer for output
        uint16_t *out_buffer = (uint16_t *)calloc(buffer_size, 1);

        if (type == SISF)
        {
            // Define map for storing already decompressed chunks
            std::map<std::tuple<size_t, size_t, size_t, size_t, size_t>, uint16_t *> chunk_cache;
            std::set<std::tuple<size_t, size_t, size_t, size_t, size_t>> back_mchunks;

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
                    const size_t xmin = mcx * (i / mcx);
                    const size_t xmax = std::min((size_t)xmin + mcx, (size_t)sizex);
                    const size_t xsize = xmax - xmin;
                    const size_t chunk_id_x = i / ((size_t)mcx);
                    const size_t x_in_chunk = i - xmin;

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

                                bool is_bad = back_mchunks.count({scale, c, chunk_id_x, chunk_id_y, chunk_id_z}) > 0;

                                if (!is_bad)
                                {
                                    auto mchunk_start = std::chrono::steady_clock::now();
                                    chunk_reader = get_mchunk(scale, c, chunk_id_x, chunk_id_y, chunk_id_z);
                                    bench_get_mchunk_us += std::chrono::duration_cast<std::chrono::microseconds>(
                                        std::chrono::steady_clock::now() - mchunk_start).count();
                                }
                                else
                                {
                                    chunk_reader = nullptr;
                                }

                                if (chunk_reader == nullptr || chunk_reader == 0)
                                {
                                    if (!is_bad)
                                    {
                                        back_mchunks.insert({scale, c, chunk_id_x, chunk_id_y, chunk_id_z});
                                    }
                                    continue;
                                }

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

                                // Find the start/stop coordinates of this chunk
                                cxmin = ((size_t)chunk_reader->chunkx) * (x_in_chunk_offset / ((size_t)chunk_reader->chunkx));
                                cxmax = std::min((size_t)cxmin + chunk_reader->chunkx, (size_t)chunk_reader->sizex);
                                cxsize = cxmax - cxmin;

                                cymin = ((size_t)chunk_reader->chunky) * (y_in_chunk_offset / ((size_t)chunk_reader->chunky));
                                cymax = std::min((size_t)cymin + chunk_reader->chunky, (size_t)chunk_reader->sizey);
                                cysize = cymax - cymin;

                                czmin = ((size_t)chunk_reader->chunkz) * (z_in_chunk_offset / ((size_t)chunk_reader->chunkz));
                                czmax = std::min((size_t)czmin + chunk_reader->chunkz, (size_t)chunk_reader->sizez);
                                czsize = czmax - czmin;

                                // Check if the chunk is in the local cache
                                auto local_cache_start = std::chrono::steady_clock::now();
                                chunk = chunk_cache[*chunk_identifier];
                                bench_local_cache_us += std::chrono::duration_cast<std::chrono::microseconds>(
                                    std::chrono::steady_clock::now() - local_cache_start).count();

                                if (chunk == 0)
                                {
                                    auto load_start = std::chrono::steady_clock::now();
                                    chunk = chunk_reader->load_chunk(sub_chunk_id, cxsize, cysize, czsize);
                                    bench_load_chunk_us += std::chrono::duration_cast<std::chrono::microseconds>(
                                        std::chrono::steady_clock::now() - load_start).count();
                                    chunk_cache[*chunk_identifier] = chunk;
                                    bench_chunk_loads++;
                                }
                                else
                                {
                                    bench_local_cache_hits++;
                                }

                                // Store this ID as the most recent chunk
                                last_sub = sub_chunk_id;
                                last_c = c;
                            }

                            // Calculate the coordinates of the input and output inside their respective buffers
                            auto copy_start = std::chrono::steady_clock::now();
                            const size_t coffset = ((x_in_chunk_offset - cxmin) * cysize * czsize) +
                                                   ((y_in_chunk_offset - cymin) * czsize) +
                                                   (z_in_chunk_offset - czmin);

                            const size_t ooffset = (c * osizey * osizex * osizez) +
                                                   ((k - zs) * osizey * osizex) +
                                                   ((j - ys) * osizex) +
                                                   ((i - xs));

                            out_buffer[ooffset] = chunk[coffset];
                            bench_voxel_copy_us += std::chrono::duration_cast<std::chrono::microseconds>(
                                std::chrono::steady_clock::now() - copy_start).count();
                            bench_voxel_count++;
                        }
                    }
                }
            }

            if (chunk_identifier != nullptr)
            {
                delete chunk_identifier;
            }

            // Aggregate chunk-level stats from the last chunk_reader
            size_t total_cache_hits = 0, total_cache_misses = 0;
            size_t total_io_us = 0, total_decomp_us = 0, total_chunk_cache_us = 0;
            if (chunk_reader != nullptr) {
                total_cache_hits = chunk_reader->bench_chunk_cache_hits;
                total_cache_misses = chunk_reader->bench_chunk_cache_misses;
                total_io_us = chunk_reader->bench_chunk_io_us;
                total_decomp_us = chunk_reader->bench_chunk_decomp_us;
                total_chunk_cache_us = chunk_reader->bench_chunk_cache_us;
            }

            // Log benchmark results
            auto total_us = std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::steady_clock::now() - begin).count();

            CROW_LOG_DEBUG << "[BENCH load_region] "
                << "region=" << osizex << "x" << osizey << "x" << osizez << " "
                << "total=" << total_us << "us | "
                << "get_mchunk=" << bench_get_mchunk_us << "us | "
                << "load_chunk=" << bench_load_chunk_us << "us (n=" << bench_chunk_loads << ") | "
                << "local_cache=" << bench_local_cache_us << "us (hits=" << bench_local_cache_hits << ") | "
                << "voxel_copy=" << bench_voxel_copy_us << "us (n=" << bench_voxel_count << ") | "
                << "chunk_io=" << total_io_us << "us | "
                << "chunk_decomp=" << total_decomp_us << "us | "
                << "chunk_cache=" << total_chunk_cache_us << "us (hits=" << total_cache_hits << " misses=" << total_cache_misses << ")";

            for (auto it = chunk_cache.begin(); it != chunk_cache.end(); it++)
            {
                free(it->second);
            }
        }
        else if (type == ZARR)
        {
            tensorstore::Context context = tensorstore::Context::Default();
            auto store_future = tensorstore::Open({{"driver", "zarr3"},
                                                   {"kvstore", {{"driver", "file"}, {"path", fname}}}},
                                                  context, tensorstore::OpenMode::open, tensorstore::ReadWriteMode::read);

            auto store_result = store_future.result();

            if (!store_result.ok())
            {
                std::cerr << "Error opening TensorStore: " << store_result.status() << std::endl;
            }
            else
            {
                auto store = std::move(store_result.value());

                const size_t read_buffer_size = osizex * osizey * osizez * sizeof(uint16_t) * channel_count;
                uint16_t *read_buffer = (uint16_t *)malloc(read_buffer_size);

                auto array_result = tensorstore::Read<tensorstore::zero_origin>(
                                        store | tensorstore::AllDims().SizedInterval(
                                                    {(tensorstore::Index)xs, (tensorstore::Index)ys, (tensorstore::Index)zs, 0},
                                                    {(tensorstore::Index)osizex, (tensorstore::Index)osizey, (tensorstore::Index)osizez, (tensorstore::Index)channel_count}))
                                        .result();

                if (array_result.ok())
                {
                    // tensorstore::Array<tensorstore::Shared<void>, -1, tensorstore::ArrayOriginKind::offset, tensorstore::ContainerKind::container>
                    auto array = array_result.value();

                    // TODO detect datatype automatically
                    uint16_t *array_ptr = (uint16_t *)array.data();

                    // std::cout << "s:" << array.num_elements() << std::endl;
                    // Access example: std::cout << "T: " << array[{xs, ys, zs, 0}] << std::endl;

                    for (size_t c = 0; c < channel_count; c++)
                    {
                        for (size_t i = xs; i < xe; i++)
                        {
                            for (size_t j = ys; j < ye; j++)
                            {
                                for (size_t k = zs; k < ze; k++)
                                {
                                    // Calculate the coordinates of the input and output inside their respective buffers
                                    const size_t coffset = ((i - xs) * osizey * osizez * channel_count) + // X
                                                           ((j - ys) * osizez * channel_count) +          // Y
                                                           (k - zs) * channel_count + c;                  // Z and C

                                    const size_t ooffset = (c * osizey * osizex * osizez) + // C
                                                           ((k - zs) * osizey * osizex) +   // Z
                                                           ((j - ys) * osizex) +            // Y
                                                           ((i - xs));                      // X

                                    out_buffer[ooffset] = array_ptr[coffset];
                                }
                            }
                        }
                    }
                }
                else
                {
                    std::cerr << "Error reading from TensorStore: " << array_result.status() << std::endl;
                }
            }
        }
        else if (type == DESCRIPTOR && parent_archive_inventory != nullptr)
        {
            for (descriptor_layer *l : descriptor_layers)
            {
                // Find the beginning and end of this layer's output, measured relative to the origin (i.e. should never be less than zero)
                const int64_t layer_start_x = l->ooffsetx - descriptor_x_origin;
                const int64_t layer_end_x = layer_start_x + l->sizex;
                const int64_t layer_start_y = l->ooffsety - descriptor_y_origin;
                const int64_t layer_end_y = layer_start_y + l->sizey;
                const int64_t layer_start_z = l->ooffsetz - descriptor_y_origin;
                const int64_t layer_end_z = layer_start_z + l->sizez;

                // Calculate the overlap start-stops, in output space
                const int64_t x_overlap_start = std::max(layer_start_x, static_cast<int64_t>(xs));
                const int64_t x_overlap_end = std::min(layer_end_x, static_cast<int64_t>(xe));
                const int64_t y_overlap_start = std::max(layer_start_y, static_cast<int64_t>(ys));
                const int64_t y_overlap_end = std::min(layer_end_y, static_cast<int64_t>(ye));
                const int64_t z_overlap_start = std::max(layer_start_z, static_cast<int64_t>(zs));
                const int64_t z_overlap_end = std::min(layer_end_z, static_cast<int64_t>(ze));

                // True if there is an overlap between the requested region and the layer region
                const bool overlaps = (x_overlap_start <= x_overlap_end) &&
                                      (y_overlap_start <= y_overlap_end) &&
                                      (z_overlap_start <= z_overlap_end);

                if (!overlaps)
                {
                    // This layer is not included in the current access
                    continue;
                }

                auto reader = parent_archive_inventory->find(l->source_name);

                if (reader == parent_archive_inventory->end())
                {
                    // Source not in inventory
                    continue;
                }

                const size_t scale = 1; // TODO, not currently implemented

                // Calculate the overlap start-stop, in input space
                const int64_t x_overlap_start_shifted = x_overlap_start + l->ioffsetx;
                const int64_t x_overlap_end_shifted = x_overlap_end + l->ioffsetx;
                const int64_t y_overlap_start_shifted = y_overlap_start + l->ioffsety;
                const int64_t y_overlap_end_shifted = y_overlap_end + l->ioffsety;
                const int64_t z_overlap_start_shifted = z_overlap_start + l->ioffsetz;
                const int64_t z_overlap_end_shifted = z_overlap_end + l->ioffsetz;

                const int64_t region_x_size = x_overlap_end_shifted - x_overlap_start_shifted;
                const int64_t region_y_size = y_overlap_end_shifted - y_overlap_start_shifted;
                const int64_t region_z_size = z_overlap_end_shifted - z_overlap_start_shifted;

                uint16_t *region = reader->second->load_region(
                    scale,
                    x_overlap_start_shifted, x_overlap_end_shifted,
                    y_overlap_start_shifted, y_overlap_end_shifted,
                    z_overlap_start_shifted, z_overlap_end_shifted);

                const int64_t cin = l->source_channel;
                const int64_t cout = l->target_channel;

                for (size_t i = x_overlap_start; i < x_overlap_end; i++)
                {
                    for (size_t j = y_overlap_start; j < y_overlap_end; j++)
                    {
                        for (size_t k = z_overlap_start; k < z_overlap_end; k++)
                        {
                            const int64_t i_s = i + l->ioffsetx;
                            const int64_t j_s = j + l->ioffsety;
                            const int64_t k_s = k + l->ioffsetz;

                            // Calculate the coordinates of the input and output inside their respective buffers
                            const size_t roffset = (cin * region_x_size * region_y_size * region_z_size) +
                                                   ((k_s - z_overlap_start_shifted) * region_y_size * region_x_size) +
                                                   ((j_s - y_overlap_start_shifted) * region_x_size) +
                                                   ((i_s - x_overlap_start_shifted));

                            const size_t ooffset = (cout * osizey * osizex * osizez) + // C
                                                   ((k - zs) * osizey * osizex) +      // Z
                                                   ((j - ys) * osizex) +               // Y
                                                   ((i - xs));                         // X

                            out_buffer[ooffset] = region[roffset];
                        }
                    }
                }

                free(region);
            }
        }

        if (CHUNK_TIMER)
        {
            std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
            size_t dt = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
            std::cout << "Time difference = " << dt << " [us]" << std::endl;
        }

        return out_buffer;
    }

    void replace_region(
        size_t scale,
        size_t xs, size_t xe,
        size_t ys, size_t ye,
        size_t zs, size_t ze,
        const char *data)
    {
        // Calculate size of output
        const size_t osizex = xe - xs;
        const size_t osizey = ye - ys;
        const size_t osizez = ze - zs;
        const size_t buffer_size = osizex * osizey * osizez * sizeof(uint16_t) * channel_count;

        // Define map for storing already decompressed chunks
        std::map<std::tuple<size_t, size_t, size_t, size_t, size_t>, uint16_t *> chunk_cache;
        std::map<uint16_t *, std::tuple<size_t, size_t, size_t>> chunk_sizes;

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

                            // Check if the chunk is in the tmp cache
                            chunk = chunk_cache[*chunk_identifier];
                            if (chunk == 0)
                            {
                                chunk = chunk_reader->load_chunk(sub_chunk_id, cxsize, cysize, czsize);
                                chunk_cache[*chunk_identifier] = chunk;

                                chunk_sizes[chunk] = std::make_tuple(cxsize, cysize, czsize);
                            }

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

                        // out_buffer[ooffset] = chunk[coffset];
                        chunk[coffset] = ((uint16_t *)data)[ooffset];
                    }
                }
            }
        }

        if (chunk_identifier != nullptr)
        {
            delete chunk_identifier;
        }

        // TODO load chunks back
        for (auto it = chunk_cache.begin(); it != chunk_cache.end(); it++)
        {
            std::tuple<size_t, size_t, size_t, size_t, size_t> id_tuple = it->first;

            // std::tuple(c, chunk_id_x, chunk_id_y, chunk_id_z, sub_chunk_id);
            // chunk_reader = get_mchunk(scale, c, chunk_id_x, chunk_id_y, chunk_id_z);

            packed_reader *chunk_writer = get_mchunk(1, std::get<0>(id_tuple), std::get<1>(id_tuple), std::get<2>(id_tuple), std::get<3>(id_tuple));

            size_t chunk_size = std::get<0>(chunk_sizes[it->second]) * std::get<1>(chunk_sizes[it->second]) * std::get<2>(chunk_sizes[it->second]) * sizeof(uint16_t);

            chunk_writer->overwrite_chunk(std::get<4>(id_tuple), it->second, chunk_size);
            free(it->second);
        }
    }

    void print_info()
    {
        std::cout << '[' << fname << "] " << " dtype=" << dtype << " channels=" << channel_count << " chunks=(" << mchunkx << ','
                  << mchunky << ',' << mchunkz << ") size=(" << sizex << ',' << sizey << ',' << sizez << ") "
                  << "tile_counts=(" << mcountx << ',' << mcounty << ',' << mcountz << ") "
                  << "scales=[";

        for (size_t n : scales)
            std::cout << n << ',';

        std::cout << "]" << std::endl;

        if (type == DESCRIPTOR)
        {
            for (size_t i = 0; i < descriptor_layers.size(); i++)
            {
                std::cout << "\t[layer " << i << "] from=\"" << descriptor_layers[i]->source_name << "\":" << descriptor_layers[i]->source_channel
                          << " ch=" << descriptor_layers[i]->target_channel
                          << " size=(" << descriptor_layers[i]->sizex << ", " << descriptor_layers[i]->sizey << ", " << descriptor_layers[i]->sizez << ")"
                          << " ooffset=(" << descriptor_layers[i]->ooffsetx << ", " << descriptor_layers[i]->ooffsety << ", " << descriptor_layers[i]->ooffsetz << ")"
                          << std::endl;
            }
        }
    }
};

// Projection function enum
enum ProjectFunction { PROJECT_NONE = 0, PROJECT_MAX, PROJECT_MIN, PROJECT_AVG };

// Parallel load_region wrapper - loads region using chunk-aligned parallel tiles
// Falls back to sequential load_region for small regions
// Optional projection: if project_frames > 0, computes projection along project_axis
inline uint16_t* parallel_load_region(archive_reader* reader, int scale,
    size_t x_begin, size_t x_end, size_t y_begin, size_t y_end, size_t z_begin, size_t z_end,
    ProjectFunction project_mode = PROJECT_NONE, char project_axis = 'z', size_t project_frames = 0,
    size_t max_parallel_tiles = 4) {

    const size_t sx = x_end - x_begin;
    const size_t sy = y_end - y_begin;
    const size_t sz = z_end - z_begin;
    const size_t channel_count = reader->channel_count;

    // Get compression chunk sizes from packed_reader
    size_t tile_x = sx, tile_y = sy, tile_z = sz;
    packed_reader *pr = reader->get_mchunk(scale, 0, 0, 0, 0);
    if (pr != nullptr) {
        tile_x = pr->chunkx;
        tile_y = pr->chunky;
        tile_z = pr->chunkz;
    }

    // No projection case
    if (project_mode == PROJECT_NONE || project_frames < 1) {
        // If region fits in one chunk, use direct load
        if (sx <= tile_x && sy <= tile_y && sz <= tile_z) {
            return reader->load_region(scale, x_begin, x_end, y_begin, y_end, z_begin, z_end);
        }

        // Allocate output buffer
        const size_t out_size = sx * sy * sz * channel_count * sizeof(uint16_t);
        uint16_t* out_buffer = (uint16_t*)malloc(out_size);
        if (out_buffer == nullptr) return nullptr;

        // Build list of chunk-aligned tiles
        struct TileReq { size_t xs, xe, ys, ye, zs, ze; };
        std::vector<TileReq> tile_reqs;

        for (size_t tzs = z_begin; tzs < z_end; tzs += tile_z) {
            size_t tze = std::min(tzs + tile_z, z_end);
            for (size_t tys = y_begin; tys < y_end; tys += tile_y) {
                size_t tye = std::min(tys + tile_y, y_end);
                for (size_t txs = x_begin; txs < x_end; txs += tile_x) {
                    size_t txe = std::min(txs + tile_x, x_end);
                    tile_reqs.push_back({txs, txe, tys, tye, tzs, tze});
                }
            }
        }

        for (size_t batch_start = 0; batch_start < tile_reqs.size(); batch_start += max_parallel_tiles) {
            size_t batch_end = std::min(batch_start + max_parallel_tiles, tile_reqs.size());

            std::vector<std::future<std::pair<size_t, uint16_t*>>> futures;
            futures.reserve(batch_end - batch_start);

            for (size_t i = batch_start; i < batch_end; i++) {
                TileReq req = tile_reqs[i];
                futures.push_back(std::async(std::launch::async, [reader, scale, req, i]() {
                    uint16_t* tile_buf = reader->load_region(scale, req.xs, req.xe, req.ys, req.ye, req.zs, req.ze);
                    return std::make_pair(i, tile_buf);
                }));
            }

            std::vector<std::pair<size_t, uint16_t*>> results;
            results.reserve(futures.size());
            for (auto& f : futures) results.push_back(f.get());

            for (auto& [idx, tile_buf] : results) {
                if (tile_buf == nullptr) continue;
                auto& req = tile_reqs[idx];
                const size_t tsx = req.xe - req.xs;
                const size_t tsy = req.ye - req.ys;
                const size_t tsz = req.ze - req.zs;

                for (size_t c = 0; c < channel_count; c++) {
                    for (size_t zi = 0; zi < tsz; zi++) {
                        for (size_t yi = 0; yi < tsy; yi++) {
                            size_t tile_offset = (c * tsx * tsy * tsz) + (zi * tsx * tsy) + (yi * tsx);
                            size_t out_x = req.xs - x_begin;
                            size_t out_y = req.ys - y_begin + yi;
                            size_t out_z = req.zs - z_begin + zi;
                            size_t out_offset = (c * sx * sy * sz) + (out_z * sx * sy) + (out_y * sx) + out_x;
                            memcpy(&out_buffer[out_offset], &tile_buf[tile_offset], tsx * sizeof(uint16_t));
                        }
                    }
                }
                free(tile_buf);
            }
        }
        return out_buffer;
    }

    // === Projection case ===
    std::tuple<size_t, size_t, size_t> dataset_size = reader->get_size(scale);
    const size_t sizex = std::get<0>(dataset_size);
    const size_t sizey = std::get<1>(dataset_size);
    const size_t sizez = std::get<2>(dataset_size);

    // Allocate output buffer
    const size_t out_voxels = sx * sy * sz * channel_count;
    const size_t out_size = out_voxels * sizeof(uint16_t);
    uint16_t* out_buffer = (uint16_t*)malloc(out_size);
    if (out_buffer == nullptr) return nullptr;

    // Initialize output based on projection mode
    if (project_mode == PROJECT_MAX) {
        for (size_t i = 0; i < out_voxels; i++) out_buffer[i] = std::numeric_limits<uint16_t>::min();
    } else if (project_mode == PROJECT_MIN) {
        for (size_t i = 0; i < out_voxels; i++) out_buffer[i] = std::numeric_limits<uint16_t>::max();
    }

    // For avg, need sum and count buffers
    std::vector<double> sum_buffer;
    std::vector<uint16_t> count_buffer;
    if (project_mode == PROJECT_AVG) {
        sum_buffer.resize(out_voxels, 0.0);
        count_buffer.resize(out_voxels, 0);
    }

    // Extended region for projection input
    size_t x_end_ext = x_end, y_end_ext = y_end, z_end_ext = z_end;
    switch (project_axis) {
    case 'x': x_end_ext = std::min(x_end + project_frames, sizex); break;
    case 'y': y_end_ext = std::min(y_end + project_frames, sizey); break;
    case 'z': z_end_ext = std::min(z_end + project_frames, sizez); break;
    }

    // Build list of input chunks (aligned to compression boundaries)
    struct ChunkReq { size_t xs, xe, ys, ye, zs, ze; };
    std::vector<ChunkReq> chunk_reqs;

    for (size_t czs = z_begin; czs < z_end_ext; czs += tile_z) {
        size_t cze = std::min(czs + tile_z, z_end_ext);
        for (size_t cys = y_begin; cys < y_end_ext; cys += tile_y) {
            size_t cye = std::min(cys + tile_y, y_end_ext);
            for (size_t cxs = x_begin; cxs < x_end_ext; cxs += tile_x) {
                size_t cxe = std::min(cxs + tile_x, x_end_ext);
                chunk_reqs.push_back({cxs, cxe, cys, cye, czs, cze});
            }
        }
    }

    std::mutex out_mutex;

    for (size_t batch_start = 0; batch_start < chunk_reqs.size(); batch_start += max_parallel_tiles) {
        size_t batch_end = std::min(batch_start + max_parallel_tiles, chunk_reqs.size());

        std::vector<std::future<void>> futures;
        futures.reserve(batch_end - batch_start);

        for (size_t ci = batch_start; ci < batch_end; ci++) {
            ChunkReq req = chunk_reqs[ci];
            futures.push_back(std::async(std::launch::async,
                [&, req]() {
                uint16_t* chunk_buf = reader->load_region(scale, req.xs, req.xe, req.ys, req.ye, req.zs, req.ze);
                if (chunk_buf == nullptr) return;

                const size_t csx = req.xe - req.xs;
                const size_t csy = req.ye - req.ys;
                const size_t csz = req.ze - req.zs;

                for (size_t c = 0; c < channel_count; c++) {
                    for (size_t cz = 0; cz < csz; cz++) {
                        size_t abs_z = req.zs + cz;
                        for (size_t cy = 0; cy < csy; cy++) {
                            size_t abs_y = req.ys + cy;
                            for (size_t cx = 0; cx < csx; cx++) {
                                size_t abs_x = req.xs + cx;

                                const size_t ioffset = (c * csx * csy * csz) + (cz * csx * csy) + (cy * csx) + cx;
                                const uint16_t vin = chunk_buf[ioffset];

                                // Determine which output voxels this input contributes to
                                size_t out_x_min, out_x_max, out_y_min, out_y_max, out_z_min, out_z_max;

                                switch (project_axis) {
                                case 'x':
                                    out_x_min = (abs_x >= project_frames - 1 + x_begin) ? abs_x - project_frames + 1 : x_begin;
                                    out_x_max = std::min(abs_x + 1, x_end);
                                    out_y_min = abs_y; out_y_max = abs_y + 1;
                                    out_z_min = abs_z; out_z_max = abs_z + 1;
                                    break;
                                case 'y':
                                    out_x_min = abs_x; out_x_max = abs_x + 1;
                                    out_y_min = (abs_y >= project_frames - 1 + y_begin) ? abs_y - project_frames + 1 : y_begin;
                                    out_y_max = std::min(abs_y + 1, y_end);
                                    out_z_min = abs_z; out_z_max = abs_z + 1;
                                    break;
                                case 'z':
                                default:
                                    out_x_min = abs_x; out_x_max = abs_x + 1;
                                    out_y_min = abs_y; out_y_max = abs_y + 1;
                                    out_z_min = (abs_z >= project_frames - 1 + z_begin) ? abs_z - project_frames + 1 : z_begin;
                                    out_z_max = std::min(abs_z + 1, z_end);
                                    break;
                                }

                                if (out_x_min >= x_end || out_y_min >= y_end || out_z_min >= z_end) continue;
                                if (out_x_max <= x_begin || out_y_max <= y_begin || out_z_max <= z_begin) continue;
                                out_x_min = std::max(out_x_min, x_begin);
                                out_y_min = std::max(out_y_min, y_begin);
                                out_z_min = std::max(out_z_min, z_begin);

                                for (size_t oz = out_z_min; oz < out_z_max; oz++) {
                                    for (size_t oy = out_y_min; oy < out_y_max; oy++) {
                                        for (size_t ox = out_x_min; ox < out_x_max; ox++) {
                                            size_t ooffset = (c * sx * sy * sz) +
                                                ((oz - z_begin) * sx * sy) + ((oy - y_begin) * sx) + (ox - x_begin);

                                            std::lock_guard<std::mutex> lock(out_mutex);
                                            switch (project_mode) {
                                            case PROJECT_MAX:
                                                out_buffer[ooffset] = std::max(out_buffer[ooffset], vin);
                                                break;
                                            case PROJECT_MIN:
                                                out_buffer[ooffset] = std::min(out_buffer[ooffset], vin);
                                                break;
                                            case PROJECT_AVG:
                                                sum_buffer[ooffset] += vin;
                                                count_buffer[ooffset]++;
                                                break;
                                            default:
                                                break;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                free(chunk_buf);
            }));
        }

        for (auto& f : futures) f.get();
    }

    // Finalize avg projection
    if (project_mode == PROJECT_AVG) {
        for (size_t i = 0; i < out_voxels; i++) {
            if (count_buffer[i] > 0) {
                out_buffer[i] = (uint16_t)(sum_buffer[i] / count_buffer[i]);
            }
        }
    }

    return out_buffer;
}
