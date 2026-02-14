#include <glob.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <fcntl.h>

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
#include <shared_mutex>
#include <atomic>
#include <map>
#include <chrono>

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
#define DEBUG_STREAM 1  // Enable detailed timing for stream requests
#define IO_RETRY_COUNT 5

std::atomic<size_t> mchunk_uuid{0};

// Debug counters for cache performance
std::atomic<size_t> mchunk_cache_hits{0};
std::atomic<size_t> mchunk_cache_misses{0};

enum ArchiveType
{
    SISF_JSON,
    SISF,
    ZARR,
    DESCRIPTOR
};

using json = nlohmann::json;

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

    // File descriptors for concurrent pread() access (no locking needed)
    int meta_fd = -1;
    int data_fd = -1;

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

        // Open file descriptors for concurrent pread() access
        meta_fd = open(meta_fname.c_str(), O_RDONLY);
        data_fd = open(data_fname.c_str(), O_RDONLY);

        if (meta_fd < 0 || data_fd < 0)
        {
            std::cerr << "Failed to open file descriptors" << std::endl;
            if (meta_fd >= 0) close(meta_fd);
            if (data_fd >= 0) close(data_fd);
            meta_fd = -1;
            data_fd = -1;
            return;
        }

        is_valid = true;
    }

    ~packed_reader()
    {
        if (meta_fd >= 0)
            close(meta_fd);
        if (data_fd >= 0)
            close(data_fd);
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

        if (meta_fd < 0)
            return out;

        const off_t offset = header_size + (entry_file_line_size * id);

        // pread is atomic and thread-safe - no locking needed
        for (size_t i = 0; i < IO_RETRY_COUNT; i++)
        {
            // Read offset and size in one pread call
            char buf[sizeof(uint64_t) + sizeof(uint32_t)];
            ssize_t bytes_read = pread(meta_fd, buf, sizeof(buf), offset);

            if (bytes_read != sizeof(buf))
            {
                std::cerr << "Metadata read failed (short read)" << std::endl;
                continue;
            }

            memcpy(&(out->offset), buf, sizeof(uint64_t));
            memcpy(&(out->size), buf + sizeof(uint64_t), sizeof(uint32_t));
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

    // Simplified load_chunk - no global cache (load_region has local cache, OS has page cache)
    uint16_t *load_chunk(size_t id, size_t sizex, size_t sizey, size_t sizez)
    {
        const size_t out_buffer_size = sizex * sizey * sizez * sizeof(uint16_t);
        metadata_entry *sel = load_meta_entry(id);

        if (sel->size == 0)
        {
            free(sel);
            return (uint16_t *)calloc(out_buffer_size, 1);
        }

        uint16_t *read_buffer = (uint16_t *)malloc(sel->size);

        bool read_failed = true;
        for (size_t i = 0; i < IO_RETRY_COUNT; i++)
        {
            if (data_fd < 0)
                continue;

            ssize_t bytes_read = pread(data_fd, (char *)read_buffer, sel->size, sel->offset);

            if (bytes_read != (ssize_t)sel->size)
                continue;

            read_failed = false;
            break;
        }

        if (read_failed)
        {
            free(read_buffer);
            free(sel);
            return (uint16_t *)calloc(out_buffer_size, 1);
        }

        uint16_t *out;

        switch (compression_type)
        {
        case 1:
            // ZSTD: decompress directly to output buffer
            out = (uint16_t *)calloc(out_buffer_size, 1);
            ZSTD_decompress(out, out_buffer_size, read_buffer, sel->size);
            free(read_buffer);
            free(sel);
            return out;

        case 2:
        case 3:
        {
            auto decode_result = decode_stack_native(read_buffer, sel->size);
            pixtype *read_decomp_buffer_pt = std::get<0>(decode_result);
            size_t decomp_size = std::get<1>(decode_result);
            uint32_t width = std::get<0>(std::get<2>(decode_result));
            uint32_t height = std::get<1>(std::get<2>(decode_result));
            uint32_t depth = std::get<2>(std::get<2>(decode_result));

            free(read_buffer);
            free(sel);

            if (std::get<3>(decode_result) == sizeof(uint8_t))
            {
                out = (uint16_t *)uint8_to_uint16_crop(read_decomp_buffer_pt, decomp_size, width, height, depth, sizex, sizey, sizez);
                free(read_decomp_buffer_pt);
                return out;
            }
            else if (std::get<3>(decode_result) == sizeof(uint16_t))
            {
                out = (uint16_t *)uint16_to_uint16_crop((uint16_t *)read_decomp_buffer_pt, decomp_size, width, height, depth, sizex, sizey, sizez);
                free(read_decomp_buffer_pt);
                return out;
            }
            else
            {
                std::cerr << "decode_stack_native returned unexpected pixel size" << std::endl;
                free(read_decomp_buffer_pt);
                return (uint16_t *)calloc(out_buffer_size, 1);
            }
        }

        default:
            free(read_buffer);
            free(sel);
            return (uint16_t *)calloc(out_buffer_size, 1);
        }

        free(sel);
        return out;
    }

    // Direct chunk load without global cache - for streaming
    uint16_t *load_chunk_direct(size_t id, size_t sizex, size_t sizey, size_t sizez)
    {
        const size_t out_buffer_size = sizex * sizey * sizez * sizeof(uint16_t);
        metadata_entry *sel = load_meta_entry(id);

        if (sel->size == 0)
        {
            free(sel);
            return (uint16_t *)calloc(out_buffer_size, 1);
        }

        uint16_t *read_buffer = (uint16_t *)malloc(sel->size);

        bool read_failed = true;
        // pread is atomic and thread-safe - no locking needed
        for (size_t i = 0; i < IO_RETRY_COUNT; i++)
        {
            if (data_fd < 0)
                continue;

            ssize_t bytes_read = pread(data_fd, (char *)read_buffer, sel->size, sel->offset);

            if (bytes_read != (ssize_t)sel->size)
                continue;

            read_failed = false;
            break;
        }

        if (read_failed)
        {
            free(read_buffer);
            free(sel);
            return (uint16_t *)calloc(out_buffer_size, 1);
        }

        uint16_t *out;

        switch (compression_type)
        {
        case 1:
            // ZSTD: decompress directly to output buffer (avoid extra copy)
            out = (uint16_t *)calloc(out_buffer_size, 1);
            ZSTD_decompress(out, out_buffer_size, read_buffer, sel->size);
            free(read_buffer);
            free(sel);
            return out;

        case 2:
        case 3:
        {
            auto decode_result = decode_stack_native(read_buffer, sel->size);
            pixtype *read_decomp_buffer_pt = std::get<0>(decode_result);
            size_t decomp_size = std::get<1>(decode_result);
            uint32_t width = std::get<0>(std::get<2>(decode_result));
            uint32_t height = std::get<1>(std::get<2>(decode_result));
            uint32_t depth = std::get<2>(std::get<2>(decode_result));

            free(read_buffer);
            free(sel);

            if (std::get<3>(decode_result) == sizeof(uint8_t))
            {
                // uint8_to_uint16_crop allocates and returns new buffer
                out = (uint16_t *)uint8_to_uint16_crop(read_decomp_buffer_pt, decomp_size, width, height, depth, sizex, sizey, sizez);
                free(read_decomp_buffer_pt);
                return out;
            }
            else if (std::get<3>(decode_result) == sizeof(uint16_t))
            {
                // uint16_to_uint16_crop allocates and returns new buffer
                out = (uint16_t *)uint16_to_uint16_crop((uint16_t *)read_decomp_buffer_pt, decomp_size, width, height, depth, sizex, sizey, sizez);
                free(read_decomp_buffer_pt);
                return out;
            }
            else
            {
                std::cerr << "decode_stack_native returned unexpected pixel size" << std::endl;
                free(read_decomp_buffer_pt);
                return (uint16_t *)calloc(out_buffer_size, 1);
            }
        }

        default:
            free(read_buffer);
            free(sel);
            return (uint16_t *)calloc(out_buffer_size, 1);
        }
    }

    void overwrite_chunk(size_t id, uint16_t *data, size_t data_size)
    {
        // Compress data using ZSTD
        size_t compressed_size = ZSTD_compressBound(data_size);
        void *compressed_data = malloc(compressed_size);

        compressed_size = ZSTD_compress(compressed_data, compressed_size, data, data_size, 5);
        if (ZSTD_isError(compressed_size))
        {
            std::cerr << "ZSTD_compress failed" << std::endl;
            free(compressed_data);
            return;
        }

        // Write to file (append to end)
        std::fstream file(data_fname, std::ios::in | std::ios::out | std::ios::binary);
        if (file.fail())
        {
            std::cerr << "Fopen failed (write)" << std::endl;
            free(compressed_data);
            return;
        }

        file.seekp(0, std::ios::end);
        size_t new_offset = file.tellp();
        file.write((char *)compressed_data, compressed_size);
        file.close();

        // Update metadata entry with new offset/size
        metadata_entry *new_entry = (metadata_entry *)malloc(sizeof(metadata_entry));
        new_entry->offset = new_offset;
        new_entry->size = compressed_size;

        replace_meta_entry(id, new_entry);
        free(new_entry);

        free(compressed_data);
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

    ~archive_reader()
    {
        clear_mchunk_buffer();
    }

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

    static const size_t MAX_MCHUNK_BUFFER_SIZE = 256;
    std::map<std::tuple<size_t, size_t, size_t, size_t, size_t>, packed_reader *> mchunk_buffer;
    std::deque<std::tuple<size_t, size_t, size_t, size_t, size_t>> mchunk_lru;
    std::shared_mutex mchunk_buffer_mutex;  // Reader-writer lock for concurrent reads

    void clear_mchunk_buffer()
    {
        std::unique_lock<std::shared_mutex> lock(mchunk_buffer_mutex);
        for (auto &pair : mchunk_buffer)
        {
            if (pair.second != nullptr)
            {
                delete pair.second;
            }
        }
        mchunk_buffer.clear();
        mchunk_lru.clear();
    }

    packed_reader *get_mchunk(size_t scale, size_t channel, size_t i, size_t j, size_t k)
    {
        std::tuple<size_t, size_t, size_t, size_t, size_t> id_tuple = std::make_tuple(scale, channel, i, j, k);

        // Fast path: read-only lookup (multiple threads can read concurrently)
        {
            std::shared_lock<std::shared_mutex> read_lock(mchunk_buffer_mutex);
            auto it = mchunk_buffer.find(id_tuple);
            if (it != mchunk_buffer.end() && it->second != nullptr)
            {
                if (DEBUG_STREAM) mchunk_cache_hits.fetch_add(1);
                return it->second;
            }
        }
        if (DEBUG_STREAM) mchunk_cache_misses.fetch_add(1);

        // Slow path: need to create new entry (exclusive lock)
        // Build the reader OUTSIDE the lock to minimize lock time
        std::stringstream ss;
        ss << "chunk_" << i << '_' << j << '_' << k << '.' << channel << '.' << scale << 'X';
        const std::string chunk_root = ss.str();
        const std::string chunk_meta_name = fname + "/meta/" + chunk_root + ".meta";
        const std::string chunk_data_name = fname + "/data/" + chunk_root + ".data";

        size_t random_id = mchunk_uuid.fetch_add(1);  // Atomic increment, no lock needed

        packed_reader *new_reader = new packed_reader(random_id, chunk_meta_name, chunk_data_name);

        if (!new_reader->is_valid)
        {
            delete new_reader;
            new_reader = nullptr;
        }

        // Now acquire exclusive lock to insert
        std::unique_lock<std::shared_mutex> write_lock(mchunk_buffer_mutex);

        // Double-check: another thread may have inserted while we were creating
        auto it = mchunk_buffer.find(id_tuple);
        if (it != mchunk_buffer.end() && it->second != nullptr)
        {
            // Another thread beat us - discard our reader and use theirs
            if (new_reader != nullptr)
                delete new_reader;
            return it->second;
        }

        // Evict oldest entries if buffer is full
        while (mchunk_buffer.size() >= MAX_MCHUNK_BUFFER_SIZE && !mchunk_lru.empty())
        {
            auto oldest = mchunk_lru.front();
            mchunk_lru.pop_front();
            auto evict_it = mchunk_buffer.find(oldest);
            if (evict_it != mchunk_buffer.end())
            {
                if (evict_it->second != nullptr)
                {
                    delete evict_it->second;
                }
                mchunk_buffer.erase(evict_it);
            }
        }

        mchunk_buffer[id_tuple] = new_reader;
        mchunk_lru.push_back(id_tuple);

        return new_reader;
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
        uint16_t *out_buffer = (uint16_t *)calloc(buffer_size, 1);

        if (type == SISF)
        {
            // Scaled metachunk size
            const size_t mcx = mchunkx / scale;
            const size_t mcy = mchunky / scale;
            const size_t mcz = mchunkz / scale;

            // Calculate mchunk index ranges that overlap the request
            const size_t mx_start = xs / mcx;
            const size_t mx_end = (xe - 1) / mcx;
            const size_t my_start = ys / mcy;
            const size_t my_end = (ye - 1) / mcy;
            const size_t mz_start = zs / mcz;
            const size_t mz_end = (ze - 1) / mcz;

            // Iterate by channel, then by mchunk
            for (size_t c = 0; c < channel_count; c++)
            {
                const size_t c_offset = c * osizex * osizey * osizez;

                for (size_t mx = mx_start; mx <= mx_end; mx++)
                {
                    const size_t mchunk_xs = mx * mcx;
                    const size_t mchunk_xe = std::min(mchunk_xs + mcx, sizex);
                    const size_t overlap_xs = std::max(xs, mchunk_xs);
                    const size_t overlap_xe = std::min(xe, mchunk_xe);

                    for (size_t my = my_start; my <= my_end; my++)
                    {
                        const size_t mchunk_ys = my * mcy;
                        const size_t mchunk_ye = std::min(mchunk_ys + mcy, sizey);
                        const size_t overlap_ys = std::max(ys, mchunk_ys);
                        const size_t overlap_ye = std::min(ye, mchunk_ye);

                        for (size_t mz = mz_start; mz <= mz_end; mz++)
                        {
                            const size_t mchunk_zs = mz * mcz;
                            const size_t mchunk_ze = std::min(mchunk_zs + mcz, sizez);
                            const size_t overlap_zs = std::max(zs, mchunk_zs);
                            const size_t overlap_ze = std::min(ze, mchunk_ze);

                            packed_reader *chunk_reader = get_mchunk(scale, c, mx, my, mz);
                            if (chunk_reader == nullptr)
                                continue;

                            const size_t scx = chunk_reader->chunkx;
                            const size_t scy = chunk_reader->chunky;
                            const size_t scz = chunk_reader->chunkz;

                            // Convert to mchunk-local coordinates with crop offset
                            const size_t local_xs = (overlap_xs - mchunk_xs) + chunk_reader->cropstartx;
                            const size_t local_xe = (overlap_xe - mchunk_xs) + chunk_reader->cropstartx;
                            const size_t local_ys = (overlap_ys - mchunk_ys) + chunk_reader->cropstarty;
                            const size_t local_ye = (overlap_ye - mchunk_ys) + chunk_reader->cropstarty;
                            const size_t local_zs = (overlap_zs - mchunk_zs) + chunk_reader->cropstartz;
                            const size_t local_ze = (overlap_ze - mchunk_zs) + chunk_reader->cropstartz;

                            // Sub-chunk index ranges
                            const size_t sx_start = local_xs / scx;
                            const size_t sx_end = (local_xe - 1) / scx;
                            const size_t sy_start = local_ys / scy;
                            const size_t sy_end = (local_ye - 1) / scy;
                            const size_t sz_start = local_zs / scz;
                            const size_t sz_end = (local_ze - 1) / scz;

                            for (size_t sx = sx_start; sx <= sx_end; sx++)
                            {
                                const size_t schunk_xs = sx * scx;
                                const size_t schunk_xe = std::min(schunk_xs + scx, chunk_reader->sizex);
                                const size_t schunk_xsize = schunk_xe - schunk_xs;
                                const size_t sc_overlap_xs = std::max(local_xs, schunk_xs);
                                const size_t sc_overlap_xe = std::min(local_xe, schunk_xe);

                                for (size_t sy = sy_start; sy <= sy_end; sy++)
                                {
                                    const size_t schunk_ys = sy * scy;
                                    const size_t schunk_ye = std::min(schunk_ys + scy, chunk_reader->sizey);
                                    const size_t schunk_ysize = schunk_ye - schunk_ys;
                                    const size_t sc_overlap_ys = std::max(local_ys, schunk_ys);
                                    const size_t sc_overlap_ye = std::min(local_ye, schunk_ye);

                                    for (size_t sz = sz_start; sz <= sz_end; sz++)
                                    {
                                        const size_t schunk_zs = sz * scz;
                                        const size_t schunk_ze = std::min(schunk_zs + scz, chunk_reader->sizez);
                                        const size_t schunk_zsize = schunk_ze - schunk_zs;
                                        const size_t sc_overlap_zs = std::max(local_zs, schunk_zs);
                                        const size_t sc_overlap_ze = std::min(local_ze, schunk_ze);

                                        const size_t sub_chunk_id = (sx * chunk_reader->countz * chunk_reader->county) +
                                                                    (sy * chunk_reader->countz) + sz;
                                        uint16_t *chunk = chunk_reader->load_chunk(sub_chunk_id, schunk_xsize, schunk_ysize, schunk_zsize);

                                        // Copy all overlapping pixels
                                        for (size_t lx = sc_overlap_xs; lx < sc_overlap_xe; lx++)
                                        {
                                            const size_t gx = lx - chunk_reader->cropstartx + mchunk_xs;
                                            const size_t cx = lx - schunk_xs;

                                            for (size_t ly = sc_overlap_ys; ly < sc_overlap_ye; ly++)
                                            {
                                                const size_t gy = ly - chunk_reader->cropstarty + mchunk_ys;
                                                const size_t cy = ly - schunk_ys;
                                                const size_t chunk_xy_offset = (cx * schunk_ysize * schunk_zsize) + (cy * schunk_zsize);
                                                const size_t out_xy_base = c_offset + ((gy - ys) * osizex) + (gx - xs);

                                                for (size_t lz = sc_overlap_zs; lz < sc_overlap_ze; lz++)
                                                {
                                                    const size_t gz = lz - chunk_reader->cropstartz + mchunk_zs;
                                                    const size_t cz = lz - schunk_zs;
                                                    const size_t coffset = chunk_xy_offset + cz;
                                                    const size_t ooffset = out_xy_base + ((gz - zs) * osizey * osizex);

                                                    out_buffer[ooffset] = chunk[coffset];
                                                }
                                            }
                                        }

                                        free(chunk);
                                    }
                                }
                            }
                        }
                    }
                }
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

    // Direct region load without global cache - for streaming endpoint
    // Optimized: chunk-first iteration eliminates per-pixel overhead
    uint16_t *load_region_direct(
        size_t scale,
        size_t xs, size_t xe,
        size_t ys, size_t ye,
        size_t zs, size_t ze)
    {
        std::chrono::steady_clock::time_point t_start, t_end;
        size_t time_get_mchunk = 0, time_load_chunk = 0, time_copy = 0;
        size_t chunks_loaded = 0, mchunks_accessed = 0;

        if (DEBUG_STREAM) t_start = std::chrono::steady_clock::now();

        const size_t osizex = xe - xs;
        const size_t osizey = ye - ys;
        const size_t osizez = ze - zs;
        const size_t buffer_size = osizex * osizey * osizez * sizeof(uint16_t) * channel_count;

        uint16_t *out_buffer = (uint16_t *)calloc(buffer_size, 1);

        if (type != SISF)
        {
            // For non-SISF types, fall back to regular load_region
            uint16_t *tmp = load_region(scale, xs, xe, ys, ye, zs, ze);
            memcpy(out_buffer, tmp, buffer_size);
            free(tmp);
            return out_buffer;
        }

        // Scaled metachunk size
        const size_t mcx = mchunkx / scale;
        const size_t mcy = mchunky / scale;
        const size_t mcz = mchunkz / scale;

        // Calculate mchunk index ranges that overlap the request
        const size_t mx_start = xs / mcx;
        const size_t mx_end = (xe - 1) / mcx;
        const size_t my_start = ys / mcy;
        const size_t my_end = (ye - 1) / mcy;
        const size_t mz_start = zs / mcz;
        const size_t mz_end = (ze - 1) / mcz;

        // Iterate by channel, then by mchunk
        for (size_t c = 0; c < channel_count; c++)
        {
            const size_t c_offset = c * osizex * osizey * osizez;

            for (size_t mx = mx_start; mx <= mx_end; mx++)
            {
                // Mchunk X bounds in global coordinates
                const size_t mchunk_xs = mx * mcx;
                const size_t mchunk_xe = std::min(mchunk_xs + mcx, sizex);

                // Overlap with request in global coordinates
                const size_t overlap_xs = std::max(xs, mchunk_xs);
                const size_t overlap_xe = std::min(xe, mchunk_xe);

                for (size_t my = my_start; my <= my_end; my++)
                {
                    const size_t mchunk_ys = my * mcy;
                    const size_t mchunk_ye = std::min(mchunk_ys + mcy, sizey);
                    const size_t overlap_ys = std::max(ys, mchunk_ys);
                    const size_t overlap_ye = std::min(ye, mchunk_ye);

                    for (size_t mz = mz_start; mz <= mz_end; mz++)
                    {
                        const size_t mchunk_zs = mz * mcz;
                        const size_t mchunk_ze = std::min(mchunk_zs + mcz, sizez);
                        const size_t overlap_zs = std::max(zs, mchunk_zs);
                        const size_t overlap_ze = std::min(ze, mchunk_ze);

                        // Get mchunk reader
                        if (DEBUG_STREAM) t_start = std::chrono::steady_clock::now();
                        packed_reader *chunk_reader = get_mchunk(scale, c, mx, my, mz);
                        if (DEBUG_STREAM) {
                            t_end = std::chrono::steady_clock::now();
                            time_get_mchunk += std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_start).count();
                            mchunks_accessed++;
                        }
                        if (chunk_reader == nullptr)
                            continue;

                        // Iterate over sub-chunks within this mchunk
                        const size_t scx = chunk_reader->chunkx;
                        const size_t scy = chunk_reader->chunky;
                        const size_t scz = chunk_reader->chunkz;

                        // Convert overlap to mchunk-local coordinates (with crop offset)
                        const size_t local_xs = (overlap_xs - mchunk_xs) + chunk_reader->cropstartx;
                        const size_t local_xe = (overlap_xe - mchunk_xs) + chunk_reader->cropstartx;
                        const size_t local_ys = (overlap_ys - mchunk_ys) + chunk_reader->cropstarty;
                        const size_t local_ye = (overlap_ye - mchunk_ys) + chunk_reader->cropstarty;
                        const size_t local_zs = (overlap_zs - mchunk_zs) + chunk_reader->cropstartz;
                        const size_t local_ze = (overlap_ze - mchunk_zs) + chunk_reader->cropstartz;

                        // Sub-chunk index ranges
                        const size_t sx_start = local_xs / scx;
                        const size_t sx_end = (local_xe - 1) / scx;
                        const size_t sy_start = local_ys / scy;
                        const size_t sy_end = (local_ye - 1) / scy;
                        const size_t sz_start = local_zs / scz;
                        const size_t sz_end = (local_ze - 1) / scz;

                        for (size_t sx = sx_start; sx <= sx_end; sx++)
                        {
                            // Sub-chunk X bounds in mchunk-local coordinates
                            const size_t schunk_xs = sx * scx;
                            const size_t schunk_xe = std::min(schunk_xs + scx, chunk_reader->sizex);
                            const size_t schunk_xsize = schunk_xe - schunk_xs;

                            // Overlap with request (in mchunk-local coords)
                            const size_t sc_overlap_xs = std::max(local_xs, schunk_xs);
                            const size_t sc_overlap_xe = std::min(local_xe, schunk_xe);

                            for (size_t sy = sy_start; sy <= sy_end; sy++)
                            {
                                const size_t schunk_ys = sy * scy;
                                const size_t schunk_ye = std::min(schunk_ys + scy, chunk_reader->sizey);
                                const size_t schunk_ysize = schunk_ye - schunk_ys;

                                const size_t sc_overlap_ys = std::max(local_ys, schunk_ys);
                                const size_t sc_overlap_ye = std::min(local_ye, schunk_ye);

                                for (size_t sz = sz_start; sz <= sz_end; sz++)
                                {
                                    const size_t schunk_zs = sz * scz;
                                    const size_t schunk_ze = std::min(schunk_zs + scz, chunk_reader->sizez);
                                    const size_t schunk_zsize = schunk_ze - schunk_zs;

                                    const size_t sc_overlap_zs = std::max(local_zs, schunk_zs);
                                    const size_t sc_overlap_ze = std::min(local_ze, schunk_ze);

                                    // Load sub-chunk
                                    const size_t sub_chunk_id = (sx * chunk_reader->countz * chunk_reader->county) +
                                                                (sy * chunk_reader->countz) + sz;

                                    if (DEBUG_STREAM) t_start = std::chrono::steady_clock::now();
                                    uint16_t *chunk = chunk_reader->load_chunk_direct(sub_chunk_id, schunk_xsize, schunk_ysize, schunk_zsize);
                                    if (DEBUG_STREAM) {
                                        t_end = std::chrono::steady_clock::now();
                                        time_load_chunk += std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_start).count();
                                        chunks_loaded++;
                                    }

                                    if (DEBUG_STREAM) t_start = std::chrono::steady_clock::now();
                                    // Copy all overlapping pixels - OPTIMIZED
                                    // Chunk layout: [X][Y][Z] - Z contiguous
                                    // Output layout: [C][Z][Y][X] - X contiguous
                                    //
                                    // Loop order Z-outer, Y-middle, X-inner for contiguous output writes
                                    // This makes writes cache-friendly (X contiguous in output)

                                    // Pre-compute constants outside all loops
                                    const size_t out_z_stride = osizey * osizex;
                                    const size_t chunk_x_stride = schunk_ysize * schunk_zsize;
                                    const size_t crop_x_offset = chunk_reader->cropstartx;
                                    const size_t crop_y_offset = chunk_reader->cropstarty;
                                    const size_t crop_z_offset = chunk_reader->cropstartz;

                                    // Pre-compute range info
                                    const size_t x_count = sc_overlap_xe - sc_overlap_xs;
                                    const size_t cx_start = sc_overlap_xs - schunk_xs;
                                    const size_t gx_start = sc_overlap_xs - crop_x_offset + mchunk_xs;

                                    for (size_t lz = sc_overlap_zs; lz < sc_overlap_ze; lz++)
                                    {
                                        const size_t gz = lz - crop_z_offset + mchunk_zs;
                                        const size_t cz = lz - schunk_zs;
                                        const size_t out_z_base = c_offset + ((gz - zs) * out_z_stride);

                                        for (size_t ly = sc_overlap_ys; ly < sc_overlap_ye; ly++)
                                        {
                                            const size_t gy = ly - crop_y_offset + mchunk_ys;
                                            const size_t cy = ly - schunk_ys;

                                            // Output row base (X will be contiguous from here)
                                            uint16_t* out_row = out_buffer + out_z_base + ((gy - ys) * osizex) + (gx_start - xs);

                                            // Chunk base for this Y,Z slice
                                            const size_t chunk_yz_base = cy * schunk_zsize + cz;

                                            // Inner X loop - writes are now contiguous!
                                            for (size_t i = 0; i < x_count; i++)
                                            {
                                                const size_t cx = cx_start + i;
                                                const size_t coffset = cx * chunk_x_stride + chunk_yz_base;
                                                out_row[i] = chunk[coffset];
                                            }
                                        }
                                    }
                                    if (DEBUG_STREAM) {
                                        t_end = std::chrono::steady_clock::now();
                                        time_copy += std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_start).count();
                                    }

                                    free(chunk);
                                }
                            }
                        }
                    }
                }
            }
        }

        if (DEBUG_STREAM) {
            std::cerr << "[STREAM DEBUG] region=" << osizex << "x" << osizey << "x" << osizez
                      << " mchunks=" << mchunks_accessed << " chunks=" << chunks_loaded
                      << " | get_mchunk=" << time_get_mchunk << "us"
                      << " load_chunk=" << time_load_chunk << "us"
                      << " copy=" << time_copy << "us"
                      << " total=" << (time_get_mchunk + time_load_chunk + time_copy) << "us"
                      << std::endl;
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