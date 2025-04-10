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

        // if (file.fail())
        //{
        //     std::cerr << "Fopen failed (chunk metadata)" << std::endl;
        //     found = false
        //     return;
        // }

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

        max_chunk_size *= channel_count * chunkx * chunky * chunkz * sizeof(uint16_t);

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

    uint16_t *load_chunk(size_t id, size_t sizex, size_t sizey, size_t sizez)
    {
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
            size_t decomp_size, pix_cnt;
            char *read_decomp_buffer;
            pixtype *read_decomp_buffer_pt;

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
                // Decompress with vidlib
                read_decomp_buffer_pt = decode_stack_264(sizex, sizey, sizez, read_buffer, sel->size);
                pix_cnt = sizex * sizey * sizez;
                decomp_size = pix_cnt * sizeof(uint16_t);
                read_decomp_buffer = (char *)pixtype_to_uint16(read_decomp_buffer_pt, pix_cnt);
                free(read_decomp_buffer_pt);
                break;

            case 3:
                // Decompress with vidlib 2
                read_decomp_buffer_pt = decode_stack_AV1(sizex, sizey, sizez, read_buffer, sel->size);
                pix_cnt = sizex * sizey * sizez;
                decomp_size = pix_cnt * sizeof(uint16_t);
                read_decomp_buffer = (char *)pixtype_to_uint16(read_decomp_buffer_pt, pix_cnt);
                free(read_decomp_buffer_pt);
                break;
            }

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
        uint16_t *out_buffer = (uint16_t *)calloc(buffer_size, 1);

        if (type == SISF)
        {
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
                                                   ((k - zs) * osizey * osizex) +   // Z
                                                   ((j - ys) * osizex) +            // Y
                                                   ((i - xs));                      // X

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