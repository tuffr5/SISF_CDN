#include "crow.h"

#include <nlohmann/json.hpp>

#include "reader.hpp"
#include "utils.hpp"

// #include "hdf5_reader.hpp"
#include "sqlite_tools.hpp"

#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <stdint.h>
#include <cstdint>

#include <iostream>
#include <fstream>
#include <sstream>
#include <stdexcept>

#include <algorithm> // std::sort
#include <string>
#include <string_view>
#include <map>
#include <set>
#include <list>
#include <tuple>
#include <vector>

#include "counter.hpp"

int port = 6000;
int THREAD_COUNT = 32;
bool READ_ONLY_MODE = false;
std::string DATA_PATH = "./data/";

using json = nlohmann::json;
using basic_json = nlohmann::json;

typedef std::tuple<float, float, float, float, int> swc_line;
typedef std::tuple<int, float, float, float, float, int> swc_line_input;

std::unordered_map<std::string, archive_reader *> archive_inventory;
void load_inventory()
{
	std::cout << "|================AVAILABLE DATASETS==================|" << std::endl;

	{
		std::vector<std::string> fnames = glob_tool(std::string(DATA_PATH + "*/metadata.bin"));
		for (std::vector<std::string>::iterator i = fnames.begin(); i != fnames.end(); i++)
		{
			size_t loc = i->find_last_of("/");

			std::string froot = std::string(i->c_str(), i->c_str() + loc);

			loc = froot.find_last_of('/');
			std::string dset_name = froot.substr(loc + 1);

			archive_inventory[dset_name] = new archive_reader(froot, SISF);

			std::cout << "[SISF] ";
			archive_inventory[dset_name]->print_info();
		}
	}

	{
		std::vector<std::string> fnames = glob_tool(std::string(DATA_PATH + "*/zarr.json"));
		for (std::vector<std::string>::iterator i = fnames.begin(); i != fnames.end(); i++)
		{
			size_t loc = i->find_last_of("/");

			std::string froot = std::string(i->c_str(), i->c_str() + loc);

			loc = froot.find_last_of('/');
			std::string dset_name = froot.substr(loc + 1);

			archive_inventory[dset_name] = new archive_reader(froot, ZARR);

			std::cout << "[ZARR] ";
			archive_inventory[dset_name]->print_info();
		}
	}

	std::cout << "|====================================================|" << std::endl;
}

void filter_run(uint16_t *data, size_t data_size, std::tuple<size_t, size_t, size_t> data_shape, size_t channel_count, std::string filter_name, std::string filter_param)
{
	if (filter_name == "offset")
	{
		float off = std::stof(filter_param);
		for (size_t i = 0; i < data_size / sizeof(uint16_t); i++)
		{
			float v = data[i];
			v += off;

			v = std::max((float)std::numeric_limits<uint16_t>::min(), v);
			v = std::min((float)std::numeric_limits<uint16_t>::max(), v);

			data[i] = v;
		}
	}

	if (filter_name == "gamma")
	{
		float gamma = std::stof(filter_param);
		for (size_t i = 0; i < data_size / sizeof(uint16_t); i++)
		{
			float v = data[i];
			v = pow(v, gamma);

			v = std::max((float)std::numeric_limits<uint16_t>::min(), v);
			v = std::min((float)std::numeric_limits<uint16_t>::max(), v);

			data[i] = v;
		}
	}

	if (filter_name == "gammascaled")
	{
		float gamma = std::stof(filter_param);
		for (size_t i = 0; i < data_size / sizeof(uint16_t); i++)
		{
			float v = data[i];

			v /= std::numeric_limits<uint16_t>::max();
			v = log(v);
			v *= gamma;
			v = exp(v);
			v *= std::numeric_limits<uint16_t>::max();

			v = std::max((float)std::numeric_limits<uint16_t>::min(), v);
			v = std::min((float)std::numeric_limits<uint16_t>::max(), v);

			data[i] = v;
		}
	}

	if (filter_name == "scale")
	{
		float scale = std::stof(filter_param);
		for (size_t i = 0; i < data_size / sizeof(uint16_t); i++)
		{
			float v = data[i];
			v *= scale;

			v = std::max((float)std::numeric_limits<uint16_t>::min(), v);
			v = std::min((float)std::numeric_limits<uint16_t>::max(), v);

			data[i] = v;
		}
	}

	if (filter_name == "weiner")
	{
		float nv = std::stof(filter_param);

		double a = 1.0 / (1.0 + nv);
		double b = nv / (1.0 + nv);

		for (size_t i = 0; i < data_size / sizeof(uint16_t); i++)
		{
			float v = data[i];

			v *= a;
			if (i > 0)
			{
				v += b * data[i - 1];
			}

			v = std::max((float)std::numeric_limits<uint16_t>::min(), v);
			v = std::min((float)std::numeric_limits<uint16_t>::max(), v);

			data[i] = v;
		}
	}

	if (filter_name == "gaussian")
	{
		float sig = std::stof(filter_param);

		float *data_tmp = (float *)calloc(data_size / sizeof(uint16_t), sizeof(float));

		for (size_t i = 0; i < data_size / sizeof(uint16_t); i++)
		{
			float v = data[i];
			data_tmp[i] = v;
		}

		size_t osizei = std::get<0>(data_shape);
		size_t osizej = std::get<1>(data_shape);
		size_t osizek = std::get<2>(data_shape);

		for (size_t c = 0; c < channel_count; c++)
		{
			for (size_t i = 0; i < osizei; i++)
			{
				for (size_t j = 0; j < osizej; j++)
				{
					for (size_t k = 0; k < osizek; k++)
					{
						const int window_size = 3;

						const int istart = ((int)i) - window_size;
						const int iend = ((int)i) + window_size;
						const int jstart = ((int)j) - window_size;
						const int jend = ((int)j) + window_size;
						const int kstart = ((int)k) - window_size;
						const int kend = ((int)k) + window_size;

						double sum = 1e-9;
						double n = 1e-9;

						for (int di = istart; di <= iend; di++)
						{
							for (int dj = jstart; dj <= jend; dj++)
							{
								for (int dk = kstart; dk <= kend; dk++)
								{
									if (di < 0 || dj < 0 || dk < 0 || di >= osizei || dj >= osizej || dk >= osizek)
									{
										continue;
									}

									const size_t ioffset = (c * osizei * osizej * osizek) + // C
														   (dk * osizei * osizej) +			// Z
														   (dj * osizei) +					// Y
														   (di);							// X

									float dist = powf32(di - i, 2) + powf32(dj - j, 2) + powf32(dk - k, 2); // d^2
									float coeff = exp(-0.5 * dist / powf32(sig, 2));						// / (sig * sqrtf32(M_2_PI));

									float toadd = data[ioffset];
									toadd *= coeff;

									sum += toadd;
									n += coeff;
								}
							}
						}

						sum /= n;
						const size_t ooffset = (c * osizei * osizej * osizek) + // C
											   (k * osizei * osizej) +			// Z
											   (j * osizei) +					// Y
											   (i);								// X

						data_tmp[ooffset] = sum;
					}
				}
			}
		}

		for (size_t i = 0; i < data_size / sizeof(uint16_t); i++)
		{
			float v = data_tmp[i];

			v = std::max((float)std::numeric_limits<uint16_t>::min(), v);
			v = std::min((float)std::numeric_limits<uint16_t>::max(), v);

			data[i] = v;
		}

		free(data_tmp);
	}
}

template <typename T>
T *compute_z_max_projection(
	T *input_buffer,
	size_t x_size, size_t y_size, size_t z_size,
	size_t channel_count)
{
	// Input validation
	if (!input_buffer || x_size == 0 || y_size == 0 || z_size == 0 || channel_count == 0)
	{
		return nullptr;
	}

	size_t xy_plane_size = x_size * y_size;

	if (xy_plane_size / x_size != y_size)
	{
		return nullptr;
	}

	size_t output_size = xy_plane_size * channel_count;
	if (output_size / xy_plane_size != channel_count)
	{
		return nullptr;
	}

	T *output_buffer = (T *)malloc(output_size * sizeof(T));
	if (!output_buffer)
	{
		return nullptr;
	}

	for (size_t xy = 0; xy < xy_plane_size; xy++)
	{
		for (size_t c = 0; c < channel_count; c++)
		{
			size_t output_idx = (xy * channel_count) + c;
			output_buffer[output_idx] = std::numeric_limits<T>::lowest();
		}
	}

	for (size_t z = 0; z < z_size; z++)
	{
		for (size_t xy = 0; xy < xy_plane_size; xy++)
		{
			for (size_t c = 0; c < channel_count; c++)
			{
				size_t output_idx = (xy * channel_count) + c;
				size_t input_idx = (z * xy_plane_size * channel_count) + output_idx;
				output_buffer[output_idx] = std::max(output_buffer[output_idx], input_buffer[input_idx]);
			}
		}
	}

	return output_buffer;
}

int main(int argc, char *argv[])
{
	{
		if (argc > 1)
		{
			std::istringstream iss(argv[1]);
			iss >> port;

			if (argc > 2)
			{
				DATA_PATH = std::string(argv[2]);
			}
		}
	}

	std::string read_only = read_env_variable("READ_ONLY");
	if (read_only.size() > 0)
	{
		READ_ONLY_MODE = true;
		std::cout << "Using read only mode." << std::endl;
	}

	std::string thread_count = read_env_variable("THREAD_COUNT");
	if (thread_count.size() > 0)
	{
		std::istringstream tc(thread_count);
		tc >> THREAD_COUNT;
	}

	load_inventory();

	// Create App with CORS enabled
	crow::App<crow::CORSHandler, CounterMiddleware> app;

	// Disable CORS directly
	auto &cors = app.get_middleware<crow::CORSHandler>();
	cors.global().origin("*");

	// Default route to check server status
	CROW_ROUTE(app, "/")
	([]()
	 { return "Server is up!"; });

	CROW_ROUTE(app, "/performance")
	([]()
	 {
		std::stringstream out;

		out << "<html>\n";
		out << "<head>\n";
		out << "<style>table, th, td {border: 1px solid black;}</style>\n";
		out << "</head>\n";
		out << "<body>\n";
		out << "<table>\n";

		out << "<tr>\n";
		out << "\t<th>Counter Name</th><th>Size</th><th>Average (&micro;s)</th><th>Std. Dev. (&micro;s)</th><th>N</th>\n";
		out << "</tr>\n";

		perf_counter_mutex.lock();
		for(const auto &a : perf_counters) {
			// Skip line if empty
			if(a.second.size() == 0) continue;

			double mu = 0;
			for(size_t a : a.second) {
				mu += a;
			}
			mu /= a.second.size();

			double std = 0;
			for(size_t a : a.second) {
				std += pow(((double) a) - mu, 2);
			}
			std /= a.second.size();
			std = pow(std, 0.5);

			out << "<tr>\n";
			out << "\t<td>" << std::get<0>(a.first) << " / " << std::get<1>(a.first) << " / " << std::get<2>(a.first) << "</td>\n";
			out << "\t<td>" << std::get<3>(a.first) << "&times;" << std::get<4>(a.first) << "&times;" << std::get<5>(a.first) << "</td>\n";
			out << "\t<td>" << mu << "</td>\n";
			out << "\t<td>" << std << "</td>\n";
			out << "\t<td>" << a.second.size() << "</td>\n";
			out << "</tr>\n";
		}
		perf_counter_mutex.unlock();

		out << "</table>\n";
		out << "</body>\n";
		out << "</html>";

		return out.str(); });

	CROW_ROUTE(app, "/access")
	([]()
	 {
		std::stringstream out;

		out << "<html>\n";
		out << "<head>\n";
		out << "<style>table, th, td {border: 1px solid black;}</style>\n";
		out << "</head>\n";
		out << "<body>\n";
		out << "<table>\n";

		out << "<tr>\n";
		out << "\t<th>IP</th><th>N</th>\n";
		out << "</tr>\n";

		//perf_counter_mutex.lock();
		for(const auto &a : ip_counter) {
			out << "<tr>\n";
			out << "\t<td>" << a.first << "</td>\n";
			out << "\t<td>" << a.second << "</td>\n";
			out << "</tr>\n";
		}
		//perf_counter_mutex.unlock();

		out << "</table>\n";
		out << "</body>\n";
		out << "</html>";

		return out.str(); });

	CROW_ROUTE(app, "/inventory")
	([]()
	 {
		std::stringstream out;

		out << "<html>\n";
		out << "<body>\n";
		out << "<style>table, th, td { border:1px solid black; }</style>\n";
		out << "<table style=\"width:100%\">\n";

		out << "<tr>";
		out << "\t<th>Dataset</th>";
		out << "\t<th>Channels</th>";
		out << "\t<th>X Size (px)</th>";
		out << "\t<th>Y Size (px)</th>";
		out << "\t<th>Z Size (px)</th>";
		out << "\t<th>X Res (nm)</th>";
		out << "\t<th>Y Res (nm)</th>";
		out << "\t<th>Z Res (nm)</th>";
		out << "\t<th>X Chunk Size (px)</th>";
		out << "\t<th>Y Chunk Size (px)</th>";
		out << "\t<th>Z Chunk Size (px)</th>";
		out << "\t<th>Scales Loaded</th>";
		out << "\t<th>Dataset URL</th>";
		out << "</tr>\n";

		std::vector<std::string> sample_names;
		std::map<std::string, std::string> sample_table;
		sample_names.reserve(archive_inventory.size());

		for ( const auto &kvpair : archive_inventory ) {
			std::string this_name(kvpair.first);
			sample_names.push_back(this_name);

			std::stringstream toadd;

			toadd << "<tr>\n";
			toadd << "\t<td>" << kvpair.first << "</td>\n";
			toadd << "\t<td>" << kvpair.second->channel_count << "</td>\n";
			toadd << "\t<td>" << kvpair.second->sizex << "</td>\n";
			toadd << "\t<td>" << kvpair.second->sizey << "</td>\n";
			toadd << "\t<td>" << kvpair.second->sizez << "</td>\n";
			toadd << "\t<td>" << kvpair.second->resx << "</td>\n";
			toadd << "\t<td>" << kvpair.second->resy << "</td>\n";
			toadd << "\t<td>" << kvpair.second->resz << "</td>\n";
			toadd << "\t<td>" << kvpair.second->mchunkx << "</td>\n";
			toadd << "\t<td>" << kvpair.second->mchunky << "</td>\n";
			toadd << "\t<td>" << kvpair.second->mchunkz << "</td>\n";
			
			toadd << "\t<td>";
			for(size_t i = 0; i < kvpair.second->scales.size(); i++) {
				toadd << kvpair.second->scales.at(i);
				if(i != (kvpair.second->scales.size() - 1))
				{
					toadd << ",";
				}
			}
			toadd << "</td>\n";

			toadd << "\t<td>" << "precomputed://https://sonic.cai-lab.org/data/" << kvpair.first << "</td>\n";
			toadd << "</tr>\n";

			sample_table[this_name] = toadd.str();
    	}

		std::sort(sample_names.begin(), sample_names.end());

		for(auto this_name : sample_names) {
			out << sample_table[this_name];
		}

		out << "</table>\n";
		out << "</body>\n";
		out << "</html>\n";

		return out.str(); });

	//	ENDPOINT: /data/<string>/info
	CROW_ROUTE(app, "/<string>/info")
	([](crow::response &res, std::string data_id)
	 {
		data_id = str_first(data_id, '+');
		auto archive_search = archive_inventory.find(data_id);

		if(archive_search == archive_inventory.end()) {
			res.end("File not found.");
			return;
		}

		archive_reader * reader = archive_search->second;

		std::vector<basic_json> scales;
		for(size_t scale : reader->scales) {
			std::tuple<size_t, size_t, size_t> res_scaled = reader->get_res(scale);
			std::vector<uint32_t> res = {
				(uint32_t) std::get<0>(res_scaled),
				(uint32_t) std::get<1>(res_scaled),
				(uint32_t) std::get<2>(res_scaled)
			};

			basic_json to_add;
			to_add["chunk_sizes"] = {
				//{64, 1, 1},
				//{1, 64, 1},
				//{1, 1, 64}
				{256, 256, 1  },
				{256, 1,   256},
				{1,   256, 256},
				{32, 32, 32},

			};

			to_add["encoding"] = "raw";
			to_add["key"] = std::to_string(scale);
			to_add["resolution"] = res;

			std::tuple<size_t, size_t, size_t> sizes = reader->get_size(scale);
			to_add["size"] = {
				std::get<0>(sizes),
				std::get<1>(sizes),
				std::get<2>(sizes)
			};
			to_add["voxel_offset"] = {0,0,0};

			scales.push_back(to_add);
		}

		json response = {
			{"type", "image"},
			{"@type", "neuroglancer_multiscale_volume"},
			{"data_type", "uint16"}, // TODO change based on dtype
			{"num_channels", reader->channel_count}
		};
		response["scales"] = scales;

		res.write(response.dump());
    	res.end(); });

	// ENDPOINT: /data/<string>/zmaxprojection/num_zslices/info
	CROW_ROUTE(app, "/<string>/zmaxprojection/<int>/info")
	([](crow::response &res, std::string data_id, int num_zslices)
	 {
			data_id = str_first(data_id, '+');
			auto archive_search = archive_inventory.find(data_id);
	
			if(archive_search == archive_inventory.end()) {
				res.end("File not found.");
				return;
			}
	
			archive_reader * reader = archive_search->second;
	
			std::vector<basic_json> scales;
			for(size_t scale : reader->scales) {
				std::tuple<size_t, size_t, size_t> res_scaled = reader->get_res(scale);
				std::vector<uint32_t> res = {
					(uint32_t) std::get<0>(res_scaled),
					(uint32_t) std::get<1>(res_scaled),
					(uint32_t) std::get<2>(res_scaled)
				};
	
				basic_json to_add;
				
				to_add["chunk_sizes"] = {
					//{64, 1, 1},
					//{1, 64, 1},
					//{1, 1, 64}
					{256, 256, 1 },
					{256, 1,   256},
					{1,   256, 256},
					{32, 32, 32},
	
				};
	
				to_add["encoding"] = "raw";
				to_add["key"] = std::to_string(scale);
				to_add["resolution"] = res;
	
				std::tuple<size_t, size_t, size_t> sizes = reader->get_size(scale);
				to_add["size"] = {
					std::get<0>(sizes),
					std::get<1>(sizes),
					std::get<2>(sizes)
				};
				to_add["voxel_offset"] = {0,0,0};
				to_add["num_zslices"] = std::to_string(num_zslices);
	
				scales.push_back(to_add);
			}
	
			json response = {
				{"type", "image"},
				{"@type", "neuroglancer_multiscale_volume"},
				{"data_type", "uint16"}, // TODO change based on dtype
				{"num_channels", reader->channel_count}
			};
			response["scales"] = scales;
	
			res.write(response.dump());
			res.end(); });

	CROW_ROUTE(app, "/<string>/tracing/<string>/<string>")
	([](const crow::request &req, crow::response &res, std::string data_id, std::string pt_in_s1, std::string pt_in_s2)
	 {
		auto soma_param = req.url_params.get("is_soma");
		bool is_soma = soma_param != nullptr && strcmp(soma_param, "true") == 0;
		std::cout << "is_soma=" << is_soma << std::endl;
		int pt1[3], pt2[3];
		if(sscanf(pt_in_s1.c_str(), "%d,%d,%d", &pt1[0], &pt1[1], &pt1[2]) != 3) {
			res.end("Invalid point descriptor " + pt_in_s1);
			return;
		}
		if(sscanf(pt_in_s2.c_str(), "%d,%d,%d", &pt2[0], &pt2[1], &pt2[2]) != 3) {
			res.end("Invalid point descriptor " + pt_in_s2);
			return;
		}

		data_id = str_first(data_id, '+');

		auto archive_search = archive_inventory.find(data_id);
		if(archive_search == archive_inventory.end()) {
			res.end("File not found.");
			return;
		}
		archive_reader * reader = archive_search->second;

		if(pt1[0] < 0 || pt1[1] < 0 || pt1[2] < 0 || pt1[0] >= reader->sizex || pt1[1] >= reader->sizey || pt1[2] >= reader->sizez) {
			res.code = 400;
			res.end("Pt 1 out of bounds.");
			return;
		}

		if(pt2[0] < 0 || pt2[1] < 0 || pt2[2] < 0 || pt2[0] >= reader->sizex || pt2[1] >= reader->sizey || pt2[2] >= reader->sizez) {
			res.code = 400;
			res.end("Pt 2 out of bounds.");
			return;
		}

		if (is_soma) {
			if (pt1[2] != pt2[2]) {
				res.code = 400;
				res.end("Soma tracing requires both points to be in the same z slice.");
				return;
			}
		}

		std::stringstream out;

		auto coord_cmp = [](astar_cell *a, astar_cell *b) { return a->x < b->x || (a->x == b->x && a->y < b->y) || (a->x == b->x && a->y == b->y && a->z < b->z); };
		std::set<astar_cell *, decltype(coord_cmp)> closed_set(coord_cmp);
		std::priority_queue<astar_cell *, std::vector<astar_cell*>, std::function<bool(astar_cell *, astar_cell *)>> open_set(compare_astar_cell);

		astar_cell *start = new astar_cell(pt1[0], pt1[1], pt1[2], 0, 0, 0);
		astar_cell *end = new astar_cell(pt2[0], pt2[1], pt2[2], 0, 0, 0);

		start->h = start->euclidean_distance(end);
		start->load_color(reader, 2);
		end->load_color(reader, 2);

		std::cout << "start=" << start->color_intensity << " end=" << end->color_intensity << std::endl;

		const double threshold_factor = 0.6;
		const double avg_color_threshold = threshold_factor * (start->color_intensity + end->color_intensity) / 2.0;

		open_set.push(start);

		astar_cell * target;
		size_t N = 0;
		const size_t N_limit = 100000;
		const size_t channel_count = reader->channel_count;

		const bool use_buffering = true;
		uint16_t * region_data = nullptr;

		int region_start_x = std::min(start->x, end->x);
		int region_end_x = std::max(start->x, end->x);
		int region_start_y = std::min(start->y, end->y);
		int region_end_y = std::max(start->y, end->y);
		int region_start_z = std::min(start->z, end->z);
		int region_end_z = std::max(start->z, end->z);

		const int window = 5; // window to dilate sampled region by for overlaps 

		region_start_x -= window;
		region_end_x += window;
		region_start_y -= window;
		region_end_y += window;
		region_start_z -= window;
		region_end_z += window;

		region_start_x = std::max(region_start_x, (int) 0);
		region_start_y = std::max(region_start_y, (int) 0);
		region_start_z = std::max(region_start_z, (int) 0);
		region_end_x = std::min(region_end_x, (int) reader->sizex); 
		region_end_y = std::min(region_end_y, (int) reader->sizey); 
		region_end_z = std::min(region_end_z, (int) reader->sizez); 

		const size_t region_size_x = region_end_x - region_start_x;
		const size_t region_size_y = region_end_y - region_start_y;
		const size_t region_size_z = region_end_z - region_start_z;

		if(true) { 
			std::cout << "Getting range: [" << region_start_x << "," << region_end_x << "], ["
					<< region_start_y << "," << region_end_y << "], [" << region_start_z << ","
					<< region_end_z << "]" << std::endl;
		}

		if (use_buffering)
		{
			region_data = reader->load_region(1, region_start_x, region_end_x, region_start_y, region_end_y, region_start_z, region_end_z);
		}

		while (!open_set.empty() && N<N_limit) {
			target = open_set.top();
			if(true) {
				std::cout << "Found " << target->x << ',' << target->y << ',' << target->z
					<< " f=" << target->f() 
					//<< " g=" << target->g()
					//<< " h=" << target->h()
					<< " c=" << target->color_intensity
					<< std::endl;
			}
			open_set.pop();

			if(*target == *end) {
				break;
			}



			for(auto step : (is_soma ? neighbor_steps_no_z : neighbor_steps)) {
				const int dx = std::get<0>(step);
				const int dy = std::get<1>(step);
				const int dz = std::get<2>(step);
				const double ds = std::get<3>(step);

				const int new_x = target->x + dx;
				const int new_y = target->y + dy;
				const int new_z = target->z + dz;

				const double new_g = target->g + ds;

				if (new_x < 0 || new_y < 0 || new_z < 0 || new_x >= reader->sizex || new_y >= reader->sizey || new_z >= reader->sizez)
				{
					continue;
				}

				if (use_buffering)
				{
					if (new_x < region_start_x ||
						new_x >= region_end_x ||
						new_y < region_start_y ||
						new_y >= region_end_y ||
						new_z < region_start_z ||
						new_z >= region_end_z)
					{
						continue;
					}
				}

				astar_cell *new_pt = new astar_cell(new_x, new_y, new_z, new_g, 0, target);
				new_pt->h = new_pt->euclidean_distance(end);

				bool dont_add = false;
				dont_add |= closed_set.find(new_pt) != closed_set.end();

				if (!dont_add)
				{
					if (!use_buffering)
					{
						const size_t window_size = 2;
						new_pt->load_color(reader, window_size);
					}
					else
					{
						new_pt->define_color_vector(channel_count);
						for (size_t c = 0; c < channel_count; c++)
						{
							size_t offset = (c * region_size_x * region_size_y * region_size_z) +
											((new_pt->z - region_start_z) * region_size_y * region_size_x) +
											((new_pt->y - region_start_y) * region_size_x) +
											(new_pt->x - region_start_x);

							new_pt->color_vector[c] += region_data[offset];
						}
						new_pt->calculate_intensity_average();
					}

					dont_add |= avg_color_threshold > new_pt->color_intensity;
				}

				if (!dont_add && new_pt->channel_count > 1)
				{
					double color_angle = start->color_angle(new_pt);
					dont_add |= color_angle > .5;
				}

				//for(size_t i = 0; i < new_pt->channel_count; i++) {
				//	std::cout << new_pt->color_vector[i];
				//	if(i != new_pt->channel_count - 1) std::cout << ',';
				//}
				//std::cout << std::endl;

				if(dont_add) {
					new_pt->delete_color_vector();
					delete new_pt;
				} else {
					open_set.push(new_pt);
					closed_set.insert(new_pt);
				}
			}

			N++;
		}

		if(use_buffering) {
			free(region_data);
		}
		std::cout << "Iterations: " << N << '\n';

		// Print results
		if(*target == *end) {
			std::deque<astar_cell*> stack;
			while(target != 0) {
				stack.push_back(target);
				target = target->parent;
			}

			while(!stack.empty()) {
				target = stack.back();
				stack.pop_back();

				out << target->x << ',' << target->y << ',' << target->z << '\n';
			}
		} else {
			res.code = 400;
			if(N == N_limit) {
				out << "Ran out of iterations" << '\n';
			} else {
				out << "Failed to find path" << '\n';
			}
		}

		delete end;
		for (auto &entry : closed_set) {
			entry->delete_color_vector();
			delete entry;
		}

		res.end(out.str()); });

	// @app.route("/meanshift/<data_id>/<point>/")
	CROW_ROUTE(app, "/<string>/meanshift/<string>")
	([](crow::response &res, std::string data_id, std::string pt_in_s)
	 {
		data_id = str_first(data_id, '+');
			
		auto archive_search = archive_inventory.find(data_id);
		if(archive_search == archive_inventory.end()) {
			res.end("File not found.");
			return;
		}
		archive_reader * reader = archive_search->second;

		float pt_in_f[3];
		sscanf(pt_in_s.c_str(), "%f,%f,%f", &pt_in_f[0], &pt_in_f[1], &pt_in_f[2]);
		int64_t pt_in[3];
		for(size_t i = 0; i < 3; i++) {
			pt_in[i] = (int) pt_in_f[i];
		}

		if(pt_in[0] < 0 || pt_in[1] < 0 || pt_in[2] < 0 || pt_in[0] >= reader->sizex || pt_in[1] >= reader->sizey || pt_in[2] >= reader->sizez) {
			res.end("Coord out of bounds");
			return;
		}

		//std::cout << "Starting on pt " << pt_in[0] << ", " << pt_in[1] << ", " << pt_in[2] << std::endl;

		const int64_t offsets[3] = {20, 20, 20};
		const int64_t x_begin = std::max(pt_in[0] - offsets[0], (int64_t) 0);
		const int64_t x_end = std::min(pt_in[0] + offsets[0], (int64_t) reader->sizex - 1);
		const int64_t y_begin = std::max(pt_in[1] - offsets[1], (int64_t) 0);
		const int64_t y_end = std::min(pt_in[1] + offsets[1], (int64_t) reader->sizey - 1);
		const int64_t z_begin = std::max(pt_in[2] - offsets[2], (int64_t) 0);
		const int64_t z_end = std::min(pt_in[2] + offsets[2], (int64_t) reader->sizez - 1);

		const int64_t chunk_sizes[3] = {x_end - x_begin, y_end - y_begin, z_end - z_begin};
		const size_t channel_count = reader->channel_count;

		uint16_t * read_buffer = reader->load_region(1, x_begin, x_end, y_begin, y_end, z_begin, z_end);
		if(read_buffer == 0) {
			res.end("Failed to open reader.");
			return;
		}

		//mean_shift(read_buffer, channel_count, chunk_offsets, chunk_sizes, pt, new_point);

		int64_t p[3] = {pt_in[0] - x_begin, pt_in[1] - y_begin, pt_in[2] - z_begin};

		const int xradius = 5;
		const int yradius = 5;
		const int zradius = 5;

		const size_t iter_count = 5;
		for (size_t r = 0; r < iter_count; ++r) {
			//std::cout << "I " << (int) p[0] << ',' << (int) p[1] << ',' << (int) p[2] << std::endl;

			const size_t xmin = std::max((int64_t) 0, (int64_t) p[0] - xradius);
			const size_t xmax = std::min((int64_t)reader->sizex - 1, (int64_t) p[0] + xradius);
			const size_t ymin = std::max((int64_t) 0, (int64_t) p[1] - yradius);
			const size_t ymax = std::min((int64_t)reader->sizey - 1,(int64_t)  p[1] + yradius);
			const size_t zmin = std::max((int64_t) 0, (int64_t) p[2] - zradius);
			const size_t zmax = std::min((int64_t)reader->sizez - 1, (int64_t) p[2] + zradius);

			double xsum = 0;
			double ysum = 0;
			double zsum = 0;
			double total_intensity = 1e-9;

			for (size_t k = zmin; k < zmax; ++k) {
				for (size_t j = ymin; j < ymax; ++j) {
					for (size_t i = xmin; i < xmax; ++i) {
						double intensity = 0;
						for (int c = 0; c < channel_count; ++c) {
							const size_t ooffset = (c * chunk_sizes[0] * chunk_sizes[1] * chunk_sizes[2]) + // C
                                               (k * chunk_sizes[0] * chunk_sizes[1]) +                  // Z
                                               (j * chunk_sizes[0]) +                                   // Y
                                               (i);                                                     // X

							////                       CH             Z              Y              X         
							//uint16_t out_buffer[channel_count][chunk_sizes[2]][chunk_sizes[1]][chunk_sizes[0]];
							intensity += read_buffer[ooffset];
						}

						xsum += intensity * (double) i;
						ysum += intensity * (double) j;
						zsum += intensity * (double) k;
						total_intensity += intensity;
					}
				}
			}

			xsum /= total_intensity;
			ysum /= total_intensity;
			zsum /= total_intensity;

			p[0] = std::round(xsum);
			p[1] = std::round(ysum);
			p[2] = std::round(zsum);
		}

		p[0] += x_begin;
		p[1] += y_begin;
		p[2] += z_begin;

		res.write(std::to_string(p[0]));
		res.write(",");
		res.write(std::to_string(p[1]));
		res.write(",");
		res.write(std::to_string(p[2]));

		// Free buffers
		free(read_buffer);

		res.end(); });

	CROW_ROUTE(app, "/<string>/skeleton/info")
	([](crow::response &res, std::string data_id)
	 {
		data_id = str_first(data_id, '+');
		auto archive_search = archive_inventory.find(data_id);
		if(archive_search == archive_inventory.end()) {
			res.end();
			return;
		}

    	json response = {
			{"type", "segmentation"},
			{"@type", "neuroglancer_skeletons"},
			{"segment_properties", "segment_properties"}
		};
		res.write(response.dump());
		res.end(); });

	CROW_ROUTE(app, "/<string>/skeleton/segment_properties/info")
	([](crow::response &res, std::string data_id)
	 {	
		data_id = str_first(data_id, '+');

		std::vector<basic_json> ids = {}; // array of std::string, numbers?
		std::vector<basic_json> values = {}; // array of std::string

		std::vector<std::string> trace_file = glob_tool(DATA_PATH + data_id + "/traces.sql");

		if(trace_file.size() > 0) {
			std::cout << "Loading file: " << trace_file[0] << std::endl; 
			
			auto db = database_interface(trace_file[0]);

			callback_vector neurons = db.run("SELECT rowid FROM NEURONS;");

			//CLOSE DB EARLY

			for(stringvec a : neurons) {
				ids.push_back(a[0]);
				values.push_back(a[0]); // Replace with file name
			}
		}
		
		json id_prop;
		id_prop["id"] = "label";
		id_prop["type"] = "label";
		id_prop["values"] = values;

		json toadd_inline;
		toadd_inline["ids"] = ids;
		// vector of one item 
		std::vector<json> properties;
		properties.push_back(id_prop);
		toadd_inline["properties"] = properties;

		json response;
		response["@type"] = "neuroglancer_segment_properties";
		response["inline"] = toadd_inline;
    	
    	res.write(response.dump());
    	res.end(); });

	CROW_ROUTE(app, "/echo").methods("POST"_method)([](const crow::request &req)
													{
		crow::multipart::message msg(req);
		
		for(int i = 0; i < msg.parts.size(); i++) {
			std::cout << "Part " << i << std::endl;
			for (auto& item_h : msg.parts[i].headers)
			{
				std::cout << item_h.first << ": " << item_h.second.value;
				for (auto& it : item_h.second.params)
				{
					std::cout << "; " << it.first << '=' << it.second;
				}
				std::cout << std::endl;
			}
		}

		return crow::response(req.body); });

	CROW_ROUTE(app, "/<string>/skeleton_api")([](std::string data_id)
											  {
		
		data_id = str_first(data_id, '+');
	
		std::string trace_file_path = DATA_PATH + data_id + "/traces.sql";
		std::vector<std::string> trace_file = glob_tool(trace_file_path);
	
		bool file_created = false;
		
		if(trace_file.size() == 0) {
			std::cout << "Trace file not found. Creating a new one." << std::endl;
			
			// Create a new trace file (assuming DATA_PATH + data_id directory already exists)
			trace_file.push_back(trace_file_path);
			file_created = true;
			
			// Initialize database with required tables
			auto db = database_interface(trace_file[0]);
			
			// Create SWC table
			db.run(R"(
				CREATE TABLE IF NOT EXISTS SWC(
					I INT                  NOT NULL,
					NEURONID INT           NOT NULL,
					PARENTID INT           NOT NULL,
					X REAL                 NOT NULL,
					Y REAL                 NOT NULL,
					Z REAL                 NOT NULL,
					R REAL                 NOT NULL,
					T INT                  NOT NULL,
					USERID INT             NOT NULL,
					TIMESTAMP DATETIME DEFAULT CURRENT_TIMESTAMP
				);
			)");
			
			// Create NEURONS table
			db.run(R"(
				CREATE TABLE IF NOT EXISTS NEURONS(
					SOMAX REAL,
					SOMAY REAL,
					SOMAZ REAL,
					CELLTYPE INT,
					NOTES TEXT,
					TIMESTAMP DATETIME DEFAULT CURRENT_TIMESTAMP
				);
			)");
			
			std::cout << "Database tables created successfully." << std::endl;
		}
		
		json response = {};
		
		// Add status to indicate if file was created
		response["status"] = file_created ? "file_created" : "file_found";
		
		return crow::response(response.dump()); });

	CROW_ROUTE(app, "/<string>/skeleton_api/delete/<int>")
	([](std::string data_id, int neuron_id)
	 {
		if(READ_ONLY_MODE) {
			return crow::response(crow::status::BAD_REQUEST);
		}

		data_id = str_first(data_id, '+');

		std::vector<std::string> trace_file = glob_tool(DATA_PATH + data_id + "/traces.sql");

		if(trace_file.size() == 0) {
			std::cerr << "Failed to find sql file." << std::endl;
			return crow::response(crow::status::BAD_REQUEST);
		}
		
		std::cout << "Loading file: " << trace_file[0] << std::endl; 

		std::string sql = "";
		sql += "DELETE FROM SWC WHERE neuronid=" + std::to_string(neuron_id) + ";";
		sql += "DELETE FROM NEURONS WHERE rowid=" + std::to_string(neuron_id) + ";";

		{
			auto db = database_interface(trace_file[0]);
			auto retv = db.run(sql);
		}
		
		json response = {
			{"neuronid", neuron_id}
		};

		return crow::response( response.dump() ); });

	CROW_ROUTE(app, "/<string>/skeleton_api/replace/<int>").methods("POST"_method)([](const crow::request &req, std::string data_id, int neuron_id)
																				   {
		if(READ_ONLY_MODE) {
			return crow::response(crow::status::BAD_REQUEST);
		}

		data_id = str_first(data_id, '+');

		crow::multipart::message msg(req);

		int part_id = -1;
		for(int i = 0; i < msg.parts.size(); i++) {
			for (auto& item_h : msg.parts[i].headers)
			{
				for (auto& it : item_h.second.params)
				{
					if(it.first == "name" && it.second == "data") { // Identify the part with the data
						part_id = i;
					}
				}
			}
		}

		if(part_id == -1) {
			std::cerr << "Failed to find file header." << std::endl;
			return crow::response(crow::status::BAD_REQUEST);
		}

		std::vector<std::string> trace_file = glob_tool(DATA_PATH + data_id + "/traces.sql");

		if(trace_file.size() == 0) {
			std::cerr << "Failed to find sql file." << std::endl;
			return crow::response(crow::status::BAD_REQUEST);
		}
		
		std::cout << "Using file: " << trace_file[0] << std::endl;

		///////////////////////

		std::map<int64_t, int64_t> old_to_new;
		old_to_new.insert( {-1, -1} );
		std::list<swc_line_input> lines;
		std::list<std::tuple<int, int>> edges;

		int id, parent, type;
		float x, y, z, r;

		// id   type   X       Y         Z      R   parent
		// 16613 2 4341.180 5706.911 9866.430 1.000 16612
		uint32_t i = 1;
		std::string line;
		std::istringstream infile(msg.parts[part_id].body);
		while (std::getline(infile, line))
		{
			if (line.length() <= 1)
				continue;
			std::istringstream iss(line);

			iss >> id;
			iss >> type;
			iss >> x;
			iss >> y;
			iss >> z;
			iss >> r;
			iss >> parent;

			lines.push_back({type, x, y, z, r, parent});
			old_to_new.insert({id, i});

			i++;
		}

		std::stringstream sql_builder("");

		sql_builder << "BEGIN TRANSACTION;";
		
		sql_builder << "DELETE FROM SWC WHERE neuronid=" + std::to_string(neuron_id) + ";";

		sql_builder << "INSERT INTO SWC"
					<< "(I,NEURONID,PARENTID,X,Y,Z,R,T,USERID)"
					<< " VALUES ";

		i = 1;
		for (swc_line_input a : lines)
		{
			const auto [type, x, y, z, r, parent] = a;
			int64_t parent_new = old_to_new[parent];
			
			int user = -1;
			sql_builder << "(" << i << ","
						<< neuron_id << ","
						<< parent_new << ","
						<< x << ',' << y << ',' << z << ',' << r << ',' << type << ',' << user
						<< "),";
			i++;
		}

		sql_builder.seekp(-1,sql_builder.cur);
		sql_builder << ";";

		sql_builder << "COMMIT;";

		{
			auto db = database_interface(trace_file[0]);
			auto retv = db.run(sql_builder.str());
		}

		json response = {
			{"neuronid", neuron_id}
		};

		return crow::response( response.dump() ); });

	CROW_ROUTE(app, "/<string>/skeleton_api/ls")
	([](std::string data_id)
	 {
		data_id = str_first(data_id, '+');

		std::vector<std::string> trace_file = glob_tool(DATA_PATH + data_id + "/traces.sql");

		if(trace_file.size() == 0) {
			std::cerr << "Failed to find sql file." << std::endl;
			return crow::response(crow::status::BAD_REQUEST);
		}
		
		auto db = database_interface(trace_file[0]);
		callback_vector neurons = db.run("SELECT rowid FROM NEURONS;");


		json response = {};
		response["neuronids"] = std::vector<basic_json>();

		std::stringstream out("");
		for(stringvec b : neurons) { 
			response["neuronids"].push_back(b[0]);
		}
		
		return crow::response( response.dump() ); });

	CROW_ROUTE(app, "/<string>/skeleton_api/upload").methods("POST"_method)([](const crow::request &req, std::string data_id)
																			{
		if(READ_ONLY_MODE) {
			return crow::response(crow::status::BAD_REQUEST);
		}

		data_id = str_first(data_id, '+');

		crow::multipart::message msg(req);

		int part_id = -1;
		for(int i = 0; i < msg.parts.size(); i++) {
			for (auto& item_h : msg.parts[i].headers)
			{
				for (auto& it : item_h.second.params)
				{
					if(it.first == "name" && it.second == "data") { // Identify the part with the data
						part_id = i;
					}
				}
			}
		}

		if(part_id == -1) {
			std::cerr << "Failed to find file header." << std::endl;
			return crow::response(crow::status::BAD_REQUEST);
		}

		std::vector<std::string> trace_file = glob_tool(DATA_PATH + data_id + "/traces.sql");

		if(trace_file.size() == 0) {
			std::cerr << "Failed to find sql file." << std::endl;
			return crow::response(crow::status::BAD_REQUEST);
		}
		
		std::cout << "Loading file: " << trace_file[0] << std::endl; 

		std::string sql = "INSERT INTO NEURONS DEFAULT VALUES;"
					"SELECT last_insert_rowid();";

		callback_vector a;
		{
			auto db = database_interface(trace_file[0]);
			a = db.run(sql);
		}
		int neuron_id = stoi(a[0][0]);

		std::cout << "Added Neuron " << neuron_id << std::endl;

		///////////////////////

		std::map<int64_t, int64_t> old_to_new;
		old_to_new.insert( {-1, -1} );
		std::list<swc_line_input> lines;
		std::list<std::tuple<int, int>> edges;

		int id, parent, type;
		float x, y, z, r;

		// id   type   X       Y         Z      R   parent
		// 16613 2 4341.180 5706.911 9866.430 1.000 16612
		uint32_t i = 1;
		std::string line;
		std::istringstream infile(msg.parts[part_id].body);
		while (std::getline(infile, line))
		{
			if (line.length() <= 1)
				continue;
			std::istringstream iss(line);

			iss >> id;
			iss >> type;
			iss >> x;
			iss >> y;
			iss >> z;
			iss >> r;
			iss >> parent;

			lines.push_back({type, x, y, z, r, parent});
			old_to_new.insert({id, i});

			i++;
		}

		std::stringstream sql_builder("");

		sql_builder << "INSERT INTO SWC"
					<< "(I,NEURONID,PARENTID,X,Y,Z,R,T,USERID)"
					<< " VALUES ";

		i = 1;
		for (swc_line_input a : lines)
		{
			const auto [type, x, y, z, r, parent] = a;
			int64_t parent_new = old_to_new[parent];
			// std::cout << x << '\t' << y << '\t' << z << '\t' << r << '\t' << parent_new << std::endl;

			int user = -1;
			sql_builder << "(" << i << ","
						<< neuron_id << ","
						<< parent_new << ","
						<< x << ',' << y << ',' << z << ',' << r << ',' << type << ',' << user
						<< "),";
			i++;
		}

		sql_builder.seekp(-1,sql_builder.cur);
		sql_builder << ";";

		{
			auto db = database_interface(trace_file[0]);
			db.run(sql_builder.str());
		}

		json response = {
			{"neuronid", neuron_id}
		};

		return crow::response( response.dump() ); });

	CROW_ROUTE(app, "/<string>/skeleton_api/upload/<int>").methods("POST"_method)([](const crow::request &req, std::string data_id, int neuron_id)
																				  {
		if(READ_ONLY_MODE) {
			return crow::response(crow::status::BAD_REQUEST);
		}

		data_id = str_first(data_id, '+');

		crow::multipart::message msg(req);

		int part_id = -1;
		for(int i = 0; i < msg.parts.size(); i++) {
			for (auto& item_h : msg.parts[i].headers)
			{
				for (auto& it : item_h.second.params)
				{
					if(it.first == "name" && it.second == "data") { // Identify the part with the data
						part_id = i;
					}
				}
			}
		}

		if(part_id == -1) {
			std::cerr << "Failed to find file header." << std::endl;
			return crow::response(crow::status::BAD_REQUEST);
		}

		std::vector<std::string> trace_file = glob_tool(DATA_PATH + data_id + "/traces.sql");

		if(trace_file.size() == 0) {
			std::cerr << "Failed to find sql file." << std::endl;
			return crow::response(crow::status::BAD_REQUEST);
		}
		
		std::cout << "Loading file: " << trace_file[0] << std::endl; 

		std::cout << "Adding Neuron " << neuron_id << std::endl;

		///////////////////////

		std::map<int64_t, int64_t> old_to_new;
		old_to_new.insert( {-1, -1} );
		std::list<swc_line_input> lines;
		std::list<std::tuple<int, int>> edges;

		int id, parent, type;
		float x, y, z, r;

		// id   type   X       Y         Z      R   parent
		// 16613 2 4341.180 5706.911 9866.430 1.000 16612
		uint32_t i = 1;
		std::string line;
		std::istringstream infile(msg.parts[part_id].body);
		while (std::getline(infile, line))
		{
			if (line.length() <= 1)
				continue;
			std::istringstream iss(line);

			iss >> id;
			iss >> type;
			iss >> x;
			iss >> y;
			iss >> z;
			iss >> r;
			iss >> parent;

			lines.push_back({type, x, y, z, r, parent});
			old_to_new.insert({id, i});

			i++;
		}

		std::stringstream sql_builder("");

		sql_builder << "INSERT INTO SWC"
					<< "(I,NEURONID,PARENTID,X,Y,Z,R,T,USERID)"
					<< " VALUES ";

		i = 1;
		for (swc_line_input a : lines)
		{
			const auto [type, x, y, z, r, parent] = a;
			int64_t parent_new = old_to_new[parent];
			// std::cout << x << '\t' << y << '\t' << z << '\t' << r << '\t' << parent_new << std::endl;

			int user = -1;
			sql_builder << "(" << i << ","
						<< neuron_id << ","
						<< parent_new << ","
						<< x << ',' << y << ',' << z << ',' << r << ',' << type << ',' << user
						<< "),";
			i++;
		}

		sql_builder.seekp(-1,sql_builder.cur);
		sql_builder << ";";

		{
			auto db = database_interface(trace_file[0]);
			db.run(sql_builder.str());
		}

		json response = {
			{"neuronid", neuron_id}
		};

		return crow::response( response.dump() ); });

	CROW_ROUTE(app, "/<string>/skeleton_api/get/<int>")
	([](crow::response &res, std::string data_id, int skeleton_id)
	 {
		data_id = str_first(data_id, '+');

		callback_vector neuron;
		{
			auto db = database_interface(DATA_PATH + data_id + "/traces.sql");
			neuron = db.get_neuron_by_id(skeleton_id);
		}

		std::stringstream out("");

		for(stringvec line : neuron) {
			for(int i = 0; i < 7; i++) {
				out << line[i] << '\t';
			}
			
			out.seekp(-1,out.cur);
			out << '\n';
		}

		res.write( out.str() );
		res.end(); });

	CROW_ROUTE(app, "/<string>/skeleton/<int>")
	([](crow::response &res, std::string data_id, int skeleton_id)
	 {
		data_id = str_first(data_id, '+');

		callback_vector neuron;
		{
			auto db = database_interface(DATA_PATH + data_id + "/traces.sql");
			neuron = db.get_neuron_by_id(skeleton_id);
		}
    	
    	std::map<uint32_t, uint32_t> old_to_new;
    	std::list<swc_line> lines;
    	std::list<std::tuple<int, int>> edges;
    	
    	int id, parent, type;
    	float x, y, z, r;
    	
    	// id   type   X       Y         Z      R   parent
    	// 16613 2 4341.180 5706.911 9866.430 1.000 16612
    	uint32_t i = 0;
		for( stringvec line : neuron ) {
			id = stoi(line[0]);
			type = stoi(line[1]);
			x = stof(line[2]);
			y = stof(line[3]);
			z = stof(line[4]);
			r = stof(line[5]);
			parent = stoi(line[6]);

			lines.push_back( {x,y,z,r,parent} );
			//lines.push_back( {z,y,x,r,parent} );
			old_to_new.insert({id, i});

			i++;
		}
    	
    	i = 0;
    	for(auto line_i = lines.begin(); line_i != lines.end(); line_i++) {
    		int old_parent = std::get<4>(*line_i);
    		if(old_parent != -1) {
    			int parent_mapped = old_to_new[old_parent];
    			edges.push_back( {i, parent_mapped} );
			}
    		i++;
    	}
    	
    	uint32_t pt_cnt = lines.size();
    	uint32_t edge_cnt = edges.size();
    	
    	res.write(std::string((char *) &pt_cnt, sizeof(uint32_t)));
    	res.write(std::string((char *) &edge_cnt, sizeof(uint32_t)));
    	
    	float out_buffer_pt[3];
    	for(auto line_i = lines.begin(); line_i != lines.end(); line_i++) {
    		out_buffer_pt[0] = std::get<0>(*line_i) * 1000; // x
    		out_buffer_pt[1] = std::get<1>(*line_i) * 1000; // y
    		out_buffer_pt[2] = std::get<2>(*line_i) * 1000; // z
    		// Still remap xyz -> yxz ??
    		
    		res.write(std::string((char *) &out_buffer_pt[0], 3*sizeof(float)));
    	}
    	
    	uint32_t out_buffer_edge[2];
    	for(auto edge_i = edges.begin(); edge_i != edges.end(); edge_i++) {
    		out_buffer_edge[0] = std::get<0>(*edge_i);
    		out_buffer_edge[1] = std::get<1>(*edge_i);
    		
    		res.write(std::string((char *) &out_buffer_edge[0], 2*sizeof(uint32_t)));
    	}
    	
    	res.end(); });

	//	ENDPOINT: /<string>/raw_access/<c>,<i>,<j>,<k>/info
	CROW_ROUTE(app, "/<string>/raw_access/<string>/info")
	([](crow::response &res, std::string data_id, std::string chunk_key)
	 {
		data_id = str_first(data_id, '+');
		auto archive_search = archive_inventory.find(data_id);

		if(archive_search == archive_inventory.end()) {
			res.end("File not found.");
			return;
		}

		unsigned int channel, chunk_i, chunk_j, chunk_k;
		sscanf(chunk_key.c_str(), "%u,%u,%u,%u", &channel, &chunk_i, &chunk_j, &chunk_k);

		archive_reader * reader = archive_search->second;

		if (channel >= reader->channel_count || chunk_i >= reader->mcountx || chunk_j >= reader->mcounty || chunk_k >= reader->mcountz)
		{
			res.code = 400;
			res.end();
			return;
		}

		std::vector<basic_json> scales;
		for(size_t scale : reader->scales) {
			std::tuple<size_t, size_t, size_t> res_scaled = reader->get_res(scale);
			std::vector<uint32_t> res = {
				(uint32_t) std::get<0>(res_scaled),
				(uint32_t) std::get<1>(res_scaled),
				(uint32_t) std::get<2>(res_scaled)
			};

			basic_json to_add;
			to_add["chunk_sizes"] = {
				//{64, 1, 1},
				//{1, 64, 1},
				//{1, 1, 64}
				{256, 256, 1  },
				{256, 1,   256},
				{1,   256, 256},
				{32, 32, 32},

			};

			to_add["encoding"] = "raw";
			to_add["key"] = std::to_string(scale);
			to_add["resolution"] = res;

			//std::tuple<size_t, size_t, size_t> sizes = reader->get_size(scale);
			//to_add["size"] = {
			//	std::get<0>(sizes),
			//	std::get<1>(sizes),
			//	std::get<2>(sizes)
			//};

			packed_reader * raw_reader = reader->get_mchunk(scale, channel, chunk_i, chunk_j, chunk_k);

			to_add["size"] = {
				raw_reader->sizex,
				raw_reader->sizey,
				raw_reader->sizez
			};

			to_add["voxel_offset"] = {0,0,0};

			scales.push_back(to_add);
		}

		json response = {
			{"type", "image"},
			{"@type", "neuroglancer_multiscale_volume"},
			{"data_type", "uint16"}, // TODO change based on dtype
			{"num_channels", reader->channel_count}
		};
		response["scales"] = scales;

		res.write(response.dump());
    	res.end(); });

	// ENDPOINT: /<data_id>/raw_access/<c>,<i>,<j>,<k>/<resolution>/<key>-<key>-<key>
	CROW_ROUTE(app, "/<string>/raw_access/<string>/<string>/<string>")
	([](crow::response &res, std::string data_id, std::string chunk_key, std::string resolution_id, std::string tile_key)
	 {
		std::vector<std::string> data_id_parts = str_split(data_id, '+');
		std::vector<std::pair<std::string, std::string>> filters;

		if(data_id_parts.size() == 2) {
				std::string filter_params = data_id_parts[1];
				std::vector<std::string> filter_keys = str_split(filter_params, '&');
				for(auto filter_key : filter_keys) {
					std::vector<std::string> filter_parsed = str_split(filter_key, '=');

					if(filter_parsed.size() == 2) {
						filters.push_back( {filter_parsed[0], filter_parsed[1]} );
					} else {
						//std::cout << "Failed to parse filter " << filter_key << std::endl;
					}
				}
		} else {
			//std::cout << "Failed to parse filterset." << std::endl;
		}

		//for (const auto& pair : filters) {
        //	std::cout << "Key: " << pair.first << ", Value: " << pair.second << std::endl;
    	//}

		data_id = data_id_parts[0];
		auto archive_search = archive_inventory.find(data_id);
		if(archive_search == archive_inventory.end()) {
			res.end();
			return;
		}

		archive_reader * reader = archive_search->second;

		unsigned int x_begin, x_end, y_begin, y_end, z_begin, z_end;
		// <xBegin>-<xEnd>_<yBegin>-<yEnd>_<zBegin>-<zEnd>
		sscanf(tile_key.c_str(), "%u-%u_%u-%u_%u-%u", &x_begin, &x_end, &y_begin, &y_end, &z_begin, &z_end);

		size_t chunk_sizes[3] = {x_end - x_begin, y_end - y_begin, z_end - z_begin};
		size_t scale = stoi(resolution_id);
		
		unsigned int channel, chunk_i, chunk_j, chunk_k;
		sscanf(chunk_key.c_str(), "%u,%u,%u,%u", &channel, &chunk_i, &chunk_j, &chunk_k);

		if (channel >= reader->channel_count || chunk_i >= reader->mcountx || chunk_j >= reader->mcounty || chunk_k >= reader->mcountz)
		{
			res.code = 400;
			res.end();
			return;
		}


		// Create the output buffer
		// The subvolume data for the chunk is stored directly in little-endian binary format in [x, y, z, channel]
		// Fortran order (i.e. consecutive x values are contiguous)
		//                          Z              Y              X             CH
		//uint16_t out_buffer[chunk_sizes[2]][chunk_sizes[1]][chunk_sizes[0]][channel_count];
		//uint16_t out_buffer[handler->channel_count][chunk_sizes[2]][chunk_sizes[1]][chunk_sizes[0]];
		const size_t out_buffer_size = sizeof(uint16_t) * chunk_sizes[0] * chunk_sizes[1] * chunk_sizes[2] * reader->channel_count;

		packed_reader * chunk_reader = reader->get_mchunk(scale, channel, chunk_i, chunk_j, chunk_k);

		uint16_t * out_buffer = (uint16_t*) malloc(out_buffer_size);

		uint16_t * chunk = nullptr;
		size_t last_sub_chunk_id = SIZE_MAX;

		std::map<size_t, uint16_t *> chunk_cache;

		const size_t sx = chunk_reader->sizex;
		const size_t sy = chunk_reader->sizey;
		const size_t sz = chunk_reader->sizez;

		const size_t mchunkx = chunk_reader->chunkx;
		const size_t mchunky = chunk_reader->chunky;
		const size_t mchunkz = chunk_reader->chunkz;

		for (size_t i = x_begin; i < x_end; i++)
		{
			// Find the start/stop coordinates of this chunk
			const size_t xmin = mchunkx * (i / mchunkx);			  // lower bound of mchunk
			const size_t xmax = std::min((size_t)xmin + mchunkx, sx); // upper bound of mchunk
			const size_t xsize = xmax - xmin;

			for (size_t j = y_begin; j < y_end; j++)
			{
				const size_t ymin = mchunky * (j / mchunky);
				const size_t ymax = std::min(ymin + mchunky, sy);
				const size_t ysize = ymax - ymin;

				for (size_t k = z_begin; k < z_end; k++)
				{
					const size_t zmin = mchunkz * (k / mchunkz);
					const size_t zmax = std::min(zmin + mchunkz, sz);
					const size_t zsize = zmax - zmin;

					const size_t sub_chunk_id = chunk_reader->find_index(i, j, k);

					if (sub_chunk_id != last_sub_chunk_id || chunk == nullptr)
					{
						chunk = chunk_cache[sub_chunk_id];

						if (chunk == 0)
						{
							chunk = chunk_reader->load_chunk(sub_chunk_id, xsize, ysize, zsize);
							chunk_cache[sub_chunk_id] = chunk;
						}

						last_sub_chunk_id = sub_chunk_id;
					}

					const size_t x_in_chunk_offset = i - xmin;
					const size_t y_in_chunk_offset = j - ymin;
					const size_t z_in_chunk_offset = k - zmin;

					// Calculate the coordinates of the input and output inside their respective buffers
					const size_t coffset = (x_in_chunk_offset * ysize * zsize) + // X
										   (y_in_chunk_offset * zsize) +		 // Y
										   (z_in_chunk_offset);					 // Z

					const size_t ooffset = ((k - z_begin) * chunk_sizes[1] * chunk_sizes[0]) + // Z
										   ((j - y_begin) * chunk_sizes[0]) +				   // Y
										   (i - x_begin);									   // X

					const uint16_t v = chunk[coffset];
					out_buffer[ooffset] = v;
				}
			}
		}

		for (auto it = chunk_cache.begin(); it != chunk_cache.end(); it++)
        {
            free(it->second);
        }

		for(const auto& pair : filters) {
			filter_run(
				out_buffer,
				out_buffer_size,
				{chunk_sizes[0], chunk_sizes[1], chunk_sizes[2]},
				reader->channel_count,
				pair.first,
				pair.second
			);
		}

		res.body = std::string((char *) out_buffer, out_buffer_size);
		free(out_buffer);

		res.end(); });

	CROW_ROUTE(app, "/<string>/zmaxprojection/<int>/<string>/<string>")
	([](crow::response &res, std::string data_id, int num_zslices, std::string resolution_id,
		std::string tile_key)
	 {
		auto begin = now();
		
		std::vector<std::string> data_id_parts = str_split(data_id, '+');
		std::vector<std::pair<std::string, std::string>> filters;
	
		if(data_id_parts.size() == 2) {
			std::string filter_params = data_id_parts[1];
			std::vector<std::string> filter_keys = str_split(filter_params, '&');
			for(auto filter_key : filter_keys) {
				std::vector<std::string> filter_parsed = str_split(filter_key, '=');
				if(filter_parsed.size() == 2) {
					filters.push_back({filter_parsed[0], filter_parsed[1]});
				}
			}
		}
	
		data_id = data_id_parts[0];
		auto archive_search = archive_inventory.find(data_id);
		if(archive_search == archive_inventory.end()) {
			res.end();
			return;
		}
	
		archive_reader* reader = archive_search->second;
	
		unsigned int x_begin, x_end, y_begin, y_end, z_begin, z_end;
		sscanf(tile_key.c_str(), "%u-%u_%u-%u_%u-%u", &x_begin, &x_end, &y_begin, &y_end, &z_begin, &z_end);
	
		size_t chunk_sizes[3] = {x_end - x_begin, y_end - y_begin, z_end - z_begin};
		size_t scale = stoi(resolution_id);
		size_t scaled_sizez = std::get<2>(reader->get_size(scale));
		size_t half_scaled_num_zslices = std::max(1, num_zslices / stoi(resolution_id) / 2);

		uint16_t* input_buffer = nullptr;
		uint16_t* projected_buffer = nullptr;

		size_t extended_z_begin = (z_begin >= half_scaled_num_zslices) ? (z_begin - half_scaled_num_zslices) : 0;
		size_t extended_z_end = (scaled_sizez - half_scaled_num_zslices >= z_end) ? (z_end + half_scaled_num_zslices) : scaled_sizez;
		
		if (half_scaled_num_zslices < 1) {
			projected_buffer = reader->load_region(
				scale,
				x_begin, x_end,
				y_begin, y_end,
				z_begin, z_end
			);
		} else {
			input_buffer = reader->load_region(
				scale,
				x_begin, x_end,
				y_begin, y_end,
				extended_z_begin, extended_z_end
			);
	
			projected_buffer = compute_z_max_projection<uint16_t>(
				input_buffer,
				chunk_sizes[0], chunk_sizes[1],
				extended_z_end - extended_z_begin,
				reader->channel_count
			);
		}
	
		if (!projected_buffer) {
			if (input_buffer) {
				free(input_buffer);
			}
			res.code = 500;
			res.end("Failed to create projection");
			return;
		}
	
		size_t projected_buffer_size = sizeof(uint16_t) * chunk_sizes[0] * chunk_sizes[1] * chunk_sizes[2] * reader->channel_count;
	
		for(const auto& pair : filters) {
			filter_run(
				projected_buffer,
				projected_buffer_size,
				{chunk_sizes[0], chunk_sizes[1], chunk_sizes[2]},
				reader->channel_count,
				pair.first,
				pair.second
			);
		}
	
		res.body = std::string((char*)projected_buffer, projected_buffer_size);
	
		if (input_buffer) {
			free(input_buffer);
		}
		free(projected_buffer);
	
		res.end();
		
		log_time(data_id, "READ_ZMAX", scale, x_end-x_begin, y_end-y_begin, z_end-z_begin, begin); });
	
	// Top-level info route with mesh reference
	CROW_ROUTE(app, "/<string>/mesh/info")
	([](crow::response &res, std::string data_id)
	{
		data_id = str_first(data_id, '+');

		std::string info_path = DATA_PATH + data_id + "/info";
		std::ifstream info_file(info_path);
		if (!info_file) {
			res.code = 404;
			res.end("Info file not found");
			return;
		}
		
		try {
			json info_json = json::parse(info_file);
			
			res.write(info_json.dump());
			res.end();
			return;
		} catch (const std::exception& e) {
			res.code = 500;
			res.end("Error parsing info file: " + std::string(e.what()));
		} });

	// Info within mesh subdirectory
	CROW_ROUTE(app, "/<string>/mesh/mesh/info")
	([](crow::response &res, std::string data_id)
	{
		data_id = str_first(data_id, '+');
		
		std::string mesh_info_path = DATA_PATH + data_id + "/mesh/info";
		std::ifstream mesh_info_file(mesh_info_path);
		if (mesh_info_file) {
			try {
				json mesh_info_json = json::parse(mesh_info_file);
				res.write(mesh_info_json.dump());
				res.end();
				return;
			} catch (const std::exception& e) {
				std::cerr << "Error parsing mesh info file: " << e.what() << std::endl;
			}
		}
		
		// If file doesn't exist or can't be parsed, create a default response
		json response = {
			{"@type", "neuroglancer_legacy_mesh"},
			{"segment_properties", "../segment_properties"}
		};
		res.write(response.dump());
		res.end(); });

	// Segment properties for meshes
	CROW_ROUTE(app, "/<string>/mesh/mesh/segment_properties")
	([](crow::response &res, std::string data_id)
	{
		data_id = str_first(data_id, '+');
		
		std::string properties_path = DATA_PATH + data_id + "/segment_properties";
		std::ifstream properties_file(properties_path);
		if (properties_file) {
			try {
				json properties_json = json::parse(properties_file);
				res.write(properties_json.dump());
				res.end();
				return;
			} catch (const std::exception& e) {
				std::cerr << "Error parsing segment properties file: " << e.what() << std::endl;
			}
		}
		
		std::vector<basic_json> ids = {};
		std::vector<basic_json> values = {};
		
		std::vector<std::string> mesh_files = glob_tool(DATA_PATH + data_id + "/mesh/*:0:0");
		
		for(const auto& mesh_file : mesh_files) {
			size_t last_slash = mesh_file.find_last_of('/');
			std::string filename = mesh_file.substr(last_slash + 1);
			std::string segment_id = filename.substr(0, filename.find(":"));
			
			ids.push_back(segment_id);
			values.push_back(segment_id);
		}
		
		json id_prop;
		id_prop["id"] = "label";
		id_prop["type"] = "label";
		id_prop["values"] = values;
		
		json toadd_inline;
		toadd_inline["ids"] = ids;
		std::vector<json> properties;
		properties.push_back(id_prop);
		toadd_inline["properties"] = properties;
		
		json response;
		response["@type"] = "neuroglancer_segment_properties";
		response["inline"] = toadd_inline;
		
		res.write(response.dump());
		res.end(); });

	// Retrieve mesh binary data or metadata
	CROW_ROUTE(app, "/<string>/mesh/mesh/<string>")
	([](crow::response &res, std::string data_id, std::string segment_meta)
	{
		data_id = str_first(data_id, '+');
		
		if (segment_meta.find(":0:0") != std::string::npos) { // Binary mesh data file
			std::string mesh_path = DATA_PATH + data_id + "/mesh/" + segment_meta;
			
			std::ifstream mesh_file(mesh_path, std::ios::binary);
			if (!mesh_file) {
				res.code = 404;
				res.end("Mesh file not found");
				return;
			}
			
			mesh_file.seekg(0, std::ios::end);
			size_t file_size = mesh_file.tellg();
			mesh_file.seekg(0, std::ios::beg);
			
			std::vector<char> buffer(file_size);
			mesh_file.read(buffer.data(), file_size);
			
			// Set content type to application/octet-stream
			res.set_header("Content-Type", "application/octet-stream");
			res.set_header("Content-Disposition", "attachment");
			
			res.body.assign(buffer.data(), file_size);
			res.code = 200;
		} else if (segment_meta.find(":0") != std::string::npos) {
			std::string meta_path = DATA_PATH + data_id + "/mesh/" + segment_meta;
			
			std::ifstream meta_file(meta_path);
			if (!meta_file) {
				std::string segment_id = segment_meta.substr(0, segment_meta.find(":"));
				std::string fragment_file = segment_id + ":0:0";
				
				std::string fragment_path = DATA_PATH + data_id + "/mesh/" + fragment_file;
				if (!std::ifstream(fragment_path)) {
					res.code = 404;
					res.end("Mesh fragment not found");
					return;
				}
				json metadata = {
					{"fragments", {fragment_file}}
				};
				res.write(metadata.dump());
			} else {
				std::string content((std::istreambuf_iterator<char>(meta_file)), std::istreambuf_iterator<char>());
				res.write(content);
			}
		} else {
			res.code = 404;
			res.end("Invalid mesh request format");
		}
		
		res.end(); });

	// @app.route("/data/<data_id>/<resolution>/<key>-<key>-<key>")
	// This has to be last in the route list because it acts as a wildcard
	CROW_ROUTE(app, "/<string>/<string>/<string>")
	([](crow::response &res, std::string data_id, std::string resolution_id, std::string tile_key)
	 {
		auto begin = now();
		
		std::vector<std::string> data_id_parts = str_split(data_id, '+');
		std::vector<std::pair<std::string, std::string>> filters;

		if(data_id_parts.size() == 2) {
				std::string filter_params = data_id_parts[1];
				std::vector<std::string> filter_keys = str_split(filter_params, '&');
				for(auto filter_key : filter_keys) {
					std::vector<std::string> filter_parsed = str_split(filter_key, '=');

					if(filter_parsed.size() == 2) {
						filters.push_back( {filter_parsed[0], filter_parsed[1]} );
					} else {
						//std::cout << "Failed to parse filter " << filter_key << std::endl;
					}
				}
		} else {
			//std::cout << "Failed to parse filterset." << std::endl;
		}

		//for (const auto& pair : filters) {
        //	std::cout << "Key: " << pair.first << ", Value: " << pair.second << std::endl;
    	//}

		data_id = data_id_parts[0];
		auto archive_search = archive_inventory.find(data_id);
		if(archive_search == archive_inventory.end()) {
			res.end();
			return;
		}

		archive_reader * reader = archive_search->second;

		unsigned int x_begin, x_end, y_begin, y_end, z_begin, z_end;
		// <xBegin>-<xEnd>_<yBegin>-<yEnd>_<zBegin>-<zEnd>
		sscanf(tile_key.c_str(), "%u-%u_%u-%u_%u-%u", &x_begin, &x_end, &y_begin, &y_end, &z_begin, &z_end);

		size_t chunk_sizes[3] = {x_end - x_begin, y_end - y_begin, z_end - z_begin};
		size_t scale = stoi(resolution_id);

		// Create the output buffer
		// The subvolume data for the chunk is stored directly in little-endian binary format in [x, y, z, channel]
		// Fortran order (i.e. consecutive x values are contiguous)
		//                          Z              Y              X             CH
		//uint16_t out_buffer[chunk_sizes[2]][chunk_sizes[1]][chunk_sizes[0]][channel_count];
		//uint16_t out_buffer[handler->channel_count][chunk_sizes[2]][chunk_sizes[1]][chunk_sizes[0]];
		const size_t out_buffer_size = sizeof(uint16_t) * chunk_sizes[0] * chunk_sizes[1] * chunk_sizes[2] * reader->channel_count;

		uint16_t * out_buffer = reader->load_region(
			scale,
			x_begin, x_end,
			y_begin, y_end,
			z_begin, z_end
		);

		for(const auto& pair : filters) {
			filter_run(
				out_buffer,
				out_buffer_size,
				{chunk_sizes[0], chunk_sizes[1], chunk_sizes[2]},
				reader->channel_count,
				pair.first,
				pair.second
			);
		}

		res.body = std::string((char *) out_buffer, out_buffer_size);
		free(out_buffer);

		res.end(); 
		
		log_time(data_id, "READ", scale, x_end-x_begin, y_end-y_begin, z_end-z_begin, begin); });

	{ // App start logic
		std::cout << "Using port: " << port << std::endl;
		std::cout << "Thread count: " << THREAD_COUNT << std::endl;

		app.port(port)
			//.use_compression(crow::compression::algorithm::DEFLATE)
			//.use_compression(crow::compression::algorithm::GZIP)
			.concurrency(THREAD_COUNT)
			//.multithreaded()
			.loglevel(crow::LogLevel::Warning)
			.timeout(5)
			.run();
	}
}
