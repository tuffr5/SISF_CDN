#include <istream>
#include <sstream>

#include <string>
#include <algorithm> // std::sort
#include <limits>
#include <chrono>
#include <vector>

#include <cstdlib>
#include <cstdint>

std::string read_env_variable(std::string name)
{
    const char *env_var = std::getenv(name.c_str());

    if (env_var == nullptr)
        return "";
    return std::string(env_var);
}

std::vector<std::string> str_split(const std::string &s, char del)
{
    std::vector<std::string> out;
    std::string t;
    std::istringstream isstream(s);

    while (std::getline(isstream, t, del))
    {
        out.push_back(t);
    }

    return out;
}

std::string str_first(const std::string &s, char del)
{
    std::string t;
    std::istringstream isstream(s);

    std::getline(isstream, t, del);

    return t;
}

std::pair<std::string, std::vector<std::pair<std::string, std::string>>> parse_filter_list(std::string data_id)
{
    std::vector<std::string> data_id_parts = str_split(data_id, '+');
    std::string data_id_out = data_id_parts[0];

    std::vector<std::pair<std::string, std::string>> filters_out;

    if (data_id_parts.size() == 2)
    {
        std::string filter_params = data_id_parts[1];
        std::vector<std::string> filter_keys = str_split(filter_params, '&');
        for (auto filter_key : filter_keys)
        {
            std::vector<std::string> filter_parsed = str_split(filter_key, '=');

            if (filter_parsed.size() == 2)
            {
                std::string filter_name = filter_parsed[0];
                std::string filter_param = filter_parsed[1];

                std::transform(filter_name.begin(), filter_name.end(), filter_name.begin(),
                               [](unsigned char c)
                               { return std::tolower(c); });

                filters_out.push_back({filter_name, filter_param});
            }
            else
            {
                // std::cout << "Failed to parse filter " << filter_key << std::endl;
            }
        }
    }
    else
    {
        // std::cout << "Failed to parse filterset." << std::endl;
    }

    // for (const auto& pair : filters) {
    //	std::cout << "Key: " << pair.first << ", Value: " << pair.second << std::endl;
    // }

    return {data_id_out, filters_out};
}

// Function to compute the mean of a vector
double mean(const std::vector<double> &data)
{
    double out = 0;
    for (double val : data)
    {
        out += val;
    }
    out /= data.size();
    return out;
}

// Function to compute the variance of a vector
double variance(const std::vector<double> &data)
{
    double mu = mean(data);
    double out = 0;
    for (double val : data)
    {
        out += (val - mu) * (val - mu);
    }
    out /= data.size();
    return out;
}

// Function to denoise a 1D signal using Weiner filter
std::vector<double> denoiseSignal(const std::vector<double> &signal, double noiseVariance)
{
    int N = signal.size();
    std::vector<double> denoisedSignal(N);

    // Weiner filter coefficients
    double a = 1.0 / (1.0 + noiseVariance);
    double b = noiseVariance / (1.0 + noiseVariance);

    // Apply Weiner filter
    denoisedSignal[0] = a * signal[0];
    for (int i = 1; i < N; ++i)
    {
        denoisedSignal[i] = a * signal[i] + b * denoisedSignal[i - 1];
    }

    return denoisedSignal;
}

float gaussian(int x, int y, int z, float sigma)
{
    return exp(-(x * x + y * y + z * z) / (2 * sigma * sigma)) / (sqrt(2 * M_PI) * sigma);
}

// CLAHE function for 16-bit images stored in a 1D array.
// image: pointer to the image pixels (modified in place)
// clipLimit: maximum allowed count in a histogram bin before clipping
void clahe_1d(uint16_t *image, size_t data_size, uint32_t clipLimit)
{
    const int bins = 256; // number of histogram bins
    const double bin_step = std::numeric_limits<uint16_t>::max() / bins;
    const size_t pixel_cnt = data_size / sizeof(uint16_t);

    std::vector<double> hist(bins, 0.0);

    for (size_t i = 0; i < pixel_cnt; i++)
    {
        const uint16_t v = image[i];

        int bin = v / bin_step;

        bin = std::min(bin, bins - 1);
        bin = std::max(0, bin);

        hist[bin] += 1.0;
    }

    // Clip the histogram: any bin count above clipLimit is reduced
    // and the excess is collected.
    uint64_t excess = 0;
    for (size_t i = 0; i < bins; i++)
    {
        if (hist[i] > clipLimit)
        {
            excess += hist[i] - clipLimit;
            hist[i] = clipLimit;
        }
    }

    // Redistribute the excess evenly among all bins.
    int redist = excess / bins;
    for (size_t i = 0; i < bins; i++)
    {
        hist[i] += redist;
    }

    // 0 .. 1
    double sum = 0.0;
    for (size_t i = 0; i < bins; i++)
    {
        hist[i] /= pixel_cnt;

        sum += hist[i];
        hist[i] = sum;
    }

    for (size_t i = 0; i < data_size / sizeof(uint16_t); i++)
    {
        // Get the original pixel value and compute its corresponding histogram bin.
        const uint16_t v = image[i];

        int bin = v / bin_step;

        bin = std::min(bin, bins - 1);
        bin = std::max(0, bin);

        image[i] = hist[bin] * std::numeric_limits<uint16_t>::max();
    }
}

// Define a structure to represent a cell in the grid
struct astar_cell
{
    int channel_count;
    int x, y, z; // Coordinates of the cell
    double g, h; // Cost values for A* algorithm
    double *color_vector, color_intensity;
    astar_cell *parent; // Pointer to the parent cell

    astar_cell(int xi, int yi, int zi, double gi, double hi, astar_cell *parenti)
    {
        color_intensity = 0;
        channel_count = 0;
        color_vector = 0;
        x = xi;
        y = yi;
        z = zi;
        g = gi;
        h = hi;
        parent = parenti;
    }

    float f() const
    {
        return g + h;
    }

    bool operator==(const astar_cell &other)
    {
        return x == other.x && y == other.y && z == other.z;
    }

    int manhattan_distance(astar_cell *other) const
    {
        int out = 0;

        out += std::abs((int)other->x - (int)x);
        out += std::abs((int)other->y - (int)y);
        out += std::abs((int)other->z - (int)z);

        return out;
    }

    double euclidean_distance(astar_cell *other) const
    {
        double out = 0;

        out += std::pow((double)other->x - (double)x, 2);
        out += std::pow((double)other->y - (double)y, 2);
        out += std::pow((double)other->z - (double)z, 2);

        return std::pow(out, 0.5);
    }

    void define_color_vector(size_t channel_count_in)
    {
        channel_count = channel_count_in;
        color_vector = (double *)malloc(sizeof(double) * channel_count);

        for (size_t c = 0; c < channel_count; c++)
        {
            color_vector[c] = 0.0;
        }
    }

    void calculate_intensity_average()
    {
        // Calculate average
        color_intensity = 0;
        for (size_t i = 0; i < channel_count; i++)
        {
            color_intensity += color_vector[i];
        }
        // color_intensity /= channel_count;
    }

    void load_color(archive_reader *image, int window)
    {
        define_color_vector(image->channel_count);

        const int xs = std::max(x - window, 0);
        const int xe = std::min(x + window, (int)image->sizex - 1);
        const int ys = std::max(y - window, 0);
        const int ye = std::min(y + window, (int)image->sizey - 1);
        const int zs = std::max(z - window, 0);
        const int ze = std::min(z + window, (int)image->sizez - 1);

        const size_t chunk_sizes[3] = {(size_t)xe - xs, (size_t)ye - ys, (size_t)ze - zs};

        uint16_t *read_buffer = image->load_region(1, xs, xe, ys, ye, zs, ze);

        // Add the image values to the color vector
        for (size_t c = 0; c < channel_count; c++)
        {
            for (size_t k = 0; k < chunk_sizes[2]; ++k)
            {
                for (size_t j = 0; j < chunk_sizes[1]; ++j)
                {
                    for (size_t i = 0; i < chunk_sizes[0]; ++i)
                    {
                        const size_t ooffset = (c * chunk_sizes[0] * chunk_sizes[1] * chunk_sizes[2]) + // C
                                               (k * chunk_sizes[0] * chunk_sizes[1]) +                  // Z
                                               (j * chunk_sizes[0]) +                                   // Y
                                               (i);                                                     // X

                        color_vector[c] += read_buffer[ooffset];
                    }
                }
            }
        }

        free(read_buffer);

        // normalize vector
        for (size_t i = 0; i < channel_count; i++)
        {
            for (size_t j = 0; j < 3; j++)
            {
                color_vector[i] /= chunk_sizes[j];
            }
        }

        calculate_intensity_average();
    }

    void delete_color_vector()
    {
        if (color_vector != 0)
        {
            free(color_vector);
        }
    }

    double color_angle(astar_cell *other)
    {
        if (channel_count == 0 || color_vector == 0 || other->channel_count != channel_count || other->color_vector == 0)
        {
            return std::numeric_limits<double>::max();
        }

        double sum = 0;
        double norma = 0;
        double normb = 0;

        for (size_t i = 0; i < channel_count; i++)
        {
            const double ca = color_vector[i];
            const double cb = other->color_vector[i];

            sum += ca * cb;
            norma += std::pow(ca, 2);
            normb += std::pow(cb, 2);
        }

        sum /= std::sqrt(norma);
        sum /= std::sqrt(normb);

        return std::acos(sum);
    }
};

bool compare_astar_cell(astar_cell *a, astar_cell *b)
{
    return a->f() > b->f();
}

const std::vector<std::tuple<int, int, int, double>>
    neighbor_steps = {
        {-1, -1, -1, 1.7320508075688772},
        {-1, -1, 0, 1.4142135623730951},
        {-1, -1, 1, 1.7320508075688772},
        {-1, 0, -1, 1.4142135623730951},
        {-1, 0, 0, 1.0},
        {-1, 0, 1, 1.4142135623730951},
        {-1, 1, -1, 1.7320508075688772},
        {-1, 1, 0, 1.4142135623730951},
        {-1, 1, 1, 1.7320508075688772},
        {0, -1, -1, 1.4142135623730951},
        {0, -1, 0, 1.0},
        {0, -1, 1, 1.4142135623730951},
        {0, 0, -1, 1.0},
        {0, 0, 1, 1.0},
        {0, 1, -1, 1.4142135623730951},
        {0, 1, 0, 1.0},
        {0, 1, 1, 1.4142135623730951},
        {1, -1, -1, 1.7320508075688772},
        {1, -1, 0, 1.4142135623730951},
        {1, -1, 1, 1.7320508075688772},
        {1, 0, -1, 1.4142135623730951},
        {1, 0, 0, 1.0},
        {1, 0, 1, 1.4142135623730951},
        {1, 1, -1, 1.7320508075688772},
        {1, 1, 0, 1.4142135623730951},
        {1, 1, 1, 1.7320508075688772}};

const std::vector<std::tuple<int, int, int, double>>
    neighbor_steps_no_z = {
        {-1, -1, 0, 1.4142135623730951},
        {-1, 0, 0, 1.0},
        {-1, 1, 0, 1.4142135623730951},
        {0, -1, 0, 1.0},
        {0, 1, 0, 1.0},
        {1, -1, 0, 1.4142135623730951},
        {1, 0, 0, 1.0},
        {1, 1, 0, 1.4142135623730951}};

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
                                                           (dk * osizei * osizej) +         // Z
                                                           (dj * osizei) +                  // Y
                                                           (di);                            // X

                                    float dist = powf32(di - i, 2) + powf32(dj - j, 2) + powf32(dk - k, 2); // d^2
                                    float coeff = exp(-0.5 * dist / powf32(sig, 2));                        // / (sig * sqrtf32(M_2_PI));

                                    float toadd = data[ioffset];
                                    toadd *= coeff;

                                    sum += toadd;
                                    n += coeff;
                                }
                            }
                        }

                        sum /= n;
                        const size_t ooffset = (c * osizei * osizej * osizek) + // C
                                               (k * osizei * osizej) +          // Z
                                               (j * osizei) +                   // Y
                                               (i);                             // X

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

    if (filter_name == "clahe")
    {
        size_t sizex = std::get<0>(data_shape);
        size_t sizey = std::get<1>(data_shape);
        size_t sizez = std::get<2>(data_shape);

        float param = std::stof(filter_param);

        clahe_1d(data, data_size, param);
    }
}