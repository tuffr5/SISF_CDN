#include <vector>
#include <string>
#include <istream>
#include <sstream>
#include <algorithm> // std::sort
#include <limits>
#include <chrono>

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

    void load_color(archive_reader *image, int window)
    {
        channel_count = image->channel_count;
        color_vector = (double *)malloc(sizeof(double) * image->channel_count);

        for (size_t c = 0; c < channel_count; c++)
        {
            color_vector[c] = 0.0;
        }

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

        // Calculate average
        color_intensity = 0;
        for (size_t i = 0; i < channel_count; i++)
        {
            for (size_t j = 0; j < 3; j++)
            {
                color_vector[i] /= chunk_sizes[j];
            }
            color_intensity += color_vector[i];
        }

        free(read_buffer);
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
