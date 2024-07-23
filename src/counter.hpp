#include "crow.h"

#include <chrono>
#include <map>
#include <vector>
#include <string>
#include <mutex>
#include <tuple>

#define PERF_COUNTER_ENABLE 1
#define IP_COUNTER_ENABLE 1
#define IP_USING_NGINX_FORWARD 1 // Read IP from X-Forwarded-For

std::mutex perf_counter_mutex;
std::map<std::tuple<std::string, std::string, uint16_t, uint16_t, uint16_t, uint16_t>, std::vector<u_int32_t>> perf_counters;

inline std::chrono::steady_clock::time_point now()
{
    return std::chrono::steady_clock::now();
}

inline u_int32_t calculate_dt(std::chrono::steady_clock::time_point then)
{
    std::chrono::steady_clock::time_point n = now();
    return (u_int32_t)std::chrono::duration_cast<std::chrono::microseconds>(n - then).count();
}

void log_time(std::string dset, std::string cmd_name, size_t scale, size_t xs, size_t ys, size_t zs, std::chrono::steady_clock::time_point then)
{
    if (PERF_COUNTER_ENABLE)
    {
        u_int32_t dt = calculate_dt(then);

        perf_counter_mutex.lock();
        perf_counters[{dset, cmd_name, (uint16_t)scale, (uint16_t)xs, (uint16_t)ys, (uint16_t)zs}].push_back(dt);
        perf_counter_mutex.unlock();
    }
}

// auto begin = now();
// log_time(data_id, "READ", begin);

std::map<std::string, size_t> ip_counter;

struct CounterMiddleware
{
    struct context
    {
    };

    void before_handle(crow::request &req, crow::response &res, context &ctx)
    {
        if (IP_COUNTER_ENABLE)
        {
            if (IP_USING_NGINX_FORWARD)
            {
                // Relies on a server that has "proxy_set_header X-Forwarded-For $remote_addr;" set in the proxy settings
                std::string ip = req.get_header_value("X-Forwarded-For");
                ip_counter[ip] = ip_counter[ip] + 1;
            }
            else
            {
                // Reads direct IP connected to the server, won't work for proxies
                std::string ip = req.remote_ip_address;
                ip_counter[ip] = ip_counter[ip] + 1;
            }
        }
    }

    void after_handle(crow::request &req, crow::response &res, context &ctx)
    {
        if (IP_COUNTER_ENABLE)
        {
        }
    }
};