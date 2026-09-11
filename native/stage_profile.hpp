#pragma once
#include <chrono>
#include <sys/resource.h>
#include <mach/mach.h>
#include <map>
#include <string>
#include <fstream>
#include <cstdlib>
#include <algorithm>
namespace prof {
struct Sample {double wall,cpu;unsigned long long rss;};
inline Sample now(){rusage r{};getrusage(RUSAGE_SELF,&r);mach_task_basic_info_data_t m{};mach_msg_type_number_t c=MACH_TASK_BASIC_INFO_COUNT;task_info(mach_task_self(),MACH_TASK_BASIC_INFO,(task_info_t)&m,&c);return {std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(),double(r.ru_utime.tv_sec+r.ru_stime.tv_sec)+1e-6*(r.ru_utime.tv_usec+r.ru_stime.tv_usec),m.resident_size};}
struct Entry {double wall=0,cpu=0,gpu=0;unsigned long long rss=0,metal=0,count=0;};
struct Collector {std::map<std::string,Entry> rows;~Collector(){const char*path=getenv("S20_PHASES");if(!path)return;std::ofstream f(path);f.precision(14);f<<"{\n";bool first=true;for(auto &[n,e]:rows){if(!first)f<<",\n";first=false;f<<"\""<<n<<"\":{\"wall_s\":"<<e.wall<<",\"cpu_s\":"<<e.cpu<<",\"boundary_rss_peak_bytes\":"<<e.rss<<",\"calls\":"<<e.count<<",\"gpu_command_s\":"<<e.gpu<<",\"metal_allocated_peak_bytes\":"<<e.metal<<"}";}f<<"\n}\n";}};
inline Collector data;
struct Scope {std::string name;Sample begin;Scope(std::string n):name(n),begin(now()){}~Scope(){auto end=now();auto&e=data.rows[name];e.wall+=end.wall-begin.wall;e.cpu+=end.cpu-begin.cpu;e.rss=std::max({e.rss,begin.rss,end.rss});++e.count;}};
inline void gpu(const std::string&n,double seconds,unsigned long long allocated){auto&e=data.rows[n];e.gpu+=seconds;e.metal=std::max(e.metal,allocated);}
}
