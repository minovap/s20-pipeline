#pragma once
#include <algorithm>
#include <Eigen/Geometry>
#include <sophus/se3.hpp>
#include <filesystem>
#include <fstream>
#include <array>
#include <vector>
#include <bit>
#include <cstdint>
#include <limits>
#include <stdexcept>
namespace observations {
constexpr uint32_t HeaderBytes=160,RecordBytes=56,NotExported=UINT32_MAX;
struct Record {
 double xyz[3]; // Native deskewed LiDAR coordinates at scan end; exact double values.
 float raw[3],range;
 uint32_t sample,offset_ns;
 uint8_t intensity,tag,line,reserved=0;
 uint32_t export_rank=NotExported; // Order in the legacy per-scan export downsample.
};
static_assert(sizeof(Record)==RecordBytes && std::endian::native==std::endian::little);
struct Header {
 uint64_t origin=0;uint32_t frame=0,raw_count=0,count=0,export_count=0;
 double begin=0,end=0;Sophus::SE3d pose;
 Eigen::Vector3d local_velocity=Eigen::Vector3d::Zero();double imu_shift=0;
};
template<class T> void put(std::ostream& f,const T& x){f.write(reinterpret_cast<const char*>(&x),sizeof(x));}
template<class T> T get(std::istream& f){T x{};f.read(reinterpret_cast<char*>(&x),sizeof(x));if(!f)throw std::runtime_error("Truncated observation stream");return x;}
inline void write(const std::filesystem::path& path,const Header& h,const std::vector<Record>& records){
 auto temporary=path;temporary+=".tmp";std::ofstream f(temporary,std::ios::binary);
 f.write("S20OBS01",8);put(f,HeaderBytes);put(f,RecordBytes);put(f,h.origin);
 for(auto x:{h.frame,h.raw_count,h.count,h.export_count})put(f,x);
 put(f,h.begin);put(f,h.end);for(int j=0;j<3;++j)put(f,h.pose.translation()[j]);
 const auto&q=h.pose.unit_quaternion();for(double x:{q.x(),q.y(),q.z(),q.w()})put(f,x);
 for(int j=0;j<3;++j)put(f,h.local_velocity[j]);put(f,h.imu_shift);put(f,uint64_t(0));put(f,uint64_t(0));
 f.write(reinterpret_cast<const char*>(records.data()),records.size()*sizeof(Record));f.close();if(!f)throw std::runtime_error("Observation write failed");std::filesystem::rename(temporary,path);
}
inline std::pair<Header,std::vector<Record>> read(const std::filesystem::path& path){
 std::ifstream f(path,std::ios::binary);std::array<char,8> magic{};f.read(magic.data(),8);
 if(std::string(magic.data(),8)!="S20OBS01"||get<uint32_t>(f)!=HeaderBytes||get<uint32_t>(f)!=RecordBytes)throw std::runtime_error("Unsupported observation format");
 Header h;h.origin=get<uint64_t>(f);h.frame=get<uint32_t>(f);h.raw_count=get<uint32_t>(f);h.count=get<uint32_t>(f);h.export_count=get<uint32_t>(f);h.begin=get<double>(f);h.end=get<double>(f);
 Eigen::Vector3d t;for(int j=0;j<3;++j)t[j]=get<double>(f);double qx=get<double>(f),qy=get<double>(f),qz=get<double>(f),qw=get<double>(f);Eigen::Quaterniond q(qw,qx,qy,qz);
 if(!t.allFinite()||!q.coeffs().allFinite()||std::abs(q.norm()-1)>1e-8)throw std::runtime_error("Invalid observation pose");h.pose=Sophus::SE3d(q,t);
 for(int j=0;j<3;++j)h.local_velocity[j]=get<double>(f);h.imu_shift=get<double>(f);
 const auto reserved1=get<uint64_t>(f),reserved2=get<uint64_t>(f);
 if(!h.origin||reserved1||reserved2)throw std::runtime_error("Invalid observation metadata");
 if(h.count>h.raw_count||h.raw_count>1000000||h.export_count>h.count||h.end<h.begin||!std::isfinite(h.begin)||!std::isfinite(h.end)||!h.local_velocity.allFinite()||!std::isfinite(h.imu_shift))throw std::runtime_error("Invalid observation header");
 if(std::filesystem::file_size(path)!=HeaderBytes+uint64_t(h.count)*RecordBytes)throw std::runtime_error("Observation size mismatch");
 std::vector<Record> r(h.count);f.read(reinterpret_cast<char*>(r.data()),r.size()*RecordBytes);if(!f)throw std::runtime_error("Truncated observations");
 uint32_t previous=0;std::vector<uint8_t> ranks(h.export_count,0);
 for(size_t i=0;i<r.size();++i){const auto&a=r[i];if(a.sample>=h.raw_count||(i&&a.sample<=previous)||!Eigen::Map<const Eigen::Vector3d>(a.xyz).allFinite()||!Eigen::Map<const Eigen::Vector3f>(a.raw).allFinite()||!std::isfinite(a.range)||a.range<0||a.reserved||a.offset_ns*1e-9>h.end-h.begin+1e-6)throw std::runtime_error("Invalid observation record");previous=a.sample;
  if(a.export_rank!=NotExported){if(a.export_rank>=h.export_count||ranks[a.export_rank]++)throw std::runtime_error("Invalid export rank");}
 }
 for(auto x:ranks)if(x!=1)throw std::runtime_error("Missing export rank");return {h,std::move(r)};
}
inline std::vector<std::filesystem::path> files(const std::filesystem::path& path){std::vector<std::filesystem::path> result;for(auto&e:std::filesystem::directory_iterator(path))if(e.path().extension()==".s20obs")result.push_back(e.path());std::sort(result.begin(),result.end());if(result.empty())throw std::runtime_error("No observation frames");return result;}
}
