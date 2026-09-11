#pragma once
#include "observations.hpp"
#include <iomanip>
#include <sstream>
#include <unordered_map>

namespace state_table {
constexpr uint32_t Velocity=1, GyroBias=2, AccelBias=4, Gravity=8, AllKnown=15;
struct State {
 uint32_t frame=0,known=0;
 double time=0;
 Sophus::SE3d pose;
 Eigen::Vector3d velocity=Eigen::Vector3d::Zero(),gyro_bias=Eigen::Vector3d::Zero(),
  accel_bias=Eigen::Vector3d::Zero(),gravity=Eigen::Vector3d::Zero();
};
struct Table {
 uint64_t origin=0;
 std::string pose_frame="lidar_to_world",source="unspecified";
 std::vector<State> states;
};
inline void validate(const Table& t){
 if(!t.origin||(t.pose_frame!="lidar_to_world"&&t.pose_frame!="imu_to_world")||
    t.source.empty()||t.source.find_first_of(" \t\r\n")!=std::string::npos||t.states.empty())
  throw std::runtime_error("Invalid state table metadata");
 for(size_t i=0;i<t.states.size();++i){const auto&s=t.states[i];
  if((s.known&~AllKnown)||!std::isfinite(s.time)||!s.pose.translation().allFinite()||
     !s.pose.unit_quaternion().coeffs().allFinite()||std::abs(s.pose.unit_quaternion().norm()-1)>1e-8||!s.velocity.allFinite()||!s.gyro_bias.allFinite()||
     !s.accel_bias.allFinite()||!s.gravity.allFinite()||
     (i&&(s.frame<=t.states[i-1].frame||s.time<=t.states[i-1].time)))throw std::runtime_error("Invalid or unordered state row");
  // Unknown values are zero placeholders, never estimates. Availability is explicit.
  for(const auto&v:std::array<std::pair<uint32_t,Eigen::Vector3d>,4>{{{Velocity,s.velocity},{GyroBias,s.gyro_bias},{AccelBias,s.accel_bias},{Gravity,s.gravity}}})
   if(!(s.known&v.first)&&!v.second.isZero(0))throw std::runtime_error("Nonzero unavailable state field");
 }
}
inline void write(const std::filesystem::path& path,const Table&t){
 validate(t);if(std::filesystem::exists(path))throw std::runtime_error("State table already exists");
 auto temp=path;temp+=".tmp";if(std::filesystem::exists(temp))throw std::runtime_error("Incomplete state table already exists");
 std::ofstream f(temp);f<<std::setprecision(17)<<"# S20STATE 1\n# origin_ns "<<t.origin
  <<"\n# pose_frame "<<t.pose_frame<<"\n# source "<<t.source
  <<"\n# velocity_frame world\n# bias_frame imu\n# gravity_frame world\n# time_unit seconds_from_raw_origin\n"
  <<"# columns frame time known px py pz qx qy qz qw vx vy vz bgx bgy bgz bax bay baz gx gy gz\n";
 for(const auto&s:t.states){const auto&q=s.pose.unit_quaternion();
  f<<s.frame<<' '<<s.time<<' '<<s.known<<' '<<s.pose.translation().transpose()<<' '
   <<q.x()<<' '<<q.y()<<' '<<q.z()<<' '<<q.w()<<' '<<s.velocity.transpose()<<' '
   <<s.gyro_bias.transpose()<<' '<<s.accel_bias.transpose()<<' '<<s.gravity.transpose()<<'\n';
 }
 f.close();if(!f)throw std::runtime_error("State write failed");std::filesystem::rename(temp,path);
}
inline Table read(const std::filesystem::path&path){
 std::ifstream f(path);std::string line;
 if(!std::getline(f,line)||line!="# S20STATE 1")throw std::runtime_error("Unsupported state format");
 Table t;t.pose_frame.clear();t.source.clear();
 auto meta=[&](const std::string&prefix){if(!std::getline(f,line)||!line.starts_with(prefix))throw std::runtime_error("Missing state metadata");return line.substr(prefix.size());};
 {auto value=meta("# origin_ns ");size_t end=0;if(value.empty()||value[0]=='-')throw std::runtime_error("Invalid state origin");t.origin=std::stoull(value,&end);if(end!=value.size())throw std::runtime_error("Invalid state origin");}
 t.pose_frame=meta("# pose_frame ");t.source=meta("# source ");
 for(const std::string expected:{"# velocity_frame world","# bias_frame imu","# gravity_frame world","# time_unit seconds_from_raw_origin",
   "# columns frame time known px py pz qx qy qz qw vx vy vz bgx bgy bgz bax bay baz gx gy gz"})
  if(!std::getline(f,line)||line!=expected)throw std::runtime_error("Unsupported state convention");
 while(std::getline(f,line)){if(line.empty())continue;std::istringstream row(line);State s;double x,y,z,qx,qy,qz,qw;int64_t frame,known;
  if(!(row>>frame>>s.time>>known>>x>>y>>z>>qx>>qy>>qz>>qw)||frame<0||frame>UINT32_MAX||known<0||known>AllKnown)throw std::runtime_error("Malformed state row");
  s.frame=uint32_t(frame);s.known=uint32_t(known);
  Eigen::Quaterniond q(qw,qx,qy,qz);
  if(!q.coeffs().allFinite()||std::abs(q.norm()-1)>1e-8||!Eigen::Vector3d(x,y,z).allFinite())throw std::runtime_error("Invalid state pose");
  s.pose=Sophus::SE3d(q,Eigen::Vector3d(x,y,z));
  for(auto*v:{&s.velocity,&s.gyro_bias,&s.accel_bias,&s.gravity})for(int j=0;j<3;++j)if(!(row>>(*v)[j]))throw std::runtime_error("Truncated state row");
  std::string extra;if(row>>extra)throw std::runtime_error("Trailing state fields");t.states.push_back(s);
 }
 if(!f.eof())throw std::runtime_error("State read failed");validate(t);return t;
}
}
