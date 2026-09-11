// Experimental S20 geometry engine. Raw inputs only; no Studio reconstruction data.
// Registration, voxel maps and threshold estimation use KISS-ICP v1.3.0 (MIT).
#include <kiss_icp/core/Registration.hpp>
#include <kiss_icp/core/Threshold.hpp>
#include <kiss_icp/core/VoxelUtils.hpp>
#include <tbb/parallel_for.h>
#include <tbb/global_control.h>
#include <Eigen/Geometry>
#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <vector>
#include <sstream>
#include "observations.hpp"
#include "state_table.hpp"
#include "inertial.hpp"
#include "plane_map.hpp"
#include <memory>

using Vec=Eigen::Vector3d;
using Quat=Eigen::Quaterniond;
using Pose=Sophus::SE3d;
using Clock=std::chrono::steady_clock;
struct Point {float x,y,z;uint32_t offset;uint8_t intensity,tag,line,pad;};
static_assert(sizeof(Point)==20);
struct Imu {double time;Vec gyro,acc;Quat q;};
struct Scan {double begin,end;std::vector<Point> points;};
struct FusionCell {Vec sum=Vec::Zero();uint32_t samples=0,frames=0;size_t last_frame=~size_t(0);};
template<typename T>T read(std::istream& f){T x;f.read(reinterpret_cast<char*>(&x),sizeof(T));if(!f)throw std::runtime_error("Truncated input");return x;}
double seconds(Clock::time_point t){return std::chrono::duration<double>(Clock::now()-t).count();}
Vec readvec(std::istream& f){Vec x;for(int j=0;j<3;++j)x[j]=read<double>(f);return x;}

int main(int argc,char**argv){try{
 if(argc<3){std::cerr<<"Usage: s20_reconstruct raw.bin output_dir [threads=12] [limit=0] [voxel=.12] [output_voxel=.025] [min_frames=1] [centroid=0] [imu_shift=input] [export_range=30] [save_far_observations=0] [save_all_observations=0] [motion_model=gyro] [accel_scale_to_m_s2=1]\n";return 2;}
 const int threads=argc>3?std::stoi(argv[3]):12;
 const size_t limit=argc>4?std::stoul(argv[4]):0;
 const double voxel=argc>5?std::stod(argv[5]):.12;
 const double output_voxel=argc>6?std::stod(argv[6]):.025;
 const int min_frames=argc>7?std::stoi(argv[7]):1;
 const int centroid=argc>8?std::stoi(argv[8]):0;
 const double export_range=argc>10?std::stod(argv[10]):30.;
 const bool save_far=argc>11?std::stoi(argv[11])!=0:false;
 const bool save_all=argc>12?std::stoi(argv[12])!=0:false;
 const std::string motion_model=argc>13?argv[13]:"gyro";
 if(motion_model!="gyro"&&motion_model!="inertial_kiss"&&motion_model!="inertial_iekf")throw std::runtime_error("Unknown motion model");
 const bool use_iekf=motion_model=="inertial_iekf";
 const bool full_inertial=motion_model!="gyro";
 const double accel_scale=argc>14?std::stod(argv[14]):1.;
 if(!std::isfinite(accel_scale)||accel_scale<=0||accel_scale>100)throw std::runtime_error("Invalid acceleration scale");
 if(!std::isfinite(export_range)||export_range<30||export_range>200)throw std::runtime_error("Invalid export range");
 if(threads<1||threads>128||!std::isfinite(voxel)||voxel<=0||!std::isfinite(output_voxel)||output_voxel<=0||min_frames<1||centroid<0||centroid>1||(!centroid&&min_frames!=1))throw std::runtime_error("Invalid configuration");
 tbb::global_control concurrency(tbb::global_control::max_allowed_parallelism,threads);
 auto begin=Clock::now();std::filesystem::path output(argv[2]);
 if(std::filesystem::exists(output)&&!std::filesystem::is_empty(output))throw std::runtime_error("Reconstruction requires a fresh output directory");
 std::filesystem::create_directories(output);
 if(save_all){if(std::filesystem::exists(output/"observations"))throw std::runtime_error("Use a fresh output directory for observation capture");std::filesystem::create_directory(output/"observations");}
 std::ifstream input(argv[1],std::ios::binary);char magic[8];input.read(magic,8);
 const std::string input_format(magic,8);
 if(input_format!="S20RAW01"&&input_format!="S20RAW02")throw std::runtime_error("Wrong input format");
 const auto origin=read<uint64_t>(input);
 const double calibration_imu_shift=read<double>(input);
 const double imu_shift=argc>9&&std::string(argv[9])!="input"?std::stod(argv[9]):calibration_imu_shift;
 if(!std::isfinite(imu_shift)||std::abs(imu_shift)>.1)throw std::runtime_error("Invalid IMU shift");
 const Vec lever=readvec(input);
 Eigen::Matrix3d lidar_imu_rotation=Eigen::Matrix3d::Identity();
 if(input_format=="S20RAW02")for(int i=0;i<3;++i)for(int j=0;j<3;++j)lidar_imu_rotation(i,j)=read<double>(input);
 if(!lever.allFinite()||!lidar_imu_rotation.allFinite()||
    (lidar_imu_rotation.transpose()*lidar_imu_rotation-Eigen::Matrix3d::Identity()).norm()>1e-8||
    std::abs(lidar_imu_rotation.determinant()-1)>1e-8)throw std::runtime_error("Invalid LiDAR/IMU extrinsics");
 const bool rotated_calibration=(lidar_imu_rotation-Eigen::Matrix3d::Identity()).squaredNorm()!=0;
 const Quat lidar_imu_q(lidar_imu_rotation);
 const auto ni=read<uint32_t>(input),nf=read<uint32_t>(input);
 if(ni<100||ni>10000000||nf>1000000)throw std::runtime_error("Invalid input counts");
 std::vector<Imu> imu;imu.reserve(ni);Vec bias=Vec::Zero(),up=Vec::Zero();int stationary=0;
 for(uint32_t i=0;i<ni;++i){
  Imu m;m.time=read<double>(input)+imu_shift;m.gyro=readvec(input);m.acc=readvec(input);
  if(full_inertial)m.acc*=accel_scale;
  if(!std::isfinite(m.time)||!m.gyro.allFinite()||!m.acc.allFinite())throw std::runtime_error("Invalid IMU sample");
  if(i&&m.time<=imu.back().time)throw std::runtime_error("Nonmonotonic IMU time");
  if(m.time>=0&&m.time<2){bias+=m.gyro;up+=m.acc;++stationary;}
  imu.push_back(m);
 }
 if(stationary<100)throw std::runtime_error("Insufficient initialization samples");
 bias/=stationary;
 const double initial_gravity=(up/double(stationary)).norm();
 if(full_inertial&&(initial_gravity<8||initial_gravity>12))throw std::runtime_error("Invalid stationary gravity magnitude: declare acceleration units with accel_scale_to_m_s2");
 up.normalize();Vec variance=Vec::Zero();
 for(auto&m:imu)if(m.time>=0&&m.time<2)variance+=(m.gyro-bias).array().square().matrix();
 if((variance/stationary).maxCoeff()>.08*.08)throw std::runtime_error("Initialization not stationary");
 imu[0].q=Quat::Identity();
 for(size_t i=1;i<imu.size();++i){
  double dt=imu[i].time-imu[i-1].time;
  if(dt>.1)throw std::runtime_error("IMU gap exceeds 0.1s");
  Vec w=.5*(imu[i].gyro+imu[i-1].gyro)-bias;
  imu[i].q=(imu[i-1].q*Sophus::SO3d::exp(w*dt).unit_quaternion()).normalized();
 }
 std::vector<inertial::Sample> inertial_samples;
 if(full_inertial)for(const auto&m:imu)inertial_samples.push_back({m.time,m.gyro,m.acc});
 auto attitude=[&](double time){
  if(time<=imu.front().time)return imu.front().q;
  if(time>=imu.back().time)return imu.back().q;
  auto hi=std::upper_bound(imu.begin(),imu.end(),time,[](double t,const Imu&m){return t<m.time;});
  const auto&lo=*(hi-1);return lo.q.slerp((time-lo.time)/(hi->time-lo.time),hi->q);
 };
 std::vector<Scan> scans;scans.reserve(nf);
 for(uint32_t i=0;i<nf;++i){
  Scan s;s.begin=read<double>(input);s.end=read<double>(input);auto n=read<uint32_t>(input);
  if(n>1000000||!std::isfinite(s.begin)||!std::isfinite(s.end)||s.end<s.begin||(i&&s.begin<scans.back().end))throw std::runtime_error("Invalid scan");
  s.points.resize(n);input.read(reinterpret_cast<char*>(s.points.data()),n*sizeof(Point));
  if(!input)throw std::runtime_error("Truncated scan");
  for(size_t j=0;j<s.points.size();++j)if((j&&s.points[j].offset<s.points[j-1].offset)||s.points[j].offset*1e-9>s.end-s.begin+1e-6)throw std::runtime_error("Invalid per-point acquisition time");
  scans.push_back(std::move(s));
 }
 if(input.peek()!=std::char_traits<char>::eof())throw std::runtime_error("Trailing raw input data");
 const double load_seconds=seconds(begin);
 kiss_icp::Registration registration(100,1e-4,threads);
 kiss_icp::VoxelHashMap local(voxel,30.,20),global(output_voxel,1e9,1);
 tsl::robin_map<kiss_icp::Voxel,FusionCell> fused;
 kiss_icp::AdaptiveThreshold threshold(.4,.03,30.);
 Pose last(Quat::FromTwoVectors(up,Vec::UnitZ()),Vec::Zero());
 if(rotated_calibration)last=Pose(last.unit_quaternion()*lidar_imu_q,Vec::Zero());
 Vec velocity=Vec::Zero();double prev_time=-1;size_t processed=0,missing_imu=0;uint64_t valid_count=0,far_count=0;
 std::ofstream trajectory(output/"trajectory.txt"),metrics(output/"frames.csv");
 std::ofstream observations;if(save_far)observations.open(output/"far-observations.bin",std::ios::binary);
 trajectory<<"# seconds_from_raw_origin x y z qx qy qz qw\n"<<std::setprecision(12);
 metrics<<"frame,time,input_points,valid_points,registration_points,seconds,correction_m,correction_deg,sigma\n"<<std::setprecision(10);
 state_table::Table state_output;state_output.origin=origin;state_output.source="gyro_kiss_icp";
 // Native experimental assumptions, separate from unknown Studio noise units.
 const inertial::Noise inertial_noise{.001,.01,.0001,.001};
 inertial::C anchor_covariance=inertial::C::Zero();
 const double anchor_std[6]={.01,.01,.1,.001,.1,.05};
 for(int k=0;k<6;++k)anchor_covariance.block<3,3>(3*k,3*k).diagonal().setConstant(anchor_std[k]*anchor_std[k]);
 inertial::State anchor;bool have_anchor=false;
 plane_map::Map planes;
 size_t iekf_accepted=0,iekf_fallback=0;
 std::ofstream iekf_metrics,posterior_output;
 if(use_iekf){
  iekf_metrics.open(output/"iekf-updates.csv");
  iekf_metrics<<std::setprecision(17)<<"frame,time,accepted,reason,matches,no_plane,backface,grazing,residual,pose_rank,iterations,converged,initial_rms_m,final_rms_m,plane_cells,covariance_reset\n";
  posterior_output.open(output/"inertial-posterior-covariance.csv");
  posterior_output<<std::setprecision(17)<<"frame,time";
  for(int r=0;r<18;++r)for(int c=0;c<18;++c)posterior_output<<",P"<<r<<"_"<<c;
  posterior_output<<"\n";
 }
 state_table::Table inertial_output;inertial_output.origin=origin;inertial_output.pose_frame="imu_to_world";
 inertial_output.source="inertial_kiss_assumed_zero_accel_bias_fixed_gravity";
 std::ofstream covariance_output;
 if(full_inertial){
  state_output.source=use_iekf?"inertial_iekf_with_explicit_kiss_fallback":"inertial_deskew_kiss_icp";
  if(use_iekf)inertial_output.source="iterated_plane_MAP_with_explicit_kiss_fallback";
  covariance_output.open(output/"inertial-prediction-covariance.csv");
  covariance_output<<std::setprecision(17)<<"frame,time";
  for(int r=0;r<18;++r)for(int c=0;c<18;++c)covariance_output<<",P"<<r<<"_"<<c;
  covariance_output<<"\n";
 }
 auto tracking_start=Clock::now();
 for(size_t i=0;i<scans.size();++i){
  if(limit&&processed>=limit)break;const auto&s=scans[i];
  if(s.begin<imu.front().time||s.end>imu.back().time){++missing_imu;continue;}
  auto tick=Clock::now();Pose guess=last;
  if(!full_inertial&&prev_time>=0){
   Quat dq=attitude(prev_time).conjugate()*attitude(s.end);
   if(rotated_calibration)dq=lidar_imu_q.conjugate()*dq*lidar_imu_q;
   guess=Pose(last.unit_quaternion()*dq,last.translation()+velocity*(s.end-prev_time));
  }
  std::unique_ptr<inertial::History> history;
  if(full_inertial){
   if(!have_anchor){
    anchor.time=s.begin;anchor.rotation=last.unit_quaternion()*lidar_imu_q.conjugate();
    anchor.position=last.translation()-anchor.rotation*lever;
    anchor.gyro_bias=bias;anchor.gravity=Vec(0,0,-initial_gravity);
    anchor.covariance=anchor_covariance;
    have_anchor=true;
   }
   // KISS supplies no posterior covariance. Start each short propagation from
   // this explicitly assumed prior; the exported matrix is prediction-only.
   if(!use_iekf)anchor.covariance=anchor_covariance;
   history=std::make_unique<inertial::History>(inertial_samples,anchor,s.end,inertial_noise);
   const auto&predicted=history->end();
   guess=Pose(predicted.rotation*lidar_imu_q,predicted.position+predicted.rotation*lever);
  }
  std::vector<Vec> points;std::vector<double> times;std::vector<uint8_t> near;std::vector<float> ranges;points.reserve(s.points.size());times.reserve(s.points.size());
  std::vector<uint32_t> sample_ids;
  for(size_t sample=0;sample<s.points.size();++sample){const auto&p=s.points[sample];
   Vec q(p.x,p.y,p.z);double r2=q.squaredNorm();
   if(!q.allFinite()||r2<.25||r2>export_range*export_range||(p.tag&0x30)!=0)continue;
   if(save_all)sample_ids.push_back(uint32_t(sample));
   points.push_back(q);times.push_back(s.begin+double(p.offset)*1e-9);near.push_back(r2<=900);ranges.push_back(std::sqrt(r2));far_count+=r2>900;
  }
  const Quat end_inv=attitude(s.end).conjugate();
  const Vec local_velocity=guess.so3().inverse()*velocity;
  tbb::parallel_for(size_t(0),points.size(),[&](size_t j){
   if(full_inertial){
    // Integer scan-end and (scan-begin + offset) can round to adjacent doubles.
    // Permit only arithmetic roundoff, never an actual uncovered acquisition.
    double time=times[j];
    const double slack=8*std::numeric_limits<double>::epsilon()*std::max(1.,std::abs(s.end));
    if(time>s.end&&time-s.end<=slack)time=s.end;
    points[j]=history->deskew(points[j],time,lidar_imu_rotation,lever);return;
   }
   const Quat rel=end_inv*attitude(times[j]);
   if(rotated_calibration)points[j]=lidar_imu_rotation.transpose()*(rel*(lidar_imu_rotation*points[j]+lever)-lever)+(times[j]-s.end)*local_velocity;
   else points[j]=rel*(points[j]+lever)-lever+(times[j]-s.end)*local_velocity;
  });
  // Mapping coverage must not be tied to the local tracking horizon. Preserve
  // the original registration samples/order so distant returns cannot steer it.
  std::vector<Vec> tracking_points;
  if(export_range>30){tracking_points.reserve(points.size());for(size_t j=0;j<points.size();++j)if(near[j])tracking_points.push_back(points[j]);}
  auto down=kiss_icp::VoxelDownsample(export_range>30?tracking_points:points,voxel*.5);
  auto source=kiss_icp::VoxelDownsample(down,voxel*1.5);
  const double sigma=threshold.ComputeThreshold();
  Pose current=guess;
  iekf::Result update;plane_map::Stats plane_stats;
  if(use_iekf){
   auto factors=planes.match(source,guess,plane_stats);
   update=iekf::update(history->end(),factors,lidar_imu_rotation,lever);
   if(update.accepted){
    current=Pose(update.state.rotation*lidar_imu_q,update.state.position+update.state.rotation*lever);++iekf_accepted;
   }else{if(processed)current=registration.AlignPointsToMap(source,local,guess,3*sigma,sigma);++iekf_fallback;}
  }else if(processed)current=registration.AlignPointsToMap(source,local,guess,3*sigma,sigma);
  const Pose correction=guess.inverse()*current;
  threshold.UpdateModelDeviation(correction);local.Update(down,current);
  if(prev_time>=0)velocity=.7*(current.translation()-last.translation())/(s.end-prev_time)+.3*velocity;
  if(full_inertial){
   const auto&predicted=history->end();
   covariance_output<<i<<","<<s.end;
   for(int r=0;r<18;++r)for(int c=0;c<18;++c)covariance_output<<","<<predicted.covariance(r,c);
   covariance_output<<"\n";
   const Vec previous_position=anchor.position,previous_velocity=anchor.velocity;
   if(use_iekf&&update.accepted)anchor=update.state;
   else {
    anchor=predicted;anchor.rotation=current.unit_quaternion()*lidar_imu_q.conjugate();
    anchor.position=current.translation()-anchor.rotation*lever;
    // KISS fallback has no posterior; explicitly reset its assumed covariance.
    if(prev_time>=0)anchor.velocity=.7*(anchor.position-previous_position)/(s.end-prev_time)+.3*previous_velocity;
    else anchor.velocity.setZero();
    if(use_iekf)anchor.covariance=anchor_covariance;
   }
   if(use_iekf){
    const auto&st=update.stats;
    iekf_metrics<<i<<","<<s.end<<","<<update.accepted<<","<<st.reason<<","<<plane_stats.matches<<","<<plane_stats.no_plane<<","<<plane_stats.backface<<","<<plane_stats.grazing<<","<<plane_stats.residual<<","<<st.pose_rank<<","<<st.iterations<<","<<st.converged<<","<<st.initial_rms<<","<<st.final_rms<<","<<planes.size()<<","<<!update.accepted<<"\n";
    posterior_output<<i<<","<<s.end;for(int r=0;r<18;++r)for(int c=0;c<18;++c)posterior_output<<","<<anchor.covariance(r,c);posterior_output<<"\n";
   }
   state_table::State row;row.frame=uint32_t(i);row.time=s.end;
   row.pose=Pose(anchor.rotation,anchor.position);row.velocity=anchor.velocity;
   row.gyro_bias=anchor.gyro_bias;row.gravity=anchor.gravity;
   row.known=state_table::Velocity|state_table::GyroBias|state_table::Gravity;
   if(use_iekf&&iekf_accepted){row.accel_bias=anchor.accel_bias;row.known|=state_table::AccelBias;}
   inertial_output.states.push_back(row);
  }
  if(use_iekf)planes.add(down,current,i);
  auto dense=kiss_icp::VoxelDownsample(points,output_voxel);
  if(save_all){
   std::vector<observations::Record> records(points.size());
   tsl::robin_map<kiss_icp::Voxel,uint32_t> first;first.reserve(points.size());
   for(size_t j=0;j<points.size();++j){auto&r=records[j];const auto&raw=s.points[sample_ids[j]];
    for(int k=0;k<3;++k)r.xyz[k]=points[j][k];r.raw[0]=raw.x;r.raw[1]=raw.y;r.raw[2]=raw.z;r.range=ranges[j];
    r.sample=sample_ids[j];r.offset_ns=raw.offset;r.intensity=raw.intensity;r.tag=raw.tag;r.line=raw.line;
    first.try_emplace(kiss_icp::PointToVoxel(points[j],output_voxel),uint32_t(j));
   }
   for(size_t rank=0;rank<dense.size();++rank)records[first.at(kiss_icp::PointToVoxel(dense[rank],output_voxel))].export_rank=uint32_t(rank);
   observations::Header h;h.origin=origin;h.frame=uint32_t(i);h.raw_count=uint32_t(s.points.size());h.count=uint32_t(points.size());h.export_count=uint32_t(dense.size());h.begin=s.begin;h.end=s.end;h.pose=current;h.local_velocity=local_velocity;h.imu_shift=imu_shift;
   std::ostringstream name;name<<std::setw(6)<<std::setfill('0')<<i<<".s20obs";observations::write(output/"observations"/name.str(),h,records);
  }
  // Optional diagnostic records: XYZ float32, raw frame uint32, range float32,
  // relative point time float32. These observations never feed tracking/fusion.
  if(save_far)for(size_t j=0;j<points.size();++j)if(ranges[j]>=10){
   Eigen::Vector3f p=(current*points[j]).cast<float>();uint32_t frame=uint32_t(i);float time=float(times[j]);
   observations.write(reinterpret_cast<char*>(p.data()),12);observations.write(reinterpret_cast<char*>(&frame),4);
   observations.write(reinterpret_cast<char*>(&ranges[j]),4);observations.write(reinterpret_cast<char*>(&time),4);
  }
  tbb::parallel_for(size_t(0),dense.size(),[&](size_t j){dense[j]=current*dense[j];});
  if(centroid){
   for(const auto&p:dense){
    auto&cell=fused[kiss_icp::PointToVoxel(p,output_voxel)];
    cell.sum+=p;++cell.samples;
    if(cell.last_frame!=i){++cell.frames;cell.last_frame=i;}
   }
  }else global.AddPoints(dense);
  const auto&t=current.translation();const auto&q=current.unit_quaternion();
  trajectory<<s.end<<" "<<t.x()<<" "<<t.y()<<" "<<t.z()<<" "<<q.x()<<" "<<q.y()<<" "<<q.z()<<" "<<q.w()<<"\n";
  metrics<<i<<","<<s.end<<","<<s.points.size()<<","<<points.size()<<","<<source.size()<<","<<seconds(tick)<<","<<correction.translation().norm()<<","<<correction.so3().log().norm()*180/M_PI<<","<<sigma<<"\n";
  state_table::State state;state.frame=uint32_t(i);state.time=s.end;state.pose=current;
  state.known=state_table::Velocity|state_table::GyroBias;state.velocity=velocity;state.gyro_bias=bias;
  if(use_iekf)state.gyro_bias=anchor.gyro_bias;
  state_output.states.push_back(state);
  last=current;prev_time=s.end;valid_count+=points.size();++processed;
  if(processed%50==0)std::cout<<processed<<"/"<<nf<<" frames; "<<seconds(tracking_start)<<"s; position "<<t.transpose()<<std::endl;
 }
 state_table::write(output/"states.s20state",state_output);
 if(full_inertial){
  state_table::write(output/"inertial-states.s20state",inertial_output);
  covariance_output.close();if(!covariance_output)throw std::runtime_error("Covariance write failed");
  std::ofstream assumptions(output/"inertial-model.json");
  assumptions<<std::setprecision(17)<<"{\n  \"schema\": 1,\n  \"estimator\": \""<<(use_iekf?"fixed_prior_iterated_point_to_plane_MAP_with_KISS_fallback":"scan_anchored_inertial_prediction_with_KISS_pose_and_velocity_heuristic")<<"\",\n"
   <<"  \"accel_bias\": \""<<(use_iekf?"joint_state_estimate_from_propagated_cross_covariance; starts_at_zero":"unestimated_zero_assumption; availability flag remains clear")<<"\",\n"
   <<"  \"accel_scale_to_m_s2\": "<<accel_scale<<",\n"
   <<"  \"gravity_m_s2\": "<<initial_gravity<<",\n  \"gravity_source\": \""<<(use_iekf?"initialized_from_first_two_seconds_then_jointly_estimated":"fixed from mean first-two-second acceleration; not jointly estimated")<<"\",\n"
   <<"  \"noise_amplitude_densities\": {\"gyro_rad_s_sqrt_Hz\":0.001,\"accel_m_s2_sqrt_Hz\":0.01,\"gyro_bias_rad_s_per_sqrt_s\":0.0001,\"accel_bias_m_s2_per_sqrt_s\":0.001},\n"
   <<"  \"anchor_prior_std_per_axis\": [0.01,0.01,0.1,0.001,0.1,0.05],\n"
   <<"  \"covariance_order\": \"right_attitude_rad,position_m,velocity_m_s,gyro_bias_rad_s,accel_bias_m_s2,gravity_m_s2\",\n"
   <<"  \"covariance_scope\": \""<<(use_iekf?"prediction and local Gauss-Newton posterior carried between accepted updates; declared diagonal reset on explicit KISS fallback; uncalibrated and ignores shared-map/deskew correlations":"scan-end prediction before KISS correction; reset to declared assumed diagonal prior at every corrected anchor; NOT a posterior or calibrated uncertainty")<<"\",\n"
   <<"  \"studio_noise_reproduced\": false,\n  \"tightly_coupled\": "<<(use_iekf?"true":"false")<<",\n"
   <<"  \"iekf_accepted_frames\": "<<iekf_accepted<<",\n  \"iekf_kiss_fallback_frames\": "<<iekf_fallback<<",\n"
   <<"  \"plane_map_model\": \"native 0.4m world-floor Welford scatter; >=12 points from >=2 frames; second eigenvalue >=0.0004m2; smallest <=0.0004m2 and <=0.1*second; normal faces mean contributing sensor; signed incidence <=75deg; |residual|<=0.15m; variance=0.03m squared plus smallest eigenvalue; fixed prior-scan correspondences\"\n}\n";
  assumptions.close();if(!assumptions)throw std::runtime_error("Inertial model write failed");
  if(use_iekf){iekf_metrics.close();posterior_output.close();if(!iekf_metrics||!posterior_output)throw std::runtime_error("IEKF diagnostics write failed");}
 }
 const double tracking_seconds=seconds(tracking_start);auto cloud=global.Pointcloud();
 size_t rejected_voxels=0;
 if(centroid){
  cloud.reserve(fused.size());
  for(const auto&item:fused){
   const auto&cell=item.second;
   if(cell.frames>=uint32_t(min_frames))cloud.push_back(cell.sum/double(cell.samples));
   else ++rejected_voxels;
  }
 }
 std::ofstream ply(output/"reconstructed.ply",std::ios::binary);
 ply<<"ply\nformat binary_little_endian 1.0\ncomment Native S20 prototype; raw LiDAR/IMU only\nelement vertex "<<cloud.size()<<"\nproperty float x\nproperty float y\nproperty float z\nend_header\n";
 for(const auto&p:cloud){Eigen::Vector3f q=p.cast<float>();ply.write(reinterpret_cast<const char*>(q.data()),12);}ply.close();
 std::ofstream report(output/"run.json");
 report<<std::setprecision(12)<<"{\n  \"engine\": \"C++20 ARM64 configurable IMU-deskew + KISS-ICP 1.3.0\",\n"
       <<"  \"motion_model\": \""<<motion_model<<"\",\n"
       <<"  \"input_format\": \""<<input_format<<"\",\n  \"nonidentity_lidar_imu_rotation\": "<<(rotated_calibration?"true":"false")<<",\n  \"skipped_missing_imu_frames\": "<<missing_imu<<",\n"
       <<"  \"uses_studio_data\": false,\n  \"loop_closure\": false,\n  \"colorization\": false,\n  \"dynamic_removal\": false,\n"
       <<"  \"all_observations_saved\": "<<(save_all?"true":"false")<<",\n  \"time_origin_ns\": "<<origin<<",\n  \"imu_time_shift_seconds\": "<<imu_shift<<",\n"
       <<"  \"calibration_imu_time_shift_seconds\": "<<calibration_imu_shift<<",\n  \"export_max_range_m\": "<<export_range<<",\n  \"tracking_max_range_m\":30,\n  \"export_returns_beyond_tracking_range\":"<<far_count<<",\n"
       <<"  \"gyro_bias\": ["<<bias.x()<<","<<bias.y()<<","<<bias.z()<<"],\n"
       <<"  \"threads\": "<<threads<<",\n  \"registration_voxel_m\": "<<voxel<<",\n  \"output_voxel_m\": "<<output_voxel<<",\n"
       <<"  \"centroid_fusion\": "<<(centroid?"true":"false")<<",\n  \"min_observation_frames\": "<<min_frames<<",\n  \"rejected_voxels\": "<<rejected_voxels<<",\n"
       <<"  \"frames\": "<<processed<<",\n  \"valid_points\": "<<valid_count<<",\n  \"output_points\": "<<cloud.size()<<",\n"
       <<"  \"input_load_seconds\": "<<load_seconds<<",\n  \"tracking_and_map_seconds\": "<<tracking_seconds<<",\n  \"total_seconds\": "<<seconds(begin)<<"\n}\n";
 std::cout<<"Saved "<<cloud.size()<<" points in "<<seconds(begin)<<" seconds\n";
 return 0;
}catch(const std::exception&e){std::cerr<<"Error: "<<e.what()<<"\n";return 1;}}
