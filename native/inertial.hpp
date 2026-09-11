#pragma once
#include <Eigen/Geometry>
#include <Eigen/Eigenvalues>
#include <sophus/so3.hpp>
#include <cmath>
#include <stdexcept>
#include <vector>
#include <algorithm>

// Independent CPU reference for IMU propagation, not recovered Studio numerics.
// Right-multiplicative attitude error, order theta,p,v,bg,ba,g (18 dimensions).
namespace inertial {
using V=Eigen::Vector3d;using C=Eigen::Matrix<double,18,18>;
struct Sample {double time=0;V gyro=V::Zero(),accel=V::Zero();};
struct State {
 double time=0;Eigen::Quaterniond rotation=Eigen::Quaterniond::Identity();
 V position=V::Zero(),velocity=V::Zero(),gyro_bias=V::Zero(),accel_bias=V::Zero(),gravity=V(0,0,-9.8);
 C covariance=C::Zero();
};
// Continuous white-noise amplitude spectral densities; square to form PSD.
// gyro: rad/s/sqrt(Hz), accel: m/s^2/sqrt(Hz); bias random walks per sqrt(s).
// Values must be explicitly selected for the native model, not copied from
// Studio's ambiguously named cov_* parameters.
struct Noise {double gyro=0,accel=0,gyro_bias=0,accel_bias=0;};
inline Eigen::Matrix3d skew(const V&v){Eigen::Matrix3d m;m<<0,-v.z(),v.y(),v.z(),0,-v.x(),-v.y(),v.x(),0;return m;}
inline void validate(const State&s){
 if(!std::isfinite(s.time)||!s.rotation.coeffs().allFinite()||std::abs(s.rotation.norm()-1)>1e-8||
    !s.position.allFinite()||!s.velocity.allFinite()||!s.gyro_bias.allFinite()||!s.accel_bias.allFinite()||!s.gravity.allFinite()||!s.covariance.allFinite())
  throw std::runtime_error("Invalid inertial state");
 if((s.covariance-s.covariance.transpose()).norm()>1e-8)throw std::runtime_error("Asymmetric inertial covariance");
}
inline State propagate(const State&initial,const Sample&a,const Sample&b,const Noise&noise={},bool covariance=true){
 validate(initial);
 const double dt=b.time-a.time;
 if(!std::isfinite(dt)||dt<=0||dt>.1||std::abs(initial.time-a.time)>1e-9||!a.gyro.allFinite()||!a.accel.allFinite()||!b.gyro.allFinite()||!b.accel.allFinite())throw std::runtime_error("Invalid IMU interval or coverage");
 for(double x:{noise.gyro,noise.accel,noise.gyro_bias,noise.accel_bias})if(!std::isfinite(x)||x<0)throw std::runtime_error("Invalid IMU noise density");
 State s=initial;const V omega=.5*(a.gyro+b.gyro)-s.gyro_bias;
 const V specific_force=.5*(a.accel+b.accel)-s.accel_bias;
 const Eigen::Quaterniond half=Sophus::SO3d::exp(omega*(dt*.5)).unit_quaternion();
 const Eigen::Matrix3d Rmid=(s.rotation*half).toRotationMatrix();
 const V acceleration=Rmid*specific_force+s.gravity;
 s.position+=s.velocity*dt+.5*acceleration*dt*dt;s.velocity+=acceleration*dt;
 s.rotation=(s.rotation*Sophus::SO3d::exp(omega*dt).unit_quaternion()).normalized();s.time=b.time;
 // Kinematic-only queries retain the knot covariance; never export it as a
 // covariance at the query timestamp. Full history knots propagate covariance.
 if(!covariance)return s;
 C F=C::Zero();const Eigen::Matrix3d I=Eigen::Matrix3d::Identity();
 F.block<3,3>(0,0)=-skew(omega);F.block<3,3>(0,9)=-I;
 F.block<3,3>(3,6)=I;F.block<3,3>(6,0)=-Rmid*skew(specific_force);
 F.block<3,3>(6,12)=-Rmid;F.block<3,3>(6,15)=I;
 const C Phi=C::Identity()+F*dt+.5*F*F*dt*dt;
 Eigen::Matrix<double,18,12> G=Eigen::Matrix<double,18,12>::Zero();
 G.block<3,3>(0,0)=-I;G.block<3,3>(6,3)=-Rmid;G.block<3,3>(9,6)=I;G.block<3,3>(12,9)=I;
 Eigen::Matrix<double,12,1> density;
 density<<V::Constant(noise.gyro*noise.gyro),V::Constant(noise.accel*noise.accel),V::Constant(noise.gyro_bias*noise.gyro_bias),V::Constant(noise.accel_bias*noise.accel_bias);
 // Midpoint process-noise discretization preserves PSD and position coupling.
 const C halfPhi=C::Identity()+F*(dt*.5);
 const C Q=halfPhi*G*density.asDiagonal()*G.transpose()*halfPhi.transpose()*dt;
 s.covariance=Phi*initial.covariance*Phi.transpose()+Q;
 s.covariance=(.5*(s.covariance+s.covariance.transpose())).eval();validate(s);return s;
}
inline Sample interpolate(const Sample&a,const Sample&b,double time){
 if(!std::isfinite(time)||!std::isfinite(a.time)||!std::isfinite(b.time)||b.time<=a.time||b.time-a.time>.1||time<a.time||time>b.time||!a.gyro.allFinite()||!b.gyro.allFinite()||!a.accel.allFinite()||!b.accel.allFinite())throw std::runtime_error("Cannot extrapolate IMU or interpolate invalid interval");
 const double u=(time-a.time)/(b.time-a.time);return {time,(1-u)*a.gyro+u*b.gyro,(1-u)*a.accel+u*b.accel};
}
inline V deskewToEnd(const V&raw,const State&at_point,const State&at_end,const Eigen::Matrix3d&Rli,const V&Tli){
 validate(at_point);validate(at_end);
 if(at_point.time>at_end.time||!raw.allFinite()||!Rli.allFinite()||!Tli.allFinite()||
    (Rli.transpose()*Rli-Eigen::Matrix3d::Identity()).norm()>1e-8||std::abs(Rli.determinant()-1)>1e-8)throw std::runtime_error("Invalid deskew transform");
 const V world=at_point.rotation*(Rli*raw+Tli)+at_point.position;
 return Rli.transpose()*(at_end.rotation.conjugate()*(world-at_end.position)-Tli);
}
// Immutable, scan-bounded history anchored in an externally supplied IMU state.
// Caller owns the ordered, finite, gap-checked samples for this history's lifetime.
class History {
 const std::vector<Sample>&samples;Noise noise;
 std::vector<State> knots;
 Sample sampleAt(double time)const{
  if(time<samples.front().time||time>samples.back().time)throw std::runtime_error("History outside IMU coverage");
  auto hi=std::lower_bound(samples.begin(),samples.end(),time,[](const Sample&s,double t){return s.time<t;});
  if(hi!=samples.end()&&hi->time==time)return *hi;
  return interpolate(*(hi-1),*hi,time);
 }
 State query(double time,bool covariance)const{
  if(!std::isfinite(time)||time<knots.front().time||time>knots.back().time)throw std::runtime_error("Cannot extrapolate inertial history");
  auto hi=std::upper_bound(knots.begin(),knots.end(),time,[](double t,const State&s){return t<s.time;});
  const State&lo=*(hi-1);
  return time==lo.time?lo:propagate(lo,sampleAt(lo.time),sampleAt(time),noise,covariance);
 }
public:
 History(const std::vector<Sample>&input,const State&anchor,double end,const Noise&n):samples(input),noise(n){
  validate(anchor);
  if(input.size()<2||!std::isfinite(end)||end<anchor.time)throw std::runtime_error("Invalid inertial history interval");
  auto a=sampleAt(anchor.time);sampleAt(end);knots.push_back(anchor);
  auto hi=std::upper_bound(samples.begin(),samples.end(),anchor.time,[](double t,const Sample&s){return t<s.time;});
  while(hi!=samples.end()&&hi->time<end){knots.push_back(propagate(knots.back(),a,*hi,noise));a=*hi++;}
  if(end>knots.back().time)knots.push_back(propagate(knots.back(),a,sampleAt(end),noise));
 }
 State at(double time)const{return query(time,true);}
 const State&end()const{return knots.back();}
 V deskew(const V&raw,double time,const Eigen::Matrix3d&Rli,const V&Tli)const{
  return deskewToEnd(raw,query(time,false),knots.back(),Rli,Tli);
 }
};
}
