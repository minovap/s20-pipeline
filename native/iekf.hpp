#pragma once
#include "inertial.hpp"
#include <Eigen/Cholesky>
#include <string>
#include <limits>

// Independent fixed-correspondence iterated MAP update. This is not Studio's
// recovered filter and does not assume its undocumented weights or covariance.
namespace iekf {
using E=Eigen::Matrix<double,18,1>;using H=Eigen::Matrix<double,1,18>;
using inertial::V;using inertial::C;using inertial::State;
struct PlaneObservation {V lidar_point=V::Zero(),world_mean=V::Zero(),world_normal=V::UnitZ();double variance=.0001;};
struct Options {
 int max_iterations=10;size_t min_observations=12;int min_pose_rank=6;
 double rank_relative_threshold=1e-7,huber_sigma=3,convergence_tolerance=1e-6;
 double max_rotation_step=.15,max_position_step=.3,max_rotation_correction=.5,max_position_correction=1;
 bool require_convergence=true;
};
struct Stats {std::string reason;int iterations=0,pose_rank=0;size_t observations=0;bool converged=false;double initial_cost=0,final_cost=0,initial_rms=0,final_rms=0;};
struct Result {State state;bool accepted=false;Stats stats;};
// Exp(phi+dphi) = Exp(phi) Exp(Jr(phi) dphi) to first order.
inline Eigen::Matrix3d rightJacobian(const V&phi){
 double a=phi.norm();auto S=inertial::skew(phi);double b,c;
 if(a<1e-5){double a2=a*a;b=.5-a2/24+a2*a2/720;c=1./6-a2/120+a2*a2/5040;}
 else {b=(1-std::cos(a))/(a*a);c=(a-std::sin(a))/(a*a*a);}
 return Eigen::Matrix3d::Identity()-b*S+c*S*S;
}
inline State retract(const State&prior,const E&e){
 State s=prior;s.rotation=(prior.rotation*Sophus::SO3d::exp(e.head<3>()).unit_quaternion()).normalized();
 s.position+=e.segment<3>(3);s.velocity+=e.segment<3>(6);s.gyro_bias+=e.segment<3>(9);s.accel_bias+=e.segment<3>(12);s.gravity+=e.segment<3>(15);return s;
}
// Jacobian for a right rotation increment at this state's tangent (not the
// fixed prior tangent used by the optimizer).
inline std::pair<double,H> residualJacobian(const State&s,const PlaneObservation&o,const Eigen::Matrix3d&Rli,const V&Tli){
 const V body=Rli*o.lidar_point+Tli;const Eigen::Matrix3d R=s.rotation.toRotationMatrix();
 H h=H::Zero();h.segment<3>(0)=-o.world_normal.transpose()*R*inertial::skew(body);h.segment<3>(3)=o.world_normal.transpose();
 return {o.world_normal.dot(R*body+s.position-o.world_mean),h};
}
inline double robustCost(double z,double cutoff){double a=std::abs(z);return cutoff==0||a<=cutoff?.5*z*z:cutoff*(a-.5*cutoff);}
inline Result update(const State&prior,const std::vector<PlaneObservation>&observations,const Eigen::Matrix3d&Rli,const V&Tli,const Options&o={}){
 inertial::validate(prior);Result out;out.state=prior;out.stats.observations=observations.size();
 if(!Rli.allFinite()||!Tli.allFinite()||(Rli.transpose()*Rli-Eigen::Matrix3d::Identity()).norm()>1e-8||std::abs(Rli.determinant()-1)>1e-8)throw std::runtime_error("Invalid IEKF extrinsics");
 if(o.max_iterations<1||o.min_observations<1||o.min_pose_rank<1||o.min_pose_rank>6||!std::isfinite(o.huber_sigma)||o.huber_sigma<0)throw std::runtime_error("Invalid IEKF options");
 for(double v:{o.rank_relative_threshold,o.convergence_tolerance,o.max_rotation_step,o.max_position_step,o.max_rotation_correction,o.max_position_correction})if(!std::isfinite(v)||v<=0)throw std::runtime_error("Invalid IEKF option bound");
 if(o.rank_relative_threshold>=1||o.max_rotation_correction>=3.141592653589793)throw std::runtime_error("Invalid IEKF rank or rotation bound");
 Eigen::LLT<C> priorLLT(prior.covariance);if(priorLLT.info()!=Eigen::Success)throw std::runtime_error("IEKF prior covariance must be positive definite");
 const C precision=priorLLT.solve(C::Identity());if(!precision.allFinite())throw std::runtime_error("Nonfinite IEKF prior precision");
 for(const auto&v:observations)if(!v.lidar_point.allFinite()||!v.world_mean.allFinite()||!v.world_normal.allFinite()||std::abs(v.world_normal.norm()-1)>1e-8||!std::isfinite(v.variance)||v.variance<=0||!std::isfinite(1/v.variance))throw std::runtime_error("Invalid IEKF plane observation");
 if(observations.size()<o.min_observations){out.stats.reason="insufficient_observations";return out;}
 struct Linearization {C information=C::Zero();E gradient=E::Zero();Eigen::Matrix<double,6,6> pose_information=Eigen::Matrix<double,6,6>::Zero();double cost=0,rms=0;};
 auto linearize=[&](const E&e){
  Linearization l;l.information=precision;l.gradient=precision*e;l.cost=.5*e.dot(l.gradient);const State s=retract(prior,e);C transport=C::Identity();transport.block<3,3>(0,0)=rightJacobian(e.head<3>());
  for(const auto&v:observations){const auto [r,localH]=residualJacobian(s,v,Rli,Tli);const H h=localH*transport;const double z=r/std::sqrt(v.variance);const double weight=(o.huber_sigma==0||std::abs(z)<=o.huber_sigma)?1:o.huber_sigma/std::abs(z);const double w=weight/v.variance;
   l.information.noalias()+=w*h.transpose()*h;l.gradient.noalias()+=w*h.transpose()*r;l.pose_information.noalias()+=w*localH.head<6>().transpose()*localH.head<6>();l.cost+=robustCost(z,o.huber_sigma);l.rms+=r*r;
  }l.rms=std::sqrt(l.rms/observations.size());return l;
 };
 auto rank=[&](const Linearization&l){Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double,6,6>> es(l.pose_information);if(es.info()!=Eigen::Success)return 0;double threshold=std::max(1e-12,es.eigenvalues().maxCoeff()*o.rank_relative_threshold);return int((es.eigenvalues().array()>threshold).count());};
 E e=E::Zero();auto l=linearize(e);if(!l.information.allFinite()||!l.gradient.allFinite()||!std::isfinite(l.cost)||!std::isfinite(l.rms)){out.stats.reason="nonfinite_initial_linearization";return out;}out.stats.initial_cost=l.cost;out.stats.initial_rms=l.rms;out.stats.final_cost=l.cost;out.stats.final_rms=l.rms;out.stats.pose_rank=rank(l);
 if(out.stats.pose_rank<o.min_pose_rank){out.stats.reason="degenerate_geometry";return out;}
 for(int iteration=0;iteration<o.max_iterations;++iteration){
  out.stats.iterations=iteration+1;Eigen::LLT<C> solver(l.information);if(solver.info()!=Eigen::Success){out.stats.reason="information_not_positive_definite";return out;}
  E step=-solver.solve(l.gradient);if(!step.allFinite()){out.stats.reason="nonfinite_step";return out;}
  if(step.norm()<o.convergence_tolerance){out.stats.converged=true;break;}
  // One scalar limits the step, preserving the coupled direction in all states.
  double scale=std::min({1.,o.max_rotation_step/std::max(step.head<3>().norm(),1e-30),o.max_position_step/std::max(step.segment<3>(3).norm(),1e-30)});
  bool moved=false;
  for(int attempt=0;attempt<20;++attempt){E next=e+scale*step;
   if(next.head<3>().norm()<=o.max_rotation_correction&&next.segment<3>(3).norm()<=o.max_position_correction){auto candidate=linearize(next);
    if(candidate.information.allFinite()&&candidate.gradient.allFinite()&&std::isfinite(candidate.rms)&&std::isfinite(candidate.cost)&&candidate.cost<=l.cost){e=next;l=std::move(candidate);moved=true;break;}}
   scale*=.5;
  }
  if(!moved){out.stats.reason="line_search_or_correction_bound";return out;}
  out.stats.final_cost=l.cost;out.stats.final_rms=l.rms;
 }
 // Recheck stationarity at final accepted iterate; reaching the iteration budget
 // alone never silently indicates convergence.
 {Eigen::LLT<C> solver(l.information);if(solver.info()==Eigen::Success&&solver.solve(l.gradient).norm()<o.convergence_tolerance)out.stats.converged=true;}
 out.stats.pose_rank=rank(l);if(out.stats.pose_rank<o.min_pose_rank){out.stats.reason="degenerate_final_geometry";return out;}
 if(o.require_convergence&&!out.stats.converged){out.stats.reason="iteration_limit";return out;}
 Eigen::LLT<C> posteriorLLT(l.information);if(posteriorLLT.info()!=Eigen::Success){out.stats.reason="posterior_not_positive_definite";return out;}
 C transport=C::Identity();transport.block<3,3>(0,0)=rightJacobian(e.head<3>());State final=retract(prior,e);final.covariance=transport*posteriorLLT.solve(C::Identity())*transport.transpose();final.covariance=(.5*(final.covariance+final.covariance.transpose())).eval();
 inertial::validate(final);Eigen::LLT<C> finalLLT(final.covariance);if(finalLLT.info()!=Eigen::Success){out.stats.reason="invalid_transported_covariance";return out;}
 out.state=final;out.accepted=true;out.stats.reason=out.stats.converged?"converged":"iteration_limit_accepted";return out;
}
}
