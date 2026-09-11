#pragma once
#include "iekf.hpp"
#include <sophus/se3.hpp>
#include <map>
#include <array>

// Native reference frontend. Fixed world-floor cells and empirical surface
// scatter, NOT Studio's adaptive octree or recovered covariance equations.
namespace plane_map {
using V=Eigen::Vector3d;using Key=std::array<int,3>;
struct Stats {size_t matches=0,no_plane=0,backface=0,grazing=0,residual=0;};
struct Cell {
 size_t count=0,frames=0,last_frame=~size_t(0);V mean=V::Zero(),view=V::Zero();
 Eigen::Matrix3d scatter=Eigen::Matrix3d::Zero();
 V normal=V::Zero();double variance=0;bool valid=false;
 void add(const V&p,const V&sensor,size_t frame){
  ++count;V delta=p-mean;mean+=delta/double(count);scatter.noalias()+=delta*(p-mean).transpose();view+=(sensor-view)/double(count);
  if(last_frame!=frame){++frames;last_frame=frame;}
 }
 void fit(){
  valid=false;if(count<12||frames<2)return;
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> es(scatter/double(count));
  if(es.info()!=Eigen::Success)return;const auto&e=es.eigenvalues();
  if(e[1]<.0004||e[0]>.0004||e[0]>.1*e[1])return;
  normal=es.eigenvectors().col(0);if(normal.dot(view-mean)<0)normal=-normal;
  variance=.03*.03+std::max(0.,e[0]);valid=true;
 }
};
class Map {
 double grid_,max_distance_;std::map<Key,Cell> cells_;
 Key key(const V&p)const {Key k;for(int j=0;j<3;++j){double v=std::floor(p[j]/grid_);if(!std::isfinite(v)||std::abs(v)>1e8)throw std::runtime_error("Invalid plane map coordinate");k[j]=int(v);}return k;}
public:
 explicit Map(double grid=.4,double max_distance=35):grid_(grid),max_distance_(max_distance){if(!std::isfinite(grid)||grid<=0||!std::isfinite(max_distance)||max_distance<=0)throw std::runtime_error("Invalid plane map grid or extent");}
 size_t size()const{return cells_.size();}
 void add(const std::vector<V>&points,const Sophus::SE3d&pose,size_t frame){
  std::map<Key,bool> touched;
  for(const auto&p:points){V world=pose*p;auto k=key(world);cells_[k].add(world,pose.translation(),frame);touched[k]=true;}
  for(const auto&[k,_]:touched)cells_.at(k).fit();
  for(auto it=cells_.begin();it!=cells_.end();)if((it->second.mean-pose.translation()).squaredNorm()>max_distance_*max_distance_)it=cells_.erase(it);else ++it;
 }
 std::vector<iekf::PlaneObservation> match(const std::vector<V>&points,const Sophus::SE3d&pose,Stats&stats)const{
  std::vector<iekf::PlaneObservation> out;out.reserve(points.size());stats={};
  const double cosine=std::cos(75.*3.141592653589793/180.);
  for(const auto&p:points){
   V world=pose*p;Key k=key(world);const Cell*best=nullptr;double distance=1e100;
   for(int x=-1;x<=1;++x)for(int y=-1;y<=1;++y)for(int z=-1;z<=1;++z){
    auto it=cells_.find({k[0]+x,k[1]+y,k[2]+z});if(it==cells_.end()||!it->second.valid)continue;
    const auto&c=it->second;double d=(world-c.mean).squaredNorm();
    if(d>grid_*grid_*2.25)continue;
    // Prefer local support; no point is assigned to a distant infinite plane.
    if(d<distance){distance=d;best=&c;}
   }
   if(!best){++stats.no_plane;continue;}
   V sight=pose.translation()-world;double range=sight.norm();double incidence=range>0?best->normal.dot(sight)/range:0;
   if(incidence<=0){++stats.backface;continue;}if(incidence<cosine){++stats.grazing;continue;}
   if(std::abs(best->normal.dot(world-best->mean))>.15){++stats.residual;continue;}
   out.push_back({p,best->mean,best->normal,best->variance});++stats.matches;
  }return out;
 }
};
}
