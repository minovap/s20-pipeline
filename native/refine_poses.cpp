// Controlled LiDAR-only offline pose experiment; not Studio BA or full inertial SLAM.
#include <kiss_icp/core/VoxelUtils.hpp>
#include <kiss_icp/core/VoxelHashMap.hpp>
#include <Eigen/Eigenvalues>
#include <tbb/parallel_for.h>
#include <tbb/global_control.h>
#include <iomanip>
#include <atomic>
#include <mutex>
#include <iostream>
#include "observations.hpp"
using V=Eigen::Vector3d;using M=Eigen::Matrix3d;using Pose=Sophus::SE3d;using Key=kiss_icp::Voxel;
using V6=Eigen::Matrix<double,6,1>;using M6=Eigen::Matrix<double,6,6>;
struct Scan{observations::Header h;std::vector<V> train,held;};
struct Cell{V sum=V::Zero();M outer=M::Zero();uint32_t count=0,frames=0,last=UINT32_MAX;};
struct Plane{V mean,normal;double weight;};
struct Map{tsl::robin_map<Key,Plane> planes;};
struct Match{V local,normal,mean;double weight;};
struct Result{Pose pose;size_t matches=0,held_before_count=0,held_after_count=0;double train_before=0,train_after=0,held_before=0,held_after=0;bool accepted=false;};
constexpr double mapGrid=.20,sourceGrid=.15,cutoff=.10,noise=.02;
Key key(const V&p,double grid){return kiss_icp::PointToVoxel(p,grid);}
std::vector<Match> matches(const std::vector<V>& points,const Pose& pose,const Map&map){std::vector<Match> out;out.reserve(points.size());
 for(const auto&local:points){V p=pose*local;Key k=key(p,mapGrid);const Plane*best=nullptr;double dist=.3*.3;
  for(int x=-1;x<=1;++x)for(int y=-1;y<=1;++y)for(int z=-1;z<=1;++z){auto it=map.planes.find(k+Key(x,y,z));if(it==map.planes.end())continue;const auto&q=it->second;double d=(p-q.mean).squaredNorm();if(d>=dist||std::abs(q.normal.dot(p-q.mean))>cutoff)continue;dist=d;best=&q;}
  if(!best)continue;const V ray=pose.so3()*local.normalized();if(std::abs(ray.dot(best->normal))<.1)continue;
  out.push_back({local,best->normal,best->mean,best->weight});
 }return out;
}
double cost(const std::vector<Match>&matches,const Pose&pose){if(matches.empty())return 0;double c=0;for(const auto&m:matches){double r=m.normal.dot(pose*m.local-m.mean);c+=m.weight*noise*noise*std::log1p(r*r/(noise*noise));}return c/matches.size();}
double medianResidual(const std::vector<Match>&m,const Pose&p){if(m.empty())return 0;std::vector<double>d;d.reserve(m.size());for(auto&a:m)d.push_back(std::abs(a.normal.dot(p*a.local-a.mean)));std::nth_element(d.begin(),d.begin()+d.size()/2,d.end());return d[d.size()/2];}
Pose increment(const Pose&p,const V6&dx,double gain){return Pose(Sophus::SO3d::exp(gain*dx.tail<3>())*p.so3(),p.translation()+gain*dx.head<3>());}
int main(int argc,char**argv){try{
 if(argc<3){std::cerr<<"s20_refine_poses observations output_dir [threads=16]\n";return 2;}
 if(std::filesystem::exists(argv[2])&&!std::filesystem::is_empty(argv[2]))throw std::runtime_error("Pose refinement requires an empty output directory");
 int threads=argc>3?std::stoi(argv[3]):16;if(threads<1||threads>128)throw std::runtime_error("Invalid threads");tbb::global_control cap(tbb::global_control::max_allowed_parallelism,threads);
 std::vector<Scan>scans;for(const auto&path:observations::files(argv[1])){auto[h,records]=observations::read(path);if(!scans.empty()&&(h.origin!=scans.front().h.origin||h.frame<=scans.back().h.frame))throw std::runtime_error("Mixed recording origins or unordered frame IDs");Scan s;s.h=h;tsl::robin_map<Key,V>train,held;
  for(const auto&r:records){if(r.range>30)continue;V p=Eigen::Map<const V>(r.xyz);auto&g=r.sample%5==0?held:train;g.try_emplace(key(p,sourceGrid),p);}
  for(const auto&v:train)s.train.push_back(v.second);for(const auto&v:held)s.held.push_back(v.second);scans.push_back(std::move(s));
 }
 if(scans.empty())throw std::runtime_error("No observations to refine");
 // One-second groups cycle through source, fit-map and independent check-map roles. Same scan and held-out returns
 // never enter its target map. All three maps remain frozen during all optimization.
 std::array<tsl::robin_map<Key,Cell>,3> cells;
 for(const auto&s:scans){auto&map=cells[(s.h.frame/10)%3];for(const auto&l:s.train){V p=s.h.pose*l;auto&c=map[key(p,mapGrid)];c.sum+=p;c.outer.noalias()+=p*p.transpose();++c.count;if(c.last!=s.h.frame){c.last=s.h.frame;++c.frames;}}}
 std::array<Map,3>maps;for(int group=0;group<3;++group)for(const auto&item:cells[group]){const auto&c=item.second;if(c.count<8||c.frames<3)continue;V mean=c.sum/c.count;M cov=c.outer/c.count-mean*mean.transpose();Eigen::SelfAdjointEigenSolver<M> e(cov);if(e.info()!=Eigen::Success)continue;auto v=e.eigenvalues();if(v[1]<1e-6||v[0]>.10*v[1]||v[1]<.05*v[2])continue;maps[group].planes.insert({item.first,{mean,e.eigenvectors().col(0),1.-std::max(0.,v[0]/v[1])}});}
 std::vector<Result>results(scans.size());
 std::atomic<size_t> refined{0};std::mutex progress_mutex;
 tbb::parallel_for(size_t(0),scans.size(),[&](size_t i){
  struct Done{std::atomic<size_t>&n;size_t total;std::mutex&m;~Done(){size_t k=++n;if(k%50==0||k==total){std::lock_guard<std::mutex> lock(m);std::cout<<"refined "<<k<<"/"<<total<<" scans"<<std::endl;}}} done{refined,scans.size(),progress_mutex};const auto&s=scans[i];const auto&map=maps[((s.h.frame/10)%3+1)%3];const auto&heldmap=maps[((s.h.frame/10)%3+2)%3];auto&result=results[i];result.pose=s.h.pose;
  auto original=matches(s.train,s.h.pose,map);result.matches=original.size();result.train_before=cost(original,s.h.pose);
  auto held=matches(s.held,s.h.pose,heldmap);result.held_before_count=held.size();result.held_before=medianResidual(held,s.h.pose);
  // Anchor first frame; underconstrained frames retain their original pose.
  if(i&&original.size()>=250){Pose current=s.h.pose;
   for(int iteration=0;iteration<8;++iteration){auto pairs=matches(s.train,current,map);if(pairs.size()<250)break;M6 A=M6::Zero();V6 b=V6::Zero();
    for(const auto&m:pairs){V arm=current.so3()*m.local;double r=m.normal.dot(arm+current.translation()-m.mean);V6 j;j.head<3>()=m.normal;j.tail<3>()=arm.cross(m.normal);double w=m.weight/(1+r*r/(noise*noise));A.noalias()+=w*j*j.transpose();b.noalias()-=w*j*r;}
    M6 scaling=M6::Identity();double lever2=0;for(const auto&m:pairs)lever2+=m.local.squaredNorm();scaling.bottomRightCorner<3,3>()*=1./std::max(1.,std::sqrt(lever2/pairs.size()));
    Eigen::SelfAdjointEigenSolver<M6> eigen(scaling*A*scaling);if(eigen.info()!=Eigen::Success||eigen.eigenvalues()[0]<1e-5||eigen.eigenvalues()[5]/eigen.eigenvalues()[0]>1e8)break;
    A.diagonal().array()+=1e-6;V6 dx=A.ldlt().solve(b);if(!dx.allFinite())break;
    bool changed=false;double old=cost(pairs,current);
    for(double gain:{1.,.5,.25,.125}){auto candidate=increment(current,dx,gain);
     if((candidate.translation()-s.h.pose.translation()).norm()>.05||(candidate.so3()*s.h.pose.so3().inverse()).log().norm()>.5*M_PI/180)continue;
     if(cost(pairs,candidate)<old){current=candidate;changed=true;break;}
    }
    if(!changed||dx.norm()<1e-7)break;
   }
   if(cost(original,current)<result.train_before){result.pose=current;result.accepted=true;}
  }
  result.train_after=cost(original,result.pose);result.held_after=medianResidual(held,result.pose);
  result.held_after_count=matches(s.held,result.pose,heldmap).size();
 });
 std::filesystem::path out(argv[2]);std::filesystem::create_directories(out);std::ofstream poses(out/"poses.txt"),csv(out/"pose-diagnostics.csv");poses<<"# origin_ns "<<scans.front().h.origin<<"\n# frame time x y z qx qy qz qw\n"<<std::setprecision(17);csv<<"frame,accepted,training_matches,train_cost_before,train_cost_after,held_matches_before,held_matches_after,held_median_before_m,held_median_after_m,translation_m,rotation_deg\n"<<std::setprecision(12);
 size_t accepted=0;for(size_t i=0;i<scans.size();++i){const auto&s=scans[i];const auto&r=results[i];const auto&p=r.pose;const auto&q=p.unit_quaternion();poses<<s.h.frame<<" "<<s.h.end<<" "<<p.translation().transpose()<<" "<<q.x()<<" "<<q.y()<<" "<<q.z()<<" "<<q.w()<<"\n";csv<<s.h.frame<<","<<r.accepted<<","<<r.matches<<","<<r.train_before<<","<<r.train_after<<","<<r.held_before_count<<","<<r.held_after_count<<","<<r.held_before<<","<<r.held_after<<","<<(p.translation()-s.h.pose.translation()).norm()<<","<<(p.so3()*s.h.pose.so3().inverse()).log().norm()*180/M_PI<<"\n";accepted+=r.accepted;}
 std::ofstream report(out/"pose-refinement.json");report<<"{\"method\":\"native-only disjoint-frame frozen-surfel point-to-plane experiment\",\"uses_studio_data\":false,\"uses_imu_residuals\":false,\"redeskews_measurements\":false,\"map_grid_m\":0.2,\"source_grid_m\":0.15,\"held_out_sample_modulus\":5,\"held_out_same_correspondences\":true,\"independent_validation_map\":true,\"map_temporal_groups\":3,\"map_temporal_group_frames\":10,\"map_plane_counts\":["<<maps[0].planes.size()<<","<<maps[1].planes.size()<<","<<maps[2].planes.size()<<"],\"frames\":"<<scans.size()<<",\"adjusted_frames\":"<<accepted<<"}";
 std::cout<<accepted<<" / "<<scans.size()<<" poses adjusted; "<<maps[0].planes.size()<<" / "<<maps[1].planes.size()<<" reference surfels\n";
 return 0;
}catch(const std::exception&e){std::cerr<<e.what()<<"\n";return 1;}}
