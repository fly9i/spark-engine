// SPDX-License-Identifier: MIT
// Shared Gumbel-max sampling helpers (keyed by (seed, pos, global vocab id)).
#pragma once
#include <math.h>

__device__ __forceinline__ unsigned long long glm53_mix64(unsigned long long x){
  x^=x>>30;x*=0xbf58476d1ce4e5b9ULL;x^=x>>27;x*=0x94d049bb133111ebULL;x^=x>>31;return x;
}
__device__ __forceinline__ float gumbel_exact(unsigned long long key,long long id,float t){
  if(!(t>0.f))return 0.f;
  unsigned long long h=glm53_mix64(key^glm53_mix64((unsigned long long)id+0x9e3779b97f4a7c15ULL));
  double r=((double)(h>>11)+0.5)*(1.0/9007199254740992.0);double e=-log1p(-r);return (float)(-(double)t*log(e));
}
__device__ __forceinline__ float gumbel_estimate(unsigned long long key,long long id,float t){
  if(!(t>0.f))return 0.f;
  unsigned long long h=glm53_mix64(key^glm53_mix64((unsigned long long)id+0x9e3779b97f4a7c15ULL));
  const unsigned long long m=h>>11;float e;
  if(m<(1ULL<<52)){const float r=((float)m+0.5f)*1.1102230246251565e-16f;e=-log1pf(-r);}
  else{const float u=((float)((1ULL<<53)-m)-0.5f)*1.1102230246251565e-16f;e=-logf(u);}
  return -t*logf(e);
}
__device__ __forceinline__ bool gumbel_better(float v,int i,float bv,int bi){
  const bool vn=v!=v,bn=bv!=bv;
  if(vn||bn)return vn&&(!bn||i<bi);
  return v>bv||(v==bv&&i<bi);
}
