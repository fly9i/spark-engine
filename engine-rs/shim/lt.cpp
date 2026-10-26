// SPDX-License-Identifier: MIT
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAStream.h>
#include <cublasLt.h>
#include <map>
#include <tuple>
#include <memory>
#include <cstdio>
#include <vector>
#include <algorithm>

struct Plan {
    cublasLtMatmulDesc_t op=nullptr;
    cublasLtMatrixLayout_t a=nullptr,b=nullptr,c=nullptr;
    cublasLtMatmulHeuristicResult_t algorithms[32];int count=0;
    ~Plan(){if(op)cublasLtMatmulDescDestroy(op);if(a)cublasLtMatrixLayoutDestroy(a);if(b)cublasLtMatrixLayoutDestroy(b);if(c)cublasLtMatrixLayoutDestroy(c);}
};
// This engine serializes all CUDA work per process. Scratch has process
// lifetime so graph executions retain a stable workspace address.
struct Runtime {
    cublasLtHandle_t handle=nullptr;at::Tensor workspace;
    std::map<std::tuple<int,int,int,int>,std::unique_ptr<Plan>> plans;
    Runtime(){TORCH_CHECK(cublasLtCreate(&handle)==CUBLAS_STATUS_SUCCESS);workspace=at::empty({32*1024*1024},at::TensorOptions().device(at::kCUDA).dtype(at::kByte));}
};
static Runtime& runtime(){static auto* r=new Runtime();return *r;}
static Plan& plan(int m,int n,int k,int fp32) {
    auto& r=runtime();auto key=std::make_tuple(m,n,k,fp32);auto found=r.plans.find(key);if(found!=r.plans.end())return *found->second;
    cudaStreamCaptureStatus captured;cudaStreamIsCapturing(at::cuda::getCurrentCUDAStream(),&captured);
    TORCH_CHECK(captured==cudaStreamCaptureStatusNone,"Lt shape must be warmed before graph capture");
    auto p=std::make_unique<Plan>();
    TORCH_CHECK(cublasLtMatmulDescCreate(&p->op,CUBLAS_COMPUTE_32F,CUDA_R_32F)==CUBLAS_STATUS_SUCCESS);
    cublasOperation_t trans=CUBLAS_OP_T;
    TORCH_CHECK(cublasLtMatmulDescSetAttribute(p->op,CUBLASLT_MATMUL_DESC_TRANSA,&trans,sizeof(trans))==CUBLAS_STATUS_SUCCESS);
    TORCH_CHECK(cublasLtMatrixLayoutCreate(&p->a,CUDA_R_16F,k,n,k)==CUBLAS_STATUS_SUCCESS);
    TORCH_CHECK(cublasLtMatrixLayoutCreate(&p->b,CUDA_R_16F,k,m,k)==CUBLAS_STATUS_SUCCESS);
    TORCH_CHECK(cublasLtMatrixLayoutCreate(&p->c,fp32?CUDA_R_32F:CUDA_R_16F,n,m,n)==CUBLAS_STATUS_SUCCESS);
    cublasLtMatmulPreference_t pref;TORCH_CHECK(cublasLtMatmulPreferenceCreate(&pref)==CUBLAS_STATUS_SUCCESS);
    size_t bytes=r.workspace.numel();cublasLtMatmulPreferenceSetAttribute(pref,CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,&bytes,sizeof(bytes));
    auto rc=cublasLtMatmulAlgoGetHeuristic(r.handle,p->op,p->a,p->b,p->c,p->c,pref,32,p->algorithms,&p->count);
    cublasLtMatmulPreferenceDestroy(pref);TORCH_CHECK(rc==CUBLAS_STATUS_SUCCESS && p->count>0,"Lt heuristic failed");
    auto* raw=p.get();r.plans.emplace(key,std::move(p));return *raw;
}
extern "C" int rs_lt_count(int m,int n,int k,int fp32){try{return plan(m,n,k,fp32).count;}catch(const std::exception& e){fprintf(stderr,"[Lt] %s\n",e.what());return -1;}}
extern "C" int rs_lt_mm16(const void* x,const void* w,void* y,int m,int n,int k,int fp32,int algorithm) {
    try{auto& p=plan(m,n,k,fp32);auto& r=runtime();if(algorithm<0||algorithm>=p.count)return -2;
        float alpha=1.f,beta=0.f;
        return int(cublasLtMatmul(r.handle,p.op,&alpha,w,p.a,x,p.b,&beta,y,p.c,y,p.c,&p.algorithms[algorithm].algo,r.workspace.data_ptr(),r.workspace.numel(),at::cuda::getCurrentCUDAStream()));
    }catch(const std::exception& e){fprintf(stderr,"[Lt] %s\n",e.what());return -1;}
}

// ---- FP32 GEMM of a row subset with the bits of the whole call (the DS41 router, sequence-parallel routing)
// y [m, n] = x [m, k] . w [n, k]^T in FP32 as at::matmul computes it for `full` rows (cuBLAS's SIMT sgemm: each output
// element one thread's FMA chain over its K slice, the split-K slices summed in order inside the kernel), for any m:
// the cublasLt configuration (algorithm, tile, stages, split-K, reduction, swizzle) that gives at::matmul's bits for the
// full call and for row subsets of it is found once per (full, n, k) on random data and then run on m rows. A row's
// result then depends only on that configuration (no other row enters it). Not found: rs_lt_rows_mm returns 1 and the
// caller keeps the full-shape call.
namespace {
struct Lay { cublasLtMatmulDesc_t op=nullptr; cublasLtMatrixLayout_t a=nullptr,b=nullptr,c=nullptr; };
Lay& lay32(int m,int n,int k) {
    static std::map<std::tuple<int,int,int>,Lay> ls; auto key=std::make_tuple(m,n,k); auto f=ls.find(key); if(f!=ls.end()) return f->second;
    Lay l; cublasOperation_t t=CUBLAS_OP_T;
    TORCH_CHECK(cublasLtMatmulDescCreate(&l.op,CUBLAS_COMPUTE_32F,CUDA_R_32F)==CUBLAS_STATUS_SUCCESS);
    TORCH_CHECK(cublasLtMatmulDescSetAttribute(l.op,CUBLASLT_MATMUL_DESC_TRANSA,&t,sizeof(t))==CUBLAS_STATUS_SUCCESS);
    TORCH_CHECK(cublasLtMatrixLayoutCreate(&l.a,CUDA_R_32F,k,n,k)==CUBLAS_STATUS_SUCCESS);
    TORCH_CHECK(cublasLtMatrixLayoutCreate(&l.b,CUDA_R_32F,k,m,k)==CUBLAS_STATUS_SUCCESS);
    TORCH_CHECK(cublasLtMatrixLayoutCreate(&l.c,CUDA_R_32F,n,m,n)==CUBLAS_STATUS_SUCCESS);
    return ls.emplace(key,l).first->second;
}
int run32(const cublasLtMatmulAlgo_t& al,const float* x,const float* w,float* y,int m,int n,int k) {
    auto& r=runtime(); auto& l=lay32(m,n,k); float one=1.f,zero=0.f;
    return int(cublasLtMatmul(r.handle,l.op,&one,w,l.a,x,l.b,&zero,y,l.c,y,l.c,&al,r.workspace.data_ptr(),r.workspace.numel(),at::cuda::getCurrentCUDAStream()));
}
bool check32(const cublasLtMatmulAlgo_t& al,int m,int n,int k) {
    auto& r=runtime(); auto& l=lay32(m,n,k); cublasLtMatmulHeuristicResult_t h{};
    return cublasLtMatmulAlgoCheck(r.handle,l.op,l.a,l.b,l.c,l.c,&al,&h)==CUBLAS_STATUS_SUCCESS && h.workspaceSize<=(size_t)r.workspace.numel();
}
template<class T> std::vector<T> cap(const cublasLtMatmulAlgo_t& al,cublasLtMatmulAlgoCapAttributes_t a) {
    size_t sz=0; cublasLtMatmulAlgoCapGetAttribute(&al,a,nullptr,0,&sz); std::vector<T> v(sz/sizeof(T));
    if(sz) cublasLtMatmulAlgoCapGetAttribute(&al,a,v.data(),sz,&sz); return v;
}
template<class T> T cap1(const cublasLtMatmulAlgo_t& al,cublasLtMatmulAlgoCapAttributes_t a) { T v{}; size_t sz=0; cublasLtMatmulAlgoCapGetAttribute(&al,a,&v,sizeof(v),&sz); return v; }
template<class T> void cfg(cublasLtMatmulAlgo_t& al,cublasLtMatmulAlgoConfigAttributes_t a,T v) { cublasLtMatmulAlgoConfigSetAttribute(&al,a,&v,sizeof(v)); }
template<class T> T cfgget(const cublasLtMatmulAlgo_t& al,cublasLtMatmulAlgoConfigAttributes_t a) { T v{}; size_t sz=0; cublasLtMatmulAlgoConfigGetAttribute(&al,a,&v,sizeof(v),&sz); return v; }
struct Found { bool ok=false; cublasLtMatmulAlgo_t algo; };
std::map<std::tuple<int,int,int>,Found>& found() { static std::map<std::tuple<int,int,int>,Found> f; return f; }
}
// the configuration for (full, n, k) with weights w [n, k] FP32 (device); 0 found, 1 none; verbose: print it
extern "C" int rs_lt_rows_find(const float* w,int full,int n,int k,int verbose) {
    try {
        auto key=std::make_tuple(full,n,k); auto f=found().find(key); if(f!=found().end()) return f->second.ok?0:1;
        cudaStreamCaptureStatus cs; cudaStreamIsCapturing(at::cuda::getCurrentCUDAStream(),&cs);
        TORCH_CHECK(cs==cudaStreamCaptureStatusNone,"rs_lt_rows_find inside a graph capture");
        auto opt=at::TensorOptions().device(at::kCUDA).dtype(at::kFloat);
        auto W=at::from_blob((void*)w,{n,k},opt);
        // test rows: BF16-valued, irregular, made on the device without a generator (no RNG state touched)
        auto X=at::sin(at::arange((long long)full*k,opt).mul(0.7548776662f).add(0.31f)).mul(at::cos(at::arange((long long)full*k,opt).mul(0.0123f))).mul(4.f)
               .to(at::kBFloat16).to(at::kFloat).view({full,k});
        auto ref=at::matmul(X,W.t());
        auto Y=at::empty({full,n},opt);
        const int subs[][2]={{0,full/3},{full/3,full/3+1},{1,1},{129,64},{full-full/3,full/3},{7,full-7}};
        auto test=[&](cublasLtMatmulAlgo_t al)->bool {
            if(!check32(al,full,n,k) || run32(al,X.data_ptr<float>(),w,Y.data_ptr<float>(),full,n,k)) return false;
            if(!at::equal(Y,ref)) return false;
            for(auto& s:subs) { int o=s[0],m=s[1]; if(m<1||o+m>full) continue;
                if(!check32(al,m,n,k)) return false;
                auto ys=at::empty({m,n},opt);
                if(run32(al,X.data_ptr<float>()+(long long)o*k,w,ys.data_ptr<float>(),m,n,k) || !at::equal(ys,ref.narrow(0,o,m))) return false; }
            return true; };
        auto& r=runtime(); auto& l=lay32(full,n,k);
        std::vector<cublasLtMatmulAlgo_t> cands;
        { cublasLtMatmulPreference_t pref; cublasLtMatmulPreferenceCreate(&pref); size_t b=r.workspace.numel();
          cublasLtMatmulPreferenceSetAttribute(pref,CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,&b,sizeof(b));
          cublasLtMatmulHeuristicResult_t hr[64]; int cnt=0; cublasLtMatmulAlgoGetHeuristic(r.handle,l.op,l.a,l.b,l.c,l.c,pref,64,hr,&cnt); cublasLtMatmulPreferenceDestroy(pref);
          for(int i=0;i<cnt;++i) cands.push_back(hr[i].algo); }
        Found res; int tried=0;
        for(auto& c:cands) { ++tried; if(test(c)) { res.ok=true; res.algo=c; break; } }
        if(!res.ok) {
            // not among the heuristic's: every enumerated configuration
            cands.clear();
            int ids[256],nid=0; cublasLtMatmulAlgoGetIds(r.handle,CUBLAS_COMPUTE_32F,CUDA_R_32F,CUDA_R_32F,CUDA_R_32F,CUDA_R_32F,CUDA_R_32F,256,ids,&nid);
            for(int i=0;i<nid;++i) { cublasLtMatmulAlgo_t al; if(cublasLtMatmulAlgoInit(r.handle,CUBLAS_COMPUTE_32F,CUDA_R_32F,CUDA_R_32F,CUDA_R_32F,CUDA_R_32F,CUDA_R_32F,ids[i],&al)) continue;
                auto tiles=cap<uint32_t>(al,CUBLASLT_ALGO_CAP_TILE_IDS); if(tiles.empty()) tiles.push_back(CUBLASLT_MATMUL_TILE_UNDEFINED);
                auto stages=cap<uint32_t>(al,CUBLASLT_ALGO_CAP_STAGES_IDS); if(stages.empty()) stages.push_back(CUBLASLT_MATMUL_STAGES_UNDEFINED);
                const int splitk=cap1<int32_t>(al,CUBLASLT_ALGO_CAP_SPLITK_SUPPORT); const uint32_t red=cap1<uint32_t>(al,CUBLASLT_ALGO_CAP_REDUCTION_SCHEME_MASK);
                const int swz=cap1<uint32_t>(al,CUBLASLT_ALGO_CAP_CTA_SWIZZLING_SUPPORT);
                for(auto t:tiles) for(auto s:stages) for(int sk=1;sk<=(splitk?16:1);++sk) for(uint32_t rs=0;rs<=8;rs=rs?rs<<1:1) for(int z=0;z<=swz;++z) {
                    if(sk==1&&rs) continue; if(sk>1&&rs&&!(red&rs)) continue; if(sk>1&&!rs) continue;
                    cublasLtMatmulAlgo_t c=al; cfg<uint32_t>(c,CUBLASLT_ALGO_CONFIG_TILE_ID,t); cfg<uint32_t>(c,CUBLASLT_ALGO_CONFIG_STAGES_ID,s);
                    cfg<int32_t>(c,CUBLASLT_ALGO_CONFIG_SPLITK_NUM,sk); cfg<uint32_t>(c,CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME,rs); cfg<uint32_t>(c,CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING,(uint32_t)z);
                    cands.push_back(c); } }
            // the heuristic's own list first, then the enumerated configurations with cuBLAS's SIMT sgemm tile (128 x 256) first
            { std::stable_partition(cands.begin(),cands.end(),[](const cublasLtMatmulAlgo_t& c){ return cfgget<uint32_t>(c,CUBLASLT_ALGO_CONFIG_TILE_ID)==CUBLASLT_MATMUL_TILE_128x256; }); }
            for(auto& c:cands) { ++tried; if(test(c)) { res.ok=true; res.algo=c; break; } }
        }
        cudaGetLastError();
        if(verbose) {
            if(res.ok) fprintf(stderr,"[Lt rows] full %d n %d k %d: config found after %d of %zu (algo %d tile %u stages %u splitk %d reduction %u swizzle %u)\n",full,n,k,tried,cands.size(),
                cfgget<int32_t>(res.algo,CUBLASLT_ALGO_CONFIG_ID),cfgget<uint32_t>(res.algo,CUBLASLT_ALGO_CONFIG_TILE_ID),cfgget<uint32_t>(res.algo,CUBLASLT_ALGO_CONFIG_STAGES_ID),
                cfgget<int32_t>(res.algo,CUBLASLT_ALGO_CONFIG_SPLITK_NUM),cfgget<uint32_t>(res.algo,CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME),cfgget<uint32_t>(res.algo,CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING));
            else fprintf(stderr,"[Lt rows] full %d n %d k %d: no configuration among %zu reproduces at::matmul\n",full,n,k,cands.size());
        }
        found()[key]=res; return res.ok?0:1;
    } catch(const std::exception& e) { fprintf(stderr,"[Lt rows] %s\n",e.what()); return -1; }
}
// y [m, n] = x [m, k] . w^T with the bits of rows of an at::matmul over `full` rows (rs_lt_rows_find first; 1: none)
extern "C" int rs_lt_rows_mm(const float* x,const float* w,float* y,int full,int m,int n,int k) {
    try { auto f=found().find(std::make_tuple(full,n,k)); if(f==found().end()||!f->second.ok) return 1;
          if(!check32(f->second.algo,m,n,k)) return 2; return run32(f->second.algo,x,w,y,m,n,k)?3:0; }
    catch(const std::exception& e) { fprintf(stderr,"[Lt rows] %s\n",e.what()); return -1; }
}
