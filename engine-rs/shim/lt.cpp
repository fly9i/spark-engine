// SPDX-License-Identifier: MIT
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAStream.h>
#include <cublasLt.h>
#include <map>
#include <tuple>
#include <memory>
#include <cstdio>

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
