// SPDX-License-Identifier: MIT
//! engine-rs C++ shim:桥接 exllamav3_ext 原生 GEMM(专家 GEMV 直接从
//! trellis 算,免 fp16 物化 = 算力换带宽)与后续 c10d/NCCL(TP2)。
//! ABI:入口拿裸数据指针 + 形状,经 at::from_blob 零拷贝重建 at::Tensor。
//! exl3_gemm 调用姿势(对齐 BC_LinearEXL3::run_gr,linear.cpp:44):
//!   exl3_gemm(x, trellis, y, suh, xh_workspace, svh, force_shape, mcg, mul1, 0)

#include <torch/all.h>
#include <ATen/cuda/CUDAGraph.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <cuda_runtime.h>
#include <torch/csrc/distributed/c10d/ProcessGroupNCCL.hpp>
#include <torch/csrc/distributed/c10d/TCPStore.hpp>
#include <torch/csrc/distributed/c10d/PrefixStore.hpp>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <unordered_map>
#include <nvtx3/nvToolsExt.h>
#include <memory>
#include <optional>

static_assert(NCCL_VERSION_CODE == 22907 && sizeof(ncclConfig_t) == 88,
              "NCCL header must match libtorch 2.13 cu130 build ABI");

class Graph; // 前向声明:mangling 需要 P5Graph

extern "C" int glm53_sinkhorn_cuda(const float*, float*, int, cudaStream_t);
extern "C" int glm53_dsa_node_append_cuda(const long long*,const float*,const float*,float*,float*,float*,const float*,const float*,const float*,void*,const void*,long long*,int,int,float*,cudaStream_t);
extern "C" int rs_dsa_node_append(const long long* parent_len,const float* ptk,const float* ptg,float* ctk,float* ctg,float* cpools,const float* ape,const float* krow,const float* grow,void* clatent,const void* lrow,long long* clen,int dim,int lat){
    return glm53_dsa_node_append_cuda(parent_len,ptk,ptg,ctk,ctg,cpools,ape,krow,grow,clatent,lrow,clen,dim,lat,nullptr,at::cuda::getCurrentCUDAStream());
}
extern "C" int rs_dsa_node_append_row(const long long* parent_len,const float* ptk,const float* ptg,float* ctk,float* ctg,float* cpools,const float* ape,const float* krow,const float* grow,void* clatent,const void* lrow,long long* clen,int dim,int lat,float* crow){
    return glm53_dsa_node_append_cuda(parent_len,ptk,ptg,ctk,ctg,cpools,ape,krow,grow,clatent,lrow,clen,dim,lat,crow,at::cuda::getCurrentCUDAStream());
}
extern "C" int glm53_latent_half_bmm_cuda(const float*,long long,long long,const void*,long long,float*,int,int,int,int,cudaStream_t);
extern "C" int rs_latent_half_bmm(const float* x,long long xh,long long xm,const void* w,long long wh,float* y,int heads,int m,int n,int k){
    return glm53_latent_half_bmm_cuda(x,xh,xm,w,wh,y,heads,m,n,k,at::cuda::getCurrentCUDAStream());
}
extern "C" int glm53_kda_conv_l2_cuda(const float*,const float*,const float*,float*,float*,float*,int,int,float,cudaStream_t);
extern "C" int rs_kda_conv_l2(const float* base,const float* projected,const float* weights,float* q,float* k,float* v,int t,int h,float scale){
    return glm53_kda_conv_l2_cuda(base,projected,weights,q,k,v,t,h,scale,at::cuda::getCurrentCUDAStream());}
extern "C" int glm53_kda_chain_l2_cuda(const float*,float*,float*,float*,int,int,float,cudaStream_t);
extern "C" int rs_kda_chain_l2(const float* act,float* q,float* k,float* v,int t,int h,float scale){return glm53_kda_chain_l2_cuda(act,q,k,v,t,h,scale,at::cuda::getCurrentCUDAStream());}
extern "C" int glm53_rms_rows_cuda(const float*,const float*,float*,int,int,float,cudaStream_t);
extern "C" int rs_rms_rows(const float* x,const float* w,float* y,int rows,int d,float eps){return glm53_rms_rows_cuda(x,w,y,rows,d,eps,at::cuda::getCurrentCUDAStream());}
extern "C" int glm53_rms_rows_ld_cuda(const float*,int,const float*,float*,int,int,float,cudaStream_t);
extern "C" int rs_rms_rows_ld(const float* x,int ldx,const float* w,float* y,int rows,int d,float eps){return glm53_rms_rows_ld_cuda(x,ldx,w,y,rows,d,eps,at::cuda::getCurrentCUDAStream());}
extern "C" int glm53_round_half_inplace_cuda(float*,long long,cudaStream_t);
extern "C" int rs_round_half_inplace(float* y,long long n){return glm53_round_half_inplace_cuda(y,n,at::cuda::getCurrentCUDAStream());}
extern "C" int glm53_half_skinny_cuda(const float*,const void*,void*,int,int,int,int,int,cudaStream_t);
extern "C" int rs_half_skinny(const float* x,const void* w,void* y,int m,int n,int k,int out,int ks){
    return glm53_half_skinny_cuda(x,w,y,m,n,k,out,ks,at::cuda::getCurrentCUDAStream());
}
extern "C" int glm53_c12_stats_cuda(const void*,int,int,void*,int*,cudaStream_t);
extern "C" int rs_c12_stats(const void* w,int n,int k,void* eb,int* cnt){return glm53_c12_stats_cuda(w,n,k,eb,cnt,at::cuda::getCurrentCUDAStream());}
extern "C" int glm53_c12_pack_cuda(const void*,int,int,const void*,const int*,void*,void*,int*,void*,cudaStream_t);
extern "C" int rs_c12_pack(const void* w,int n,int k,const void* eb,const int* ptr,void* m8,void* e4,int* col,void* val){
    return glm53_c12_pack_cuda(w,n,k,eb,ptr,m8,e4,col,val,at::cuda::getCurrentCUDAStream());
}
extern "C" int glm53_c12_verify_cuda(const void*,const void*,const void*,const void*,const int*,const int*,const void*,int,int,unsigned long long*,cudaStream_t);
extern "C" int rs_c12_verify(const void* w,const void* m8,const void* e4,const void* eb,const int* ptr,const int* col,const void* val,int n,int k,unsigned long long* bad){
    return glm53_c12_verify_cuda(w,m8,e4,eb,ptr,col,val,n,k,bad,at::cuda::getCurrentCUDAStream());
}
extern "C" int glm53_c12_big_cuda(const void*,const void*,const void*,const void*,const int*,const int*,const void*,float*,int,int,int,int,int,int,cudaStream_t);
extern "C" int rs_c12_big(const void* x,const void* m8,const void* e4,const void* eb,const int* ptr,const int* col,const void* val,float* y,int m,int n,int k,int ldy,int rounded,int stages){
    return glm53_c12_big_cuda(x,m8,e4,eb,ptr,col,val,y,m,n,k,ldy,rounded,stages,at::cuda::getCurrentCUDAStream());
}
extern "C" int glm53_c12_gemm_cuda(const float*,const void*,const void*,const void*,const int*,const int*,const void*,int,int,void*,int,int,int,cudaStream_t);
extern "C" int glm53_c12_gemm_xh_cuda(const void*,const void*,const void*,const void*,const int*,const int*,const void*,int,int,void*,int,int,int,cudaStream_t);
extern "C" int rs_c12_gemm_xh(const void* x,const void* m8,const void* e4,const void* eb,const int* ptr,const int* col,const void* val,int n,int k,void* y,int m,int out,int ks){
    return glm53_c12_gemm_xh_cuda(x,m8,e4,eb,ptr,col,val,n,k,y,m,out,ks,at::cuda::getCurrentCUDAStream());}

extern "C" int rs_c12_gemm(const float* x,const void* m8,const void* e4,const void* eb,const int* ptr,const int* col,const void* val,int n,int k,void* y,int m,int out,int ks){
    return glm53_c12_gemm_cuda(x,m8,e4,eb,ptr,col,val,n,k,y,m,out,ks,at::cuda::getCurrentCUDAStream());
}
extern "C" int glm53_draft_q4_encode_cuda(const void*,int,int,int,void*,void*,cudaStream_t);
extern "C" int rs_draft_q4_encode(const void* w,int n,int k,int mse,void* sm,void* q){return glm53_draft_q4_encode_cuda(w,n,k,mse,sm,q,at::cuda::getCurrentCUDAStream());}
extern "C" int glm53_draft_q4_gemm_cuda(const void*,const void*,const void*,float*,int,int,int,cudaStream_t);
extern "C" int rs_draft_q4_gemm(const void* x,const void* q,const void* sm,float* y,int m,int n,int k){return glm53_draft_q4_gemm_cuda(x,q,sm,y,m,n,k,at::cuda::getCurrentCUDAStream());}
extern "C" int glm53_q4d_encode_cuda(const void*,int,int,int,void*,void*,double*,cudaStream_t);
extern "C" int rs_q4d_encode(const void* w,int n,int k,int mse,void* sm,void* q,double* err){return glm53_q4d_encode_cuda(w,n,k,mse,sm,q,err,at::cuda::getCurrentCUDAStream());}
extern "C" int glm53_q4d_gemm_cuda(const float*,const void*,const void*,int,int,void*,int,int,int,cudaStream_t);
extern "C" int rs_q4d_gemm(const float* x,const void* q,const void* sm,int n,int k,void* y,int m,int out,int ks){return glm53_q4d_gemm_cuda(x,q,sm,n,k,y,m,out,ks,at::cuda::getCurrentCUDAStream());}
extern "C" int glm53_dsa_key_ln_cuda(const float*,int,int,int,const float*,const float*,float*,float*,int,cudaStream_t);
extern "C" int rs_dsa_key_ln(const float* y,int ldy,int dim,int gcols,const float* w,const float* b,float* k,float* g,int rows){return glm53_dsa_key_ln_cuda(y,ldy,dim,gcols,w,b,k,g,rows,at::cuda::getCurrentCUDAStream());}
extern "C" int glm53_draft_q4_gemm2_cuda(const void*,int,const void*,const void*,void*,int,int,int,int,cudaStream_t);
extern "C" int rs_draft_q4_gemm2(const void* x,int xf,const void* q,const void* sm,void* y,int yb,int m,int n,int k){return glm53_draft_q4_gemm2_cuda(x,xf,q,sm,y,yb,m,n,k,at::cuda::getCurrentCUDAStream());}
extern "C" int glm53_q8_encode_cuda(const void*,int,int,int,float*,void*,double*,cudaStream_t);
extern "C" int rs_q8_encode(const void* w,int n,int k,int mse,float* s,void* q,double* err){return glm53_q8_encode_cuda(w,n,k,mse,s,q,err,at::cuda::getCurrentCUDAStream());}
extern "C" int glm53_q8_gemm_cuda(const float*,const void*,const float*,int,int,void*,int,int,int,cudaStream_t);
extern "C" int rs_q8_gemm(const float* x,const void* q,const float* s,int n,int k,void* y,int m,int out,int ks){return glm53_q8_gemm_cuda(x,q,s,n,k,y,m,out,ks,at::cuda::getCurrentCUDAStream());}
extern "C" int glm53_q8_gemm_xh_cuda(const void*,const void*,const float*,int,int,void*,int,int,int,cudaStream_t);
extern "C" int rs_q8_gemm_xh(const void* x,const void* q,const float* s,int n,int k,void* y,int m,int out,int ks){return glm53_q8_gemm_xh_cuda(x,q,s,n,k,y,m,out,ks,at::cuda::getCurrentCUDAStream());}
extern "C" int glm53_dsa_prefill_topk_expand_cuda(const float*,const int64_t*,int64_t*,int64_t*,int,int,int,cudaStream_t);
extern "C" int rs_dsa_prefill_topk_expand(const float* scores,const int64_t* pos,int64_t* selected,int64_t* out,int pools,int k,int rows){return glm53_dsa_prefill_topk_expand_cuda(scores,pos,selected,out,pools,k,rows,at::cuda::getCurrentCUDAStream());}
extern "C" int glm53_q8t_pack_cuda(const void*,const float*,int,int,void*,cudaStream_t);
extern "C" int rs_q8t_pack(const void* q,const float* s,int n,int k,void* t){return glm53_q8t_pack_cuda(q,s,n,k,t,at::cuda::getCurrentCUDAStream());}
extern "C" int glm53_q8t_gemm_cuda(const void*,const void*,int,int,void*,int,int,int,int,cudaStream_t);
extern "C" int rs_q8t_gemm(const void* x,const void* t,int n,int k,void* y,int m,int out,int ks,int xhalf){return glm53_q8t_gemm_cuda(x,t,n,k,y,m,out,ks,xhalf,at::cuda::getCurrentCUDAStream());}
extern "C" int glm53_latent_c12_bmm_cuda(const float*,long long,long long,const void*,const void*,const void*,const int*,const int*,const void*,int,int,int,float*,int,cudaStream_t);
extern "C" int rs_latent_c12_bmm(const float* x,int64_t xh,int64_t xm,const void* m8,const void* e4,const void* eb,const int* ptr,const int* col,const void* val,int heads,int n,int k,float* y,int m){
    return glm53_latent_c12_bmm_cuda(x,xh,xm,m8,e4,eb,ptr,col,val,heads,n,k,y,m,at::cuda::getCurrentCUDAStream());
}
extern "C" int glm53_draft_add_norm_cuda(const void*,const void*,const float*,void*,void*,int,int,cudaStream_t);
extern "C" int rs_draft_add_norm(const void* x,const void* r,const float* w,void* z,void* rout,int rows,int d){return glm53_draft_add_norm_cuda(x,r,w,z,rout,rows,d,at::cuda::getCurrentCUDAStream());}
extern "C" int glm53_draft_head_norm_rope_cuda(const void*,const float*,const long long*,const float*,void*,int,int,cudaStream_t);
extern "C" int rs_draft_head_norm_rope(const void* y,const float* w,const long long* pos,const float* inv,void* out,int n,int heads){return glm53_draft_head_norm_rope_cuda(y,w,pos,inv,out,n,heads,at::cuda::getCurrentCUDAStream());}
extern "C" int glm53_draft_silu_mul_cuda(const void*,const void*,void*,long long,cudaStream_t);
extern "C" int rs_draft_silu_mul(const void* g,const void* u,void* out,long long n){return glm53_draft_silu_mul_cuda(g,u,out,n,at::cuda::getCurrentCUDAStream());}
extern "C" int glm53_draft_attn_dev_cuda(const void*,const void*,const void*,const void*,const void*,const long long*,float*,float*,void*,int,cudaStream_t);
extern "C" int rs_draft_attn_dev(const void* q,const void* sk,const void* sv,const void* nk,const void* nv,const long long* meta,float* part_o,float* part_ml,void* out,int kvh){
    return glm53_draft_attn_dev_cuda(q,sk,sv,nk,nv,meta,part_o,part_ml,out,kvh,at::cuda::getCurrentCUDAStream());
}
extern "C" int glm53_router_fused_cuda(const float*,const void*,int,const float*,float*,long long*,float*,int,int,int,float,cudaStream_t);
extern "C" int rs_router_fused(const float* h,const void* w,int w_bf16,const float* bias,float* partial,long long* ids,float* weights,int rows,int experts,int k,float scaling){
    return glm53_router_fused_cuda(h,w,w_bf16,bias,partial,ids,weights,rows,experts,k,scaling,at::cuda::getCurrentCUDAStream());
}
extern "C" int glm53_shared_gu_packed_f32_cuda(const void*,float*,int,int,cudaStream_t);
extern "C" int rs_shared_gu_packed_f32(const void* gu,float* out,int rows,int n){return glm53_shared_gu_packed_f32_cuda(gu,out,rows,n,at::cuda::getCurrentCUDAStream());}
extern "C" int glm53_router_fused_h_cuda(const float*,const void*,int,const float*,float*,long long*,float*,void*,int,int,int,float,cudaStream_t);
extern "C" int rs_router_fused_h(const float* h,const void* w,int w_bf16,const float* bias,float* partial,long long* ids,float* weights,void* wh,int rows,int experts,int k,float scaling){
    return glm53_router_fused_h_cuda(h,w,w_bf16,bias,partial,ids,weights,wh,rows,experts,k,scaling,at::cuda::getCurrentCUDAStream());
}
extern "C" int glm53_dsa_index_expand_rows_cuda(const int64_t*,const int64_t*,int64_t*,int,int,cudaStream_t);
extern "C" int rs_dsa_index_expand_rows(const int64_t* selected,const int64_t* pos,int64_t* out,int k,int rows){return glm53_dsa_index_expand_rows_cuda(selected,pos,out,k,rows,at::cuda::getCurrentCUDAStream());}
extern "C" int glm53_dsa_index_mask_capture_rows_cuda(const float*,const int64_t* const*,float*,int64_t*,int,int,cudaStream_t);
extern "C" int rs_dsa_index_mask_capture_rows(const float* scores,const int64_t* const* pos,float* masked,int64_t* captured,int pools,int rows){return glm53_dsa_index_mask_capture_rows_cuda(scores,pos,masked,captured,pools,rows,at::cuda::getCurrentCUDAStream());}
extern "C" int glm53_dsa_capture_rows_cuda(const int64_t* const*,int64_t*,int,cudaStream_t);
extern "C" int rs_dsa_capture_rows(const int64_t* const* pos,int64_t* captured,int rows){return glm53_dsa_capture_rows_cuda(pos,captured,rows,at::cuda::getCurrentCUDAStream());}
extern "C" int glm53_dsa_topk_rows_cuda(const float*,const int64_t*,int64_t*,int,int,int,cudaStream_t);
extern "C" int rs_dsa_topk_rows(const float* masked,const int64_t* pos,int64_t* selected,int pools,int k,int rows){return glm53_dsa_topk_rows_cuda(masked,pos,selected,pools,k,rows,at::cuda::getCurrentCUDAStream());}
extern "C" int glm53_dsa_node_append_chain_cuda(const void*,int,const float*,int,cudaStream_t);
extern "C" int rs_dsa_node_append_chain(const void* ptrs,int nodes,const float* ape,int dim){return glm53_dsa_node_append_chain_cuda(ptrs,nodes,ape,dim,at::cuda::getCurrentCUDAStream());}
extern "C" int glm53_draft_head4_tile_cuda(void*,int,int,cudaStream_t);
extern "C" int rs_draft_head4_tile(void* q4,int n,int k){return glm53_draft_head4_tile_cuda(q4,n,k,at::cuda::getCurrentCUDAStream());}
extern "C" int glm53_draft_head_int4_cuda(const void*,const void*,const float*,float*,int,int,int,cudaStream_t);
extern "C" int rs_draft_head_int4(const void* x,const void* q4,const float* s,float* y,int m,int n,int k){return glm53_draft_head_int4_cuda(x,q4,s,y,m,n,k,at::cuda::getCurrentCUDAStream());}
extern "C" int glm53_kda_gate_cuda(const float*,const void*,const void*,const float*,const void*,const void*,const float*,const float*,float*,float*,float*,float*,int,int,int,int,int,cudaStream_t);
extern "C" int rs_kda_gate(const float* x,const void* fa,const void* ga,const float* wb,const void* fb,const void* gb,const float* a_log,const float* dt_bias,float* mid,float* beta,float* decay,float* sg2,int rows,int k,int rank_a,int nb,int n){
    return glm53_kda_gate_cuda(x,fa,ga,wb,fb,gb,a_log,dt_bias,mid,beta,decay,sg2,rows,k,rank_a,nb,n,at::cuda::getCurrentCUDAStream());
}
extern "C" int glm53_kda_onorm_gate_cuda(const float*,const float*,int,const float*,float*,int,cudaStream_t);
extern "C" int rs_kda_onorm_gate(const float* o,const float* onorm,int onorm_len,const float* sg2,float* out,int groups){
    return glm53_kda_onorm_gate_cuda(o,onorm,onorm_len,sg2,out,groups,at::cuda::getCurrentCUDAStream());
}
extern "C" int glm53_kda_onorm_gate_sig_cuda(const float*,const float*,int,const float*,float*,int,cudaStream_t);
extern "C" int rs_kda_onorm_gate_sig(const float* o,const float* onorm,int onorm_len,const float* g2,float* out,int groups){
    return glm53_kda_onorm_gate_sig_cuda(o,onorm,onorm_len,g2,out,groups,at::cuda::getCurrentCUDAStream());
}
extern "C" int glm53_mhc_pre_fused_cuda(const float*,const void*,int,const float*,const float*,const float*,float*,float*,float*,float*,int,int,cudaStream_t);
extern "C" int rs_mhc_pre_fused(const float* residual,const void* fn,int fn_bf16,const float* scale,const float* base,const float* ln,float* partial,float* z,float* post,float* comb,int rows,int h){
    return glm53_mhc_pre_fused_cuda(residual,fn,fn_bf16,scale,base,ln,partial,z,post,comb,rows,h,at::cuda::getCurrentCUDAStream());
}
extern "C" int rs_mhc_sinkhorn(const float* input, float* output, int rows) {
    return glm53_sinkhorn_cuda(input, output, rows, at::cuda::getCurrentCUDAStream());
}
extern "C" int glm53_mhc_post_cuda(const float*,const float*,const float*,const float*,float*,int,int,long long,long long,long long,cudaStream_t);
extern "C" int rs_mhc_post(const float* x,const float* residual,const float* comb,const float* post,float* output,
                           int rows,int hidden,int64_t s0,int64_t s1,int64_t s2) {
    return glm53_mhc_post_cuda(x,residual,comb,post,output,rows,hidden,s0,s1,s2,at::cuda::getCurrentCUDAStream());
}
extern "C" int glm53_mhc_post_four_cuda(const float*,const float*,const float*,const float*,float*,int,int,long long,long long,long long,cudaStream_t);
extern "C" int rs_mhc_post_four(const float* x,const float* residual,const float* comb,const float* post,float* output,
                                int rows,int hidden,int64_t s0,int64_t s1,int64_t s2) {
    return glm53_mhc_post_four_cuda(x,residual,comb,post,output,rows,hidden,s0,s1,s2,at::cuda::getCurrentCUDAStream());
}
extern "C" int glm53_mhc_post_packed_cuda(const float*,const float*,const float*,const float*,float*,int,int,long long,long long,long long,int,int,cudaStream_t);
extern "C" int rs_mhc_post_packed(const float* packed,const float* residual,const float* comb,const float* post,float* output,
                                  int rows,int hidden,int64_t s0,int64_t s1,int64_t s2,int round_shared,int four_streams) {
    return glm53_mhc_post_packed_cuda(packed,residual,comb,post,output,rows,hidden,s0,s1,s2,round_shared,four_streams,at::cuda::getCurrentCUDAStream());
}
extern "C" int glm53_mhc_post_four_ar_cuda(const float*,const float*,const float*,const float*,float*,int,int,long long,long long,long long,int,int,cudaStream_t);
extern "C" int rs_mhc_post_four_ar(const float* x,const float* residual,const float* comb,const float* post,float* output,
                                   int rows,int hidden,int64_t s0,int64_t s1,int64_t s2,int packed,int round) {
    return glm53_mhc_post_four_ar_cuda(x,residual,comb,post,output,rows,hidden,s0,s1,s2,packed,round,at::cuda::getCurrentCUDAStream());
}
extern "C" int glm53_gemv_cuda(const float*,const void*,float*,int,int,int,cudaStream_t);
extern "C" int rs_dense_gemv(const float* x,const void* w,float* y,int n,int k,int round_output) {
    return glm53_gemv_cuda(x,w,y,n,k,round_output,at::cuda::getCurrentCUDAStream());
}
extern "C" int glm53_small_cuda(const float*,const void*,float*,int,int,int,int,cudaStream_t);
extern "C" int rs_dense_small(const float* x,const void* w,float* y,int m,int n,int k,int round_output) {
    return glm53_small_cuda(x,w,y,m,n,k,round_output,at::cuda::getCurrentCUDAStream());
}
extern "C" int glm53_kda_cuda(float*,const float*,const float*,const float*,const float*,const float*,float*,int,cudaStream_t);
extern "C" int rs_kda_recurrent(float* h,const float* q,const float* k,const float* v,
                                const float* beta,const float* decay,float* output,int heads) {
    return glm53_kda_cuda(h,q,k,v,beta,decay,output,heads,at::cuda::getCurrentCUDAStream());
}

extern "C" int glm53_kda_fork_cuda(const float*,float*,const float*,const float*,const float*,const float*,const float*,float*,int,cudaStream_t);
extern "C" int rs_kda_fork(const float* source,float* h,const float* q,const float* k,const float* v,
                           const float* beta,const float* decay,float* output,int heads) {
    return glm53_kda_fork_cuda(source,h,q,k,v,beta,decay,output,heads,at::cuda::getCurrentCUDAStream());
}

extern "C" int glm53_kda_sequence_cuda(float*,const float*,const float*,const float*,const float*,const float*,float*,int,int,cudaStream_t);
extern "C" int rs_kda_sequence(float* h,const float* q,const float* k,const float* v,
    const float* beta,const float* decay,float* out,int heads,int steps) {
    return glm53_kda_sequence_cuda(h,q,k,v,beta,decay,out,heads,steps,at::cuda::getCurrentCUDAStream());
}

int exl3_gemm_gr(at::Tensor const&, at::Tensor const&, at::Tensor&,
                  std::optional<at::Tensor> const&, std::optional<at::Tensor> const&,
                  std::optional<at::Tensor> const&,
                  int, bool, bool, int, Graph*);

static at::Tensor blob(const void* p, std::initializer_list<int64_t> sizes,
                       int dtype /*0=fp16,1=i16,2=fp32*/) {
    auto opts = at::TensorOptions().device(at::kCUDA);
    switch (dtype) {
        case 0: return at::from_blob(const_cast<void*>(p), sizes, opts.dtype(at::kHalf));
        case 1: return at::from_blob(const_cast<void*>(p), sizes, opts.dtype(at::kShort));
        case 3: return at::from_blob(const_cast<void*>(p), sizes, opts.dtype(at::kLong));
        default: return at::from_blob(const_cast<void*>(p), sizes, opts.dtype(at::kFloat));
    }
}

// exllamav3 v1.4.9 (quant/exl3_gemm.cuh); mcg_mult / mul1_mult act as flags (non-zero selects the codebook).
int exl3_mgemm(at::Tensor const& A, at::Tensor const& B, at::Tensor& C,
               at::Tensor const& suh, at::Tensor const& A_had, at::Tensor const& svh,
               std::optional<at::Tensor> const& indices, std::optional<at::Tensor> const& weights,
               int K, int force_shape_idx, unsigned int mcg_mult, unsigned int mul1_mult, int min_index, int max_index,
               int force_num_sms, int num_tokens = 1, std::optional<at::Tensor> const& size_n_list = {},
               std::optional<at::Tensor> const& c_ptrs = {}, std::optional<at::Tensor> const& n_stride_list = {},
               std::optional<at::Tensor> const& had_src_list = {}, int num_had_src = 0);

extern "C" {

// GLM53_MLA_PREFILL_STRIDED (L1): C[b] (m x n, ldc) = A[b] (m x k, lda) * B[b] (k x n row-major, ldb), FP32 data with TF32
// tensor-core math (as the TF32 bmm it replaces), arbitrary batch strides: reads transposed views and writes the o_proj
// layout in place, removing two full-size copies per MLA prefill chunk.
extern "C" int rs_bmm_f32_tf32(const float* a,long long lda,long long sa,const float* b,long long ldb,long long sb,float* c,long long ldc,long long sc,
    int m,int n,int k,int batch){
    try{
        const float alpha=1.f,beta=0.f;auto h=at::cuda::getCurrentCUDABlasHandle();
        auto st=cublasGemmStridedBatchedEx(h,CUBLAS_OP_N,CUBLAS_OP_N,n,m,k,&alpha,b,CUDA_R_32F,(int)ldb,sb,a,CUDA_R_32F,(int)lda,sa,&beta,
            c,CUDA_R_32F,(int)ldc,sc,batch,CUBLAS_COMPUTE_32F_FAST_TF32,CUBLAS_GEMM_DEFAULT);
        if(st!=CUBLAS_STATUS_SUCCESS){fprintf(stderr,"[shim] bmm_f32_tf32: cublas %d\n",(int)st);return 1;}
        return 0;
    }catch(const std::exception& e){fprintf(stderr,"[shim] bmm_f32_tf32: %s\n",e.what());return 1;}
}

// Row-parallel dense GEMM: retain fp32 partial sums until the collective.
// Row-major X[M,K], W[N,K], Y[M,N] is column-major W^T * X.
int rs_mm16_f32(const void* x, const void* w, void* y, int m, int n, int k) {
    try {
        const float alpha = 1.0f, beta = 0.0f;
        auto handle = at::cuda::getCurrentCUDABlasHandle();
        return static_cast<int>(cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N,
            n, m, k, &alpha, w, CUDA_R_16F, k, x, CUDA_R_16F, k,
            &beta, y, CUDA_R_32F, n, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
    } catch (const std::exception& e) {
        fprintf(stderr, "[shim] dense GEMM exception: %s\n", e.what());
        return -1;
    }
}

// Drafter BF16 row shard: retain FP32 partial sums, accepting a strided view
// during resident A/B and a materialized shard in the production load path.
int rs_bf16_partial(const void* x,const void* w,void* y,int m,int n,int k,int ldw) {
    try {
        const float alpha=1.f,beta=0.f;auto handle=at::cuda::getCurrentCUDABlasHandle();
        return static_cast<int>(cublasGemmEx(handle,CUBLAS_OP_T,CUBLAS_OP_N,n,m,k,
            &alpha,w,CUDA_R_16BF,ldw,x,CUDA_R_16BF,k,&beta,y,CUDA_R_32F,n,
            CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT));
    }catch(const std::exception& e){fprintf(stderr,"[shim] BF16 partial: %s\n",e.what());return -1;}
}

// trellis 直算 GEMM:x [R,K] fp16 → y [R,N] fp16(预分配)。
// xh [R,K] fp16 为工作区;mcg [65536] fp16 码本(设备端)。
int rs_exl3_gemm(const void* x_p, void* y_p, void* xh_p,
                 const void* tr_p, const void* suh_p, const void* svh_p,
                 int64_t rows, int64_t k, int64_t n, int64_t kt, int64_t nt) {
    try {
        auto x = blob(x_p, {rows, k}, 0);
        auto y = blob(y_p, {rows, n}, 0);
        auto xh = blob(xh_p, {rows, k}, 0);
        auto tr = blob(tr_p, {kt, nt, 64}, 1);
        auto suh = blob(suh_p, {k}, 0);   // exllamav3 约定 suh/svh 为 fp16
        auto svh = blob(svh_p, {n}, 0);
        exl3_gemm_gr(x, tr, y, suh, xh, svh, -1, true, false, 0, (Graph*)nullptr);
        return 0;
    } catch (const std::exception& e) {
        fprintf(stderr, "[shim] exl3_gemm 异常: %s\n", e.what());
        return 1;
    }
}

// ───────── mgemm 多专家合批(算力换带宽 v2:8 专家 3 调用) ─────────
// A [1,K] fp16;C [k,1,N] fp16(逐专家输出);yh [k,1,K] 工作区;
// ptrs_* 为设备端 int64 指针表;selected [k] int64;weights 可空。

// K 参数 = trellis 词组数(shape[-1]/16 = 64/16 = 4),内核模板/表索引;
// 不是专家数(expert 数由 C/indices 形状给出)。2026-09-21 实证:
// K=8(专家数)→ rel≈√2 垃圾;K=4096(in_features)→ "No kernel for GEMM shape"。
extern "C" int rs_exl3_mgemm2(
    const void* a_p, void* c_p, void* yh_p,
    const void* ptrtr_p, const void* ptrsuh_p, const void* ptrsvh_p,
    const void* sel_p, int64_t k_sel, int64_t kk, int64_t nn,
    int64_t rows_a, int64_t k_words, int64_t nt, int64_t cap) {
    (void)nt;
    try {
        auto A = blob(a_p, {rows_a, 1, kk}, 0);
        auto C = blob(c_p, {k_sel, 1, nn}, 0);
        auto YH = blob(yh_p, {k_sel, 1, kk}, 0);
        auto ptrtr = blob(ptrtr_p, {cap}, 3);
        auto ptrsuh = blob(ptrsuh_p, {cap}, 3);
        auto ptrsvh = blob(ptrsvh_p, {cap}, 3);
        auto sel = blob(sel_p, {1, k_sel}, 3);
        exl3_mgemm(A, ptrtr, C, ptrsuh, YH, ptrsvh, sel, std::nullopt,
                   (int)k_words, -1, 1u, 0u, -1, -1, 0);
        return 0;
    } catch (const std::exception& e) {
        fprintf(stderr, "[shim] exl3_mgemm 异常: %s\n", e.what());
        return 1;
    }
}

} // extern "C"

// Diagnostic entry: keep the production wrapper above unchanged, vary only C dtype.
extern "C" int rs_exl3_mgemm_probe(const void* a, void* c, void* yh,
    const void* tr, const void* sh, const void* sv, const void* sel,
    int64_t k, int64_t ki, int64_t no, int64_t rows, int64_t cap, int fp32) {
    try {
        auto A = blob(a, {rows, 1, ki}, 0);
        auto C = blob(c, {k, 1, no}, fp32 ? 2 : 0);
        auto YH = blob(yh, {k, 1, ki}, 0);
        exl3_mgemm(A, blob(tr, {cap}, 3), C, blob(sh, {cap}, 3), YH,
            blob(sv, {cap}, 3), blob(sel, {1, k}, 3), std::nullopt,
            4, -1, 1u, 0u, -1, -1, 0);
        return 0;
    } catch (const std::exception& e) {
        fprintf(stderr, "[probe] mgemm: %s\n", e.what()); return -1;
    }
}

// Diagnostic-only launch control; requires the explicitly preloaded test interposer.
extern "C" int rs_exl3_mgemm_launch_probe(const void* a, void* c, void* yh,
    const void* tr, const void* sh, const void* sv, const void* sel,
    int64_t k, int64_t ki, int64_t no, int64_t rows, int64_t cap, int fp32,
    int grid_x, int grid_z, int force_shape, int64_t* metadata, char* name, size_t name_size) {
    using Arm = int (*)(int, int);
    using Finish = int (*)(int64_t*, char*, size_t);
    auto arm = reinterpret_cast<Arm>(dlsym(RTLD_DEFAULT, "m1_launch_probe_arm"));
    auto finish = reinterpret_cast<Finish>(dlsym(RTLD_DEFAULT, "m1_launch_probe_finish"));
    if (!arm || !finish || arm(grid_x, grid_z) != 0) return -10;
    int rc = 0;
    try {
        auto A = blob(a, {rows, 1, ki}, 0);
        auto C = blob(c, {k, 1, no}, fp32 ? 2 : 0);
        auto YH = blob(yh, {k, 1, ki}, 0);
        exl3_mgemm(A, blob(tr, {cap}, 3), C, blob(sh, {cap}, 3), YH,
            blob(sv, {cap}, 3), blob(sel, {1, k}, 3), std::nullopt,
            4, force_shape, 1u, 0u, -1, -1, 0);
    } catch (const std::exception& e) {
        fprintf(stderr, "[launch-probe] mgemm: %s\n", e.what()); rc = -1;
    }
    int launches = finish(metadata, name, name_size);
    if (rc != 0) return rc;
    return launches == 1 ? 0 : -11;
}

// Load the pinned recipe artifact only when explicitly requested by the probe.
extern "C" int glm53_dsa_node_append_chain_multi_cuda(const void*,const int*,int,const float*,int,cudaStream_t);
extern "C" int rs_dsa_node_append_chain_multi(const void* ptrs,const int* nodes,int nseq,const float* ape,int dim){return glm53_dsa_node_append_chain_multi_cuda(ptrs,nodes,nseq,ape,dim,at::cuda::getCurrentCUDAStream());}
extern "C" int glm53_dsa_capture_rows64_cuda(const int64_t* const*,int64_t*,int,cudaStream_t);
extern "C" int rs_dsa_capture_rows64(const int64_t* const* pos,int64_t* captured,int rows){return glm53_dsa_capture_rows64_cuda(pos,captured,rows,at::cuda::getCurrentCUDAStream());}
extern "C" int glm53_dsa_score_multi_seq_cuda(const float*,const float*,const void*,float*,int,int,cudaStream_t);
extern "C" int rs_dsa_score_multi_seq(const float* q,const float* mixing,const void* table,float* out,int capacity,int mode){return glm53_dsa_score_multi_seq_cuda(q,mixing,table,out,capacity,mode,at::cuda::getCurrentCUDAStream());}
extern "C" int glm53_latent_fp8_store_multi_cuda(const void*,const void*,cudaStream_t);
extern "C" int rs_latent_fp8_store_multi(const void* rows,const void* table){return glm53_latent_fp8_store_multi_cuda(rows,table,at::cuda::getCurrentCUDAStream());}
extern "C" int glm53_latent_attention_fp8_multi_cuda(const float*,const void*,const long long*,float*,float*,int,int,int,int,int,cudaStream_t);
extern "C" int rs_latent_attention_fp8_multi(const float* q,const void* table,const int64_t* selected,float* out,float* scratch,int heads,int queries,int slots,int stride,int splits){
    return glm53_latent_attention_fp8_multi_cuda(q,table,reinterpret_cast<const long long*>(selected),out,scratch,heads,queries,slots,stride,splits,at::cuda::getCurrentCUDAStream());}
extern "C" int glm53_kda_conv_chain_multi_cuda(const void*,const float*,const float*,float*,int,int,cudaStream_t);
extern "C" int rs_kda_conv_chain_multi(const void* table,const float* projected,const float* weights,float* conv,int tokens,int width){
    return glm53_kda_conv_chain_multi_cuda(table,projected,weights,conv,tokens,width,at::cuda::getCurrentCUDAStream());}
extern "C" int glm53_kda_correction_chain_multi_cuda(const void*,const float*,const float*,const float*,const float*,const float*,float*,float*,int,cudaStream_t);
extern "C" int rs_kda_correction_chain_multi(const void* table,const float* q,const float* k,const float* v,const float* beta,const float* decay,float* correction,float* out,int heads){
    return glm53_kda_correction_chain_multi_cuda(table,q,k,v,beta,decay,correction,out,heads,at::cuda::getCurrentCUDAStream());}
extern "C" int glm53_kda_conv_chain_cuda(const float*,const float*,const float*,float*,float*,int,int,cudaStream_t);
extern "C" int rs_kda_conv_chain(const float* base,const float* projected,const float* weights,
    float* conv,float* states,int tokens,int width) {
    return glm53_kda_conv_chain_cuda(base,projected,weights,conv,states,tokens,width,at::cuda::getCurrentCUDAStream());
}
extern "C" int glm53_kda_recurrent_chain_cuda(const float*,const float*,const float*,const float*,const float*,const float*,float*,float*,int,int,cudaStream_t);
extern "C" int rs_kda_recurrent_chain(const float* base,const float* q,const float* k,const float* v,
    const float* beta,const float* decay,float* states,float* out,int heads,int tokens) {
    return glm53_kda_recurrent_chain_cuda(base,q,k,v,beta,decay,states,out,heads,tokens,at::cuda::getCurrentCUDAStream());
}

extern "C" int glm53_kda_correction_chain_cuda(const float*,const float*,const float*,const float*,const float*,const float*,float*,float*,int,int,cudaStream_t);
extern "C" int rs_kda_correction_chain(const float* base,const float* q,const float* k,const float* v,
    const float* beta,const float* decay,float* correction,float* out,int heads,int tokens) {
    return glm53_kda_correction_chain_cuda(base,q,k,v,beta,decay,correction,out,heads,tokens,at::cuda::getCurrentCUDAStream());
}
extern "C" int glm53_kda_correction_commit_cuda(const void*,int,int,float*,int64_t,cudaStream_t);
extern "C" int glm53_kda_correction_commit_inplace_cuda(const void*,int,int,cudaStream_t);
extern "C" int glm53_kda_conv_commit_many_cuda(const void*,int,int,int,cudaStream_t);
extern "C" int rs_kda_conv_commit_many(const void* entries,int layers,int node,int width){return glm53_kda_conv_commit_many_cuda(entries,layers,node,width,at::cuda::getCurrentCUDAStream());}
extern "C" int rs_kda_correction_commit_inplace(const void* entries,int layers,int steps){
    return glm53_kda_correction_commit_inplace_cuda(entries,layers,steps,at::cuda::getCurrentCUDAStream());
}
extern "C" int glm53_range_cache_copy_cuda(const void*,void*,const long long*,const long long*,int,int,int,int,cudaStream_t);
extern "C" int rs_range_cache_copy(const void* src,void* dst,const long long* lo,const long long* hi,int capacity,int row_vectors,int divisor,int max_rows){
    return glm53_range_cache_copy_cuda(src,dst,lo,hi,capacity,row_vectors,divisor,max_rows,at::cuda::getCurrentCUDAStream());
}
extern "C" int rs_kda_correction_commit(const void* entries,int layers,int steps,float* dst,int64_t elements) {
    return glm53_kda_correction_commit_cuda(entries,layers,steps,dst,elements,at::cuda::getCurrentCUDAStream());
}


// ───────── TP2:c10d/NCCL(M1.5;NCCL env 由启动器注入,见 env-tp2.sh)─────────

static c10::intrusive_ptr<c10d::ProcessGroupNCCL> g_pg;
static c10::intrusive_ptr<c10d::ProcessGroupNCCL> g_pg_small;

extern "C" {

// fp16 GEMM 强制 fp32 累加(默认 allow_fp16_reduced_precision_reduction=true
// 会让 4096 维 GEMV 用 fp16 累加,误差 ~1e-2 量级,翻转 argmax 的元凶之一)。
extern "C" int rs_disable_tf32() {
    at::globalContext().setAllowTF32CuBLAS(false);
    return 0;
}

extern "C" int rs_set_tf32(int enabled) {
    at::globalContext().setAllowTF32CuBLAS(enabled!=0);return 0;
}
extern "C" int rs_set_fp32_accum() {
    at::globalContext().setAllowFP16ReductionCuBLAS(false);
    at::globalContext().setAllowBF16ReductionCuBLAS(false);
    return 0;
}

// ───────── CUDA Graph(M1②:全步捕获,消 12k launch/步)─────────
static at::cuda::CUDAGraph* g_graph = nullptr;
static std::unordered_map<int64_t,std::unique_ptr<at::cuda::CUDAGraph>> g_owned_graphs;
static int64_t g_next_graph_id = 1;
static at::cuda::CUDAStreamGuard* g_sg = nullptr; // 捕获期间切到侧流(图不许在默认流上捕获)
// Capture stream ownership. PyTorch caches cuBLAS/cuBLASLt workspaces per (handle, stream) and
// CUDAGraph::reset() (run by ~CUDAGraph) calls clearCublasWorkspacesForStream(capture_stream).
// With pooled capture streams, a later graph captured on the same stream reuses the workspace
// allocated during an earlier capture; destroying the earlier graph then frees that workspace
// (and emptyCache returns it to the driver) while the later graph still writes its split-K
// partials there -> Warp Illegal Address at replay. Each capture therefore gets its own stream,
// destroyed only after its graph, so a workspace lives exactly as long as the graph using it.
static cudaStream_t g_capture_raw = nullptr;   // stream of the capture in progress / g_graph
static std::unordered_map<int64_t,cudaStream_t> g_owned_streams;
static bool g_dead_pools = false;   // a graph was destroyed since the last emptyCache
static bool own_capture_stream(){const char* e=std::getenv("GLM53_GRAPH_OWN_STREAM");return !(e&&std::strcmp(e,"0")==0);}

// GLM53_GRAPH_SHARED_WS=1: one cuBLAS/cuBLASLt workspace for every graph (like vLLM's shared capture resources)
// instead of one per capture stream (~32 MiB per graph, ~2 GiB for four serving stores). All graphs capture on one
// stream that is never destroyed, and that stream's workspaces are allocated, before each capture, from a private
// pool that is never released. Destroying a graph still clears the workspace entry (see above), but the memory only
// returns to that pool, which nothing else allocates from: surviving graphs keep a valid workspace, and the next
// capture re-registers the same free block.
static bool shared_ws(){const char* e=std::getenv("GLM53_GRAPH_SHARED_WS");return e&&std::strcmp(e,"1")==0;}

static cudaStream_t g_ws_stream=nullptr;
static c10::MempoolId_t g_ws_pool{0,0};
static void ws_prepare(){
    const auto dev=c10::cuda::current_device();
    if(!g_ws_stream){
        if(cudaStreamCreateWithFlags(&g_ws_stream,cudaStreamNonBlocking)!=cudaSuccess) throw std::runtime_error("shared capture stream");
        g_ws_pool=at::cuda::graph_pool_handle();
        c10::cuda::CUDACachingAllocator::get()->createOrIncrefPool(dev,g_ws_pool);
    }
    at::cuda::CUDAStreamGuard guard(at::cuda::getStreamFromExternal(g_ws_stream,dev));
    const cudaStream_t raw=g_ws_stream;
    c10::cuda::CUDACachingAllocator::beginAllocateToPool(dev,g_ws_pool,[raw](cudaStream_t t){return t==raw;});
    try{(void)at::cuda::getCurrentCUDABlasHandle();(void)at::cuda::getCUDABlasLtWorkspace();}
    catch(...){c10::cuda::CUDACachingAllocator::endAllocateToPool(dev,g_ws_pool);throw;}
    c10::cuda::CUDACachingAllocator::endAllocateToPool(dev,g_ws_pool);
}

// Side stream for work that may overlap the current stream (Qwen: the GDN / conv / PLE state commit runs next to the
// MTP draft chain). side_begin: the side stream waits for the current stream's work so far and becomes current;
// side_end: restores the previous current stream; side_join: the current stream waits for the side work so far.
static cudaStream_t g_side = nullptr;
static cudaEvent_t g_side_fork = nullptr, g_side_done = nullptr;
static std::optional<at::cuda::CUDAStream> g_side_prev;
extern "C" int rs_side_begin() {
    if (!g_side) {
        if (cudaStreamCreateWithFlags(&g_side, cudaStreamNonBlocking) != cudaSuccess) return 1;
        cudaEventCreateWithFlags(&g_side_fork, cudaEventDisableTiming);
        cudaEventCreateWithFlags(&g_side_done, cudaEventDisableTiming);
    }
    auto cur = at::cuda::getCurrentCUDAStream();
    if (cudaEventRecord(g_side_fork, cur.stream()) != cudaSuccess) return 2;
    if (cudaStreamWaitEvent(g_side, g_side_fork, 0) != cudaSuccess) return 3;
    g_side_prev = cur;
    at::cuda::setCurrentCUDAStream(at::cuda::getStreamFromExternal(g_side, cur.device_index()));
    return 0;
}
extern "C" int rs_side_end() {
    if (!g_side_prev) return 1;
    if (cudaEventRecord(g_side_done, g_side) != cudaSuccess) return 2;
    at::cuda::setCurrentCUDAStream(*g_side_prev);
    g_side_prev.reset();
    return 0;
}
extern "C" int rs_side_join() {
    if (!g_side_done) return 0;
    return cudaStreamWaitEvent(at::cuda::getCurrentCUDAStream().stream(), g_side_done, 0) == cudaSuccess ? 0 : 1;
}
// Numbered events for cross-stream ordering (Qwen prefill overlap): record on / wait from the current stream.
static cudaEvent_t g_slots[8] = {};
extern "C" int rs_ev_record(int i) {
    if (i < 0 || i >= 8) return 1;
    if (!g_slots[i] && cudaEventCreateWithFlags(&g_slots[i], cudaEventDisableTiming) != cudaSuccess) return 2;
    return cudaEventRecord(g_slots[i], at::cuda::getCurrentCUDAStream().stream()) == cudaSuccess ? 0 : 3;
}
extern "C" int rs_ev_wait(int i) {
    if (i < 0 || i >= 8) return 1;
    if (!g_slots[i]) return 0;
    return cudaStreamWaitEvent(at::cuda::getCurrentCUDAStream().stream(), g_slots[i], 0) == cudaSuccess ? 0 : 3;
}
// Wait for the current stream only (cudaDeviceSynchronize would also wait for the side stream).
extern "C" int rs_stream_sync() { return cudaStreamSynchronize(at::cuda::getCurrentCUDAStream().stream()) == cudaSuccess ? 0 : 1; }

// Return unused cached blocks to the driver (load-time temporaries, e.g. C12 encoding); live tensors stay valid.
extern "C" void rs_empty_cache() {c10::cuda::CUDACachingAllocator::emptyCache();}
static int graph_begin_in(c10::MempoolId_t pool);
extern "C" int rs_graph_begin() { return graph_begin_in(c10::MempoolId_t{0, 0}); }
// Capture into a memory pool shared by every graph captured with the same key (Qwen decode graphs: they replay one
// after another on one stream, so temporaries of one graph may reuse those of another; tensors read across graphs
// must stay alive on the caller side).
static std::unordered_map<int64_t, c10::MempoolId_t> g_key_pools;
extern "C" int rs_graph_begin_pool(int64_t key) {
    auto it = g_key_pools.find(key);
    // (the caller keeps one graph of the pool alive for the process: a pool whose graphs are all gone cannot be reused)
    if (it == g_key_pools.end()) it = g_key_pools.emplace(key, at::cuda::graph_pool_handle()).first;
    return graph_begin_in(it->second);
}
static int graph_begin_in(c10::MempoolId_t pool) {
    try {
        if (g_graph) { delete g_graph; g_graph = nullptr; g_dead_pools = true; }
        if (g_capture_raw) { cudaStreamSynchronize(g_capture_raw); cudaStreamDestroy(g_capture_raw); g_capture_raw = nullptr; }
        // Low-level CUDAGraph capture does not perform Python's graph-context
        // cache cleanup. Retired private pools otherwise accumulate across
        // captures, exhausting GB10 unified memory before cudaMalloc fails.
        // Only unused blocks are released; live tensors/weights remain valid.
        // Release retired private pools. A pool only becomes freeable when its graph is destroyed, so by
        // default emptyCache runs only if a graph was dropped since the last one; an unconditional call on
        // every capture also returned live-workload cached blocks (prefill workspaces) to the driver and
        // forced a device-synchronizing cudaFree + later re-cudaMalloc on each capture.
        // GLM53_GRAPH_EMPTY_CACHE: "dead" (default) | "always" (old behaviour) | "0" (never).
        { const char* e=std::getenv("GLM53_GRAPH_EMPTY_CACHE");
          const bool never=e&&std::strcmp(e,"0")==0, always=e&&std::strcmp(e,"always")==0;
          if(!never && (always || g_dead_pools)) { c10::cuda::CUDACachingAllocator::emptyCache(); g_dead_pools=false; } }
        cudaStream_t raw = nullptr;
        if (shared_ws()) {
            ws_prepare();
            raw = g_ws_stream;   // not owned by the graph: g_capture_raw stays null, the stream is never destroyed
        } else if (own_capture_stream()) {
            if (cudaStreamCreateWithFlags(&raw, cudaStreamNonBlocking) != cudaSuccess) { fprintf(stderr, "[shim] capture stream create failed\n"); return 1; }
            g_capture_raw = raw;
        }
        at::cuda::CUDAStream s = raw ? at::cuda::getStreamFromExternal(raw, c10::cuda::current_device()) : at::cuda::getStreamFromPool();
        g_sg = new at::cuda::CUDAStreamGuard(s);
        g_graph = new at::cuda::CUDAGraph();
        g_graph->capture_begin(pool, cudaStreamCaptureModeThreadLocal);
        return 0;
    } catch (const std::exception& e) {
        fprintf(stderr, "[shim] graph_begin 异常: %s\n", e.what());
        // Undo partial state so a failed begin leaves no guard/graph/stream behind.
        delete g_sg; g_sg = nullptr;
        if (g_graph) { delete g_graph; g_graph = nullptr; g_dead_pools = true; }
        if (g_capture_raw) { cudaStreamDestroy(g_capture_raw); g_capture_raw = nullptr; }
        return 1;
    }
}

extern "C" int rs_graph_end() {
    try {
        g_graph->capture_end();
        delete g_sg; g_sg = nullptr; // 恢复默认流
        return 0;
    } catch (const std::exception& e) {
        fprintf(stderr, "[shim] graph_end 异常: %s\n", e.what());
        return 1;
    }
}

extern "C" int rs_graph_replay() {
    try {
        g_graph->replay();
        return 0;
    } catch (const std::exception& e) {
        fprintf(stderr, "[shim] graph_replay 异常: %s\n", e.what());
        return 1;
    }
}

// Detach completed capture from the legacy single-graph slot. Owned graph
// handles coexist with future captures; IDs are monotonic, never raw pointers.
extern "C" int64_t rs_graph_take() {
    if (!g_graph || g_sg) return 0;
    int64_t id=g_next_graph_id++;
    g_owned_graphs.emplace(id,std::unique_ptr<at::cuda::CUDAGraph>(g_graph));g_graph=nullptr;
    if (g_capture_raw) { g_owned_streams.emplace(id,g_capture_raw); g_capture_raw=nullptr; }
    return id;
}
extern "C" int rs_graph_replay_owned(int64_t id) {
    try {auto it=g_owned_graphs.find(id);if(it==g_owned_graphs.end())return 1;it->second->replay();return 0;}
    catch(const std::exception& e){fprintf(stderr,"[shim] owned replay: %s\n",e.what());return 1;}
}
extern "C" int rs_graph_drop_owned(int64_t id) {
    // Graph first (its reset clears the workspace keyed by this stream), then the stream itself.
    const int rc = g_owned_graphs.erase(id)==1?0:1;
    if (rc == 0) g_dead_pools = true;
    auto it = g_owned_streams.find(id);
    if (it != g_owned_streams.end()) { cudaStreamSynchronize(it->second); cudaStreamDestroy(it->second); g_owned_streams.erase(it); }
    return rc;
}

// True while the current stream is being captured into a CUDA graph. Caches that create device
// tensors on a miss assert this is false: an entry created during a capture lives in that graph's
// private pool and other graphs would bake its address.
extern "C" int rs_is_capturing() {
    cudaStreamCaptureStatus st = cudaStreamCaptureStatusNone;
    if (cudaStreamIsCapturing(at::cuda::getCurrentCUDAStream().stream(), &st) != cudaSuccess) return -1;
    return st == cudaStreamCaptureStatusNone ? 0 : 1;
}
// The caller's current CUDA stream (prefix cache: the writer thread waits on an event recorded on it).
extern "C" void* rs_current_stream() { return (void*)at::cuda::getCurrentCUDAStream().stream(); }
extern "C" int rs_graph_destroy() {
    if (g_graph) { delete g_graph; g_graph = nullptr; g_dead_pools = true; }
    if (g_capture_raw) { cudaStreamSynchronize(g_capture_raw); cudaStreamDestroy(g_capture_raw); g_capture_raw = nullptr; }
    return 0;
}

// Telemetry: owned graphs alive, and the cuBLAS workspace size PyTorch gives each (handle, stream).
// Bytes still allocated (live tensors) in the memory pool of the graph just captured. With graph pools shared across
// graphs this must be 0 after every capture: a graph may only leave temporaries in the pool (other graphs reuse them).
extern "C" int64_t rs_graph_pool_live_bytes() {
    if (!g_graph) return -1;
    const auto pool = g_graph->pool();
    const auto snap = c10::cuda::CUDACachingAllocator::get()->snapshot(pool, false);
    int64_t live = 0;
    for (const auto& seg : snap.segments) if (seg.owner_private_pool_id == pool) live += (int64_t)seg.allocated_size;
    return live;
}
extern "C" int64_t rs_graph_count() { return (int64_t)g_owned_graphs.size() + (g_graph ? 1 : 0); }
extern "C" int64_t rs_cublas_workspace_bytes() { return (int64_t)at::cuda::getChosenWorkspaceSize(); }
extern "C" void rs_cuda_memory(int64_t* allocated, int64_t* reserved) {
    auto stats = c10::cuda::CUDACachingAllocator::getDeviceStats(0);
    *allocated = stats.allocated_bytes[0].current;
    *reserved = stats.reserved_bytes[0].current;
}

extern "C" int glm53_rdma_ar_init(int);
extern "C" int glm53_rdma_big_init(int);
extern "C" unsigned long long glm53_rdma_big_max_bytes();
extern "C" int glm53_rdma_big_all_gather(const void*,void*,unsigned long long,cudaStream_t);
extern "C" int glm53_rdma_big_reduce_scatter(const void*,void*,unsigned long long,cudaStream_t);
static bool g_rdma_big_ready=false;
static bool rdma_big_on(){const char* on=std::getenv("GLM53_RDMA_BIG");return g_rdma_big_ready&&on&&std::strcmp(on,"1")==0;}
extern "C" size_t glm53_rdma_ar_max_bytes();
extern "C" int glm53_rdma_ar_cuda(float*,long long,cudaStream_t);
static bool g_rdma_ready = false;
extern "C" int rs_pg_init(int rank, int world, const char* master_addr, int port) {
    try {
        c10d::TCPStoreOptions so;
        so.port = (uint16_t) port;
        so.isServer = (rank == 0);
        so.numWorkers = world;
        so.waitWorkers = true;
        auto store = c10::make_intrusive<c10d::TCPStore>(std::string(master_addr), so);
        auto opts = c10d::ProcessGroupNCCL::Options::create();
        opts->timeout = std::chrono::seconds(std::getenv("GLM53_NCCL_TIMEOUT_S")?std::atoi(std::getenv("GLM53_NCCL_TIMEOUT_S")):180);
        g_pg = c10::make_intrusive<c10d::ProcessGroupNCCL>(store, rank, world, opts);
        // 暖场:一次小张量 allreduce 触发 NCCL 建连
        auto t = at::zeros({8}, at::TensorOptions().device(at::kCUDA).dtype(at::kFloat));
        std::vector<at::Tensor> v = {t};
        g_pg->allreduce(v)->wait();
        cudaDeviceSynchronize();
        const char* small = std::getenv("GLM53_TP_SMALL_COMM");
        if (small && std::strcmp(small, "1") == 0) {
            TORCH_CHECK(world == 2, "small-message communicator is qualified for TP2 only");
            for (const char* key : {"NCCL_MIN_CTAS", "NCCL_MAX_CTAS", "NCCL_MIN_NCHANNELS", "NCCL_MAX_NCHANNELS"})
                TORCH_CHECK(!std::getenv(key), "unset ", key, " when using the mixed-size communicator policy");
            auto small_store = c10::make_intrusive<c10d::PrefixStore>("glm53-small", store);
            auto small_opts = c10d::ProcessGroupNCCL::Options::create();
            small_opts->timeout = std::chrono::seconds(std::getenv("GLM53_NCCL_TIMEOUT_S")?std::atoi(std::getenv("GLM53_NCCL_TIMEOUT_S")):180);
            small_opts->config.minCTAs = 4;
            small_opts->config.maxCTAs = 4;
            g_pg_small = c10::make_intrusive<c10d::ProcessGroupNCCL>(small_store, rank, world, small_opts);
            g_pg_small->allreduce(v)->wait();
            cudaDeviceSynchronize();
            fprintf(stderr, "[tp-small] rank%d initialized 4-CTA group, payload <= 262144 bytes; larger payloads use automatic group\n", rank);
        }
        const char* rdma_init = std::getenv("GLM53_RDMA_AR_INIT");
        if (rdma_init && std::strcmp(rdma_init, "1") == 0) {
            TORCH_CHECK(world == 2, "lean RDMA allreduce is TP2 only");
            TORCH_CHECK(glm53_rdma_ar_init(rank) == 0, "lean RDMA allreduce init failed");
            g_rdma_ready = true;
        }
        const char* big_init = std::getenv("GLM53_RDMA_BIG_INIT");
        if (big_init && std::strcmp(big_init, "1") == 0) {
            TORCH_CHECK(world == 2, "dual-port RDMA collectives are TP2 only");
            TORCH_CHECK(glm53_rdma_big_init(rank) == 0, "dual-port RDMA collective init failed");
            g_rdma_big_ready = true;
        }
        return 0;
    } catch (const std::exception& e) {
        fprintf(stderr, "[shim] pg_init 异常: %s\n", e.what());
        return 1;
    }
}

// Release before CUDA/libtorch static destructors run at process exit.
extern "C" int rs_runtime_shutdown() {
    try {
        if (g_graph || !g_owned_graphs.empty() || g_pg || g_pg_small) cudaDeviceSynchronize();
        g_owned_graphs.clear();
        if (g_graph) { delete g_graph; g_graph = nullptr; g_dead_pools = true; }
        if (g_sg) { delete g_sg; g_sg = nullptr; }
        if (g_pg_small) { g_pg_small->shutdown(); g_pg_small.reset(); }
        if (g_pg) { g_pg->shutdown(); g_pg.reset(); }
        return 0;
    } catch (const std::exception& e) {
        fprintf(stderr, "[shim] shutdown exception: %s\n", e.what());
        return 1;
    }
}

// 原地 allreduce(SUM);wait() 只挂流依赖,不阻塞 host。
extern "C" int rs_allreduce(void* p, int64_t numel, int dtype) {
    try {
        if (g_rdma_ready && dtype == 2 && (size_t)numel * 4 <= glm53_rdma_ar_max_bytes()) {
            const char* on = std::getenv("GLM53_RDMA_AR");
            if (on && std::strcmp(on, "1") == 0)
                return glm53_rdma_ar_cuda(static_cast<float*>(p), numel, at::cuda::getCurrentCUDAStream());
        }
        auto t = blob(p, {numel}, dtype);
        std::vector<at::Tensor> v = {t};
        const char* active = std::getenv("GLM53_TP_SMALL_COMM_ACTIVE");
        const bool use_small = g_pg_small && (!active || std::strcmp(active, "1") == 0)
            && t.numel() * t.element_size() <= 262144;
        (use_small ? g_pg_small : g_pg)->allreduce(v)->wait();
        return 0;
    } catch (const std::exception& e) {
        fprintf(stderr, "[shim] allreduce 异常: %s\n", e.what());
        return 1;
    }
}

// I3 step 2: send-only lean allreduce. 0 = launched (the caller's consumer forms the sum from the peer slot),
// 1 = not eligible (caller falls back to rs_allreduce), other = error.
extern "C" int glm53_rdma_ar_send_cuda(float*,long long,cudaStream_t);
extern "C" int glm53_rdma_ar_materialize_cuda(float*,long long,int,cudaStream_t);
extern "C" int rs_ar_materialize(float* p, int64_t numel, int round_half) {
    return glm53_rdma_ar_materialize_cuda(p, numel, round_half, at::cuda::getCurrentCUDAStream());
}
extern "C" int rs_allreduce_send(void* p, int64_t numel) {
    const char* on = std::getenv("GLM53_RDMA_AR");
    if (!g_rdma_ready || !on || std::strcmp(on, "1") != 0 || (size_t)numel * 4 > glm53_rdma_ar_max_bytes()) return 1;
    const int rc = glm53_rdma_ar_send_cuda(static_cast<float*>(p), numel, at::cuda::getCurrentCUDAStream());
    return rc == 0 ? 0 : 2;
}

// Sequence-parallel prefill (GLM53_PREFILL_SP): reduce-scatter (sum) of `in` [world*numel_out] into
// `out` [numel_out] (rank r receives chunk r), and all-gather of `in` [numel_in] into `out` [world*numel_in].
extern "C" int rs_reduce_scatter(void* in, void* out, int64_t numel_out, int dtype) {
    try {
        // P5: FP32 over both RoCE ports (bitwise equal: 2-rank sum x0+x1).
        if (rdma_big_on() && dtype == 2 && (unsigned long long)numel_out * 4 <= glm53_rdma_big_max_bytes() && ((unsigned long long)numel_out * 4) % 16 == 0)
            return glm53_rdma_big_reduce_scatter(in, out, (unsigned long long)numel_out * 4, at::cuda::getCurrentCUDAStream());
        auto i = blob(in, {numel_out * 2}, dtype);auto o = blob(out, {numel_out}, dtype);
        c10d::ReduceScatterOptions opts;opts.reduceOp=c10d::ReduceOp::SUM;
        g_pg->_reduce_scatter_base(o, i, opts)->wait();return 0;
    } catch (const std::exception& e) {fprintf(stderr, "[shim] reduce_scatter: %s\n", e.what());return 1;}
}
// GLM53_PREFILL_SP_RS_SPLIT: reduce-scatter issued on NCCL's stream without making the current stream wait yet, so
// the next column part's GEMM overlaps it; rs_work_wait(h) adds the dependency (no host block). Callers keep the
// tensors alive until then. Handles are indices into a per-thread list, cleared when every one was waited on.
static thread_local std::vector<c10::intrusive_ptr<c10d::Work>> g_works;static thread_local int g_works_open=0;
extern "C" int rs_reduce_scatter_async(void* in, void* out, int64_t numel_out, int dtype) {
    try {
        auto i = blob(in, {numel_out * 2}, dtype);auto o = blob(out, {numel_out}, dtype);
        c10d::ReduceScatterOptions opts;opts.reduceOp=c10d::ReduceOp::SUM;
        g_works.push_back(g_pg->_reduce_scatter_base(o, i, opts));g_works_open++;return (int)g_works.size()-1;
    } catch (const std::exception& e) {fprintf(stderr, "[shim] reduce_scatter_async: %s\n", e.what());return -1;}
}
extern "C" int rs_work_wait(int h) {
    try {
        if(h<0||h>=(int)g_works.size()||!g_works[h])return 1;
        g_works[h]->wait();g_works[h].reset();if(--g_works_open==0)g_works.clear();return 0;
    } catch (const std::exception& e) {fprintf(stderr, "[shim] work_wait: %s\n", e.what());return 1;}
}
// GLM53_PREFILL_AG_OVERLAP: all-gather issued on NCCL's stream; rs_work_wait(h) later makes the current stream wait,
// so kernels launched in between overlap it (same bytes as rs_all_gather).
extern "C" int rs_all_gather_async(void* in, void* out, int64_t numel_in, int dtype) {
    try {
        auto i = blob(in, {numel_in}, dtype);auto o = blob(out, {numel_in * 2}, dtype);
        g_works.push_back(g_pg->_allgather_base(o, i));g_works_open++;return (int)g_works.size()-1;
    } catch (const std::exception& e) {fprintf(stderr, "[shim] all_gather_async: %s\n", e.what());return -1;}
}
extern "C" int rs_all_gather(void* in, void* out, int64_t numel_in, int dtype) {
    try {
        const int64_t esize = dtype == 2 ? 4 : (dtype == 3 ? 8 : 2);
        if (rdma_big_on() && (unsigned long long)(numel_in * esize) <= glm53_rdma_big_max_bytes() && (numel_in * esize) % 16 == 0)
            return glm53_rdma_big_all_gather(in, out, (unsigned long long)(numel_in * esize), at::cuda::getCurrentCUDAStream());
        auto i = blob(in, {numel_in}, dtype);auto o = blob(out, {numel_in * 2}, dtype);
        g_pg->_allgather_base(o, i)->wait();return 0;
    } catch (const std::exception& e) {fprintf(stderr, "[shim] all_gather: %s\n", e.what());return 1;}
}

} // extern "C"

extern "C" int glm53_latent_attention_cuda(const float*,const void*,const long long*,float*,float*,int,int,int,int,int,cudaStream_t);
extern "C" int glm53_active_cache_copy_cuda(const void*,void*,const long long*,int,int,int,cudaStream_t);
extern "C" int rs_active_cache_copy(const void* src,void* dst,const int64_t* len,int capacity,int row_vectors,int divisor) {
    return glm53_active_cache_copy_cuda(src,dst,reinterpret_cast<const long long*>(len),capacity,row_vectors,divisor,at::cuda::getCurrentCUDAStream());
}
extern "C" int rs_latent_attention(const float* q,const void* latent,const int64_t* ids,float* out,float* scratch,int heads,int queries,int slots,int stride,int splits) {
    return glm53_latent_attention_cuda(q,latent,reinterpret_cast<const long long*>(ids),out,scratch,heads,queries,slots,stride,splits,at::cuda::getCurrentCUDAStream());
}
extern "C" int glm53_visible_latent_cuda(const float*,const void*,const long long*,float*,float*,int,int,int,int,int,cudaStream_t);
extern "C" int rs_visible_latent(const float* q,const void* latent,const int64_t* pos,float* out,float* scratch,int heads,int queries,int slots,int stride,int splits) {
    return glm53_visible_latent_cuda(q,latent,reinterpret_cast<const long long*>(pos),out,scratch,heads,queries,slots,stride,splits,at::cuda::getCurrentCUDAStream());
}
extern "C" int glm53_shared_latent_cuda(const float*,const void*,const long long*,float*,int,int,int,int,int,cudaStream_t);
extern "C" int rs_shared_latent(const float* q,const void* latent,const int64_t* ids,float* out,int heads,int queries,int slots,int stride,int mode){
    return glm53_shared_latent_cuda(q,latent,reinterpret_cast<const long long*>(ids),out,heads,queries,slots,stride,mode,at::cuda::getCurrentCUDAStream());
}

void reconstruct(at::Tensor,at::Tensor,int,bool,bool);
void had_r_128(at::Tensor const&,at::Tensor const&,std::optional<at::Tensor> const&,std::optional<at::Tensor> const&,float);
// Decoded EXL3 weights (mcg codebook, 4 bits) of one expert projection: inner [k, n] fp16 (glm-moe-ab reference).
extern "C" int rs_exl3_inner(const void* tr_p,void* inner_p,int64_t k,int64_t n){
    try{reconstruct(blob(inner_p,{k,n},0),blob(tr_p,{k/16,n/16,64},1),4,true,false);return 0;}
    catch(const std::exception& e){fprintf(stderr,"[exl3-inner] %s\n",e.what());return -1;}
}
// Prefill tier: transiently reconstruct one expert projection; never expand
// the resident model or keep reconstructed weights across layers/requests.
extern "C" int rs_exl3_recon_gemm(const void* x_p,void* y_p,const void* tr_p,const void* sh_p,const void* sv_p,
    int64_t rows,int64_t k,int64_t n) {
    try {
        auto x=blob(x_p,{rows,k},0),y=blob(y_p,{rows,n},0);
        auto sh=blob(sh_p,{k},0),sv=blob(sv_p,{n},0);
        auto inner=at::empty({k,n},x.options()),xh=at::empty_like(x);
        reconstruct(inner,blob(tr_p,{k/16,n/16,64},1),4,true,false);
        had_r_128(x,xh,sh,std::nullopt,1.f);
        at::mm_out(y,xh,inner);
        had_r_128(y,y,std::nullopt,sv,1.f);
        return 0;
    }catch(const std::exception& e){fprintf(stderr,"[recon-gemm] %s\n",e.what());return -1;}
}
extern "C" int glm53_kda_warp_sequence_cuda(float*,const float*,const float*,const float*,const float*,const float*,float*,int,int,cudaStream_t);
extern "C" int rs_kda_warp_sequence(float* h,const float* q,const float* k,const float* v,const float* b,const float* d,float* out,int heads,int steps){
    return glm53_kda_warp_sequence_cuda(h,q,k,v,b,d,out,heads,steps,at::cuda::getCurrentCUDAStream());
}
extern "C" int glm53_fp8_dense_cuda(const float*,const void*,const float*,float*,float*,int,int,int,int,int,int,cudaStream_t);
extern "C" int rs_fp8_dense(const float* x,const void* w,const float* scales,float* y,int m,int n,int k,int rounded){
    const char* option=std::getenv("GLM53_FP8_WMMA");int mode=option?std::atoi(option):0;
    const char* split_option=std::getenv("GLM53_FP8_SPLITS");int selected=split_option?std::atoi(split_option):0;
    if(!(selected==0||selected==1||selected==2||selected==4||selected==8||selected==16))return int(cudaErrorInvalidValue);
    int splits=selected?selected:(mode>=2?8:1);
    auto scratch=at::empty({m>1&&mode>0&&splits>1?1LL*splits*m*n:0LL},at::TensorOptions().device(at::kCUDA).dtype(at::kFloat));
    return glm53_fp8_dense_cuda(x,w,scales,y,scratch.data_ptr<float>(),m,n,k,rounded,mode,selected,at::cuda::getCurrentCUDAStream());
}
extern "C" int glm53_fp8_bf16_cuda(const float*,const void*,const float*,float*,float*,int,int,int,int,cudaStream_t);
extern "C" int glm53_fp8_bf16_skinny_cuda(const void*,const void*,const float*,float*,int,int,int,cudaStream_t);
extern "C" int rs_fp8_bf16_skinny(const void* x,const void* w,const float* scales,float* y,int m,int n,int k){
    return glm53_fp8_bf16_skinny_cuda(x,w,scales,y,m,n,k,at::cuda::getCurrentCUDAStream());
}
extern "C" int rs_fp8_bf16(const float* x,const void* w,const float* scales,float* y,int m,int n,int k){
    const char* option=std::getenv("GLM53_DRAFT_FP8_SPLITS");int splits=option?std::atoi(option):8;
    if(!(splits==1||splits==2||splits==4||splits==8||splits==16))return int(cudaErrorInvalidValue);
    auto scratch=at::empty({m>1&&splits>1?1LL*splits*m*n:0LL},at::TensorOptions().device(at::kCUDA).dtype(at::kFloat));
    return glm53_fp8_bf16_cuda(x,w,scales,y,scratch.data_ptr<float>(),m,n,k,splits,at::cuda::getCurrentCUDAStream());
}

extern "C" int glm53_bf16w_rows_cuda(const float*,int,const void*,float*,int,int,int,cudaStream_t);
extern "C" int rs_bf16w_rows(const float* x,int ldx,const void* w,float* y,int m,int n,int k){return glm53_bf16w_rows_cuda(x,ldx,w,y,m,n,k,at::cuda::getCurrentCUDAStream());}
extern "C" int glm53_mla_commit_many_cuda(const void*,int,cudaStream_t);
extern "C" int rs_mla_commit_many(const void* entries,int n){return glm53_mla_commit_many_cuda(entries,n,at::cuda::getCurrentCUDAStream());}
extern "C" int glm53_grouped_reduce_add_cuda(const void*,const void*,const float*,float*,int,cudaStream_t);
extern "C" int rs_grouped_reduce_add(const void* x,const void* w,const float* add,float* y,int rows){
    return glm53_grouped_reduce_add_cuda(x,w,add,y,rows,at::cuda::getCurrentCUDAStream());
}
extern "C" int glm53_route_group_cuda(const long long*,int,int,int,int,long long*,int*,int*,cudaStream_t);
extern "C" int rs_route_group(const int64_t* ids,int R,int n_exp,int tile,int S,int64_t* rows64,int* rows32,int* error){
    return glm53_route_group_cuda(reinterpret_cast<const long long*>(ids),R,n_exp,tile,S,reinterpret_cast<long long*>(rows64),rows32,error,at::cuda::getCurrentCUDAStream());
}
extern "C" int glm53_grouped_reduce_cuda(const void*,const void*,float*,int,cudaStream_t);
extern "C" int rs_grouped_reduce(const void* x,const void* w,float* y,int rows){
    return glm53_grouped_reduce_cuda(x,w,y,rows,at::cuda::getCurrentCUDAStream());
}
extern "C" int glm53_draft_attention_cuda(const void*,const void*,const void*,const void*,const void*,void*,int,int,int,cudaStream_t);
extern "C" int rs_draft_attention(const void* q,const void* ck,const void* cv,const void* k,const void* v,void* out,int history,int n,int kvh){
    return glm53_draft_attention_cuda(q,ck,cv,k,v,out,history,n,kvh,at::cuda::getCurrentCUDAStream());
}
extern "C" int glm53_dsa_score_cuda(const float*,const float*,const float*,const long long*,float*,int,cudaStream_t);
extern "C" int glm53_dsa_score_tile4_cuda(const float*,const float*,const float*,const long long*,float*,int,int,cudaStream_t);
extern "C" int glm53_dsa_score_tensor_cuda(const float*,const float*,const float*,const long long*,float*,int,cudaStream_t);
extern "C" int glm53_dsa_score_multi_cuda(const float*,const float*,const float*,const long long* const*,int,float*,int,int,cudaStream_t);
extern "C" int rs_dsa_score_multi(const float* q,const float* pools,const float* mixing,const int64_t* const* pos,int t,float* out,int capacity,int mode){
    return glm53_dsa_score_multi_cuda(q,pools,mixing,reinterpret_cast<const long long* const*>(pos),t,out,capacity,mode,at::cuda::getCurrentCUDAStream());
}
extern "C" int rs_dsa_score(const float* q,const float* pools,const float* mixing,const int64_t* pos,float* out,int capacity,int mode){
    if(mode==5)return glm53_dsa_score_tensor_cuda(q,pools,mixing,reinterpret_cast<const long long*>(pos),out,capacity,at::cuda::getCurrentCUDAStream());
    if(mode>=2)return glm53_dsa_score_tile4_cuda(q,pools,mixing,reinterpret_cast<const long long*>(pos),out,capacity,mode,at::cuda::getCurrentCUDAStream());
    return glm53_dsa_score_cuda(q,pools,mixing,reinterpret_cast<const long long*>(pos),out,capacity,at::cuda::getCurrentCUDAStream());
}
extern "C" int glm53_fp8_large_cuda(const float*,const void*,const float*,float*,int,int,int,int,cudaStream_t);
extern "C" int glm53_fp8_large_reuse_cuda(const float*,const void*,const float*,float*,int,int,int,int,int,cudaStream_t);
extern "C" int rs_fp8_large(const float* x,const void* w,const float* scale,float* y,int m,int n,int k,int rounded,int mode){
    if(mode>=3)return glm53_fp8_large_reuse_cuda(x,w,scale,y,m,n,k,rounded,mode,at::cuda::getCurrentCUDAStream());
    return glm53_fp8_large_cuda(x,w,scale,y,m,n,k,rounded,at::cuda::getCurrentCUDAStream());
}
// Preserve FP32 probability/value multiplication when changing GQA GEMM shape.
// The guard is scoped to this operation; no process-wide precision toggle.
extern "C" int rs_bmm_fp32(const float* a,const float* b,float* out,const int64_t* ashape,const int64_t* astride,const int64_t* bshape,const int64_t* bstride){
    try {
        auto opt=at::TensorOptions().device(at::kCUDA).dtype(at::kFloat);
        auto aa=at::from_blob(const_cast<float*>(a),{ashape[0],ashape[1],ashape[2]},{astride[0],astride[1],astride[2]},opt);
        auto bb=at::from_blob(const_cast<float*>(b),{bshape[0],bshape[1],bshape[2]},{bstride[0],bstride[1],bstride[2]},opt);
        auto y=at::from_blob(out,{ashape[0],ashape[1],bshape[2]},opt);
        at::NoTF32Guard precise;at::bmm_out(y,aa,bb);return 0;
    }catch(const std::exception& e){fprintf(stderr,"[GQA FP32] %s\n",e.what());return -1;}
}
extern "C" int glm53_grouped_swiglu_cuda(const void*,const void*,void*,int,float,cudaStream_t);
extern "C" int rs_grouped_swiglu(const void* gate,const void* up,void* out,int count,float limit){
    return glm53_grouped_swiglu_cuda(gate,up,out,count,limit,at::cuda::getCurrentCUDAStream());
}
extern "C" int glm53_fp8_epilogue_cuda(float*,const float*,int,int,int,cudaStream_t);
extern "C" int rs_fp8_epilogue(float* y,const float* scale,int rows,int cols,int rounded){
    return glm53_fp8_epilogue_cuda(y,scale,rows,cols,rounded,at::cuda::getCurrentCUDAStream());
}
extern "C" int glm53_fp8_big_cuda(const void*,const void*,const float*,float*,int,int,int,int,int,int,cudaStream_t);
extern "C" int rs_fp8_big(const void* x,const void* w,const float* scale,float* y,int m,int n,int k,int ldy,int rounded,int stages){
    return glm53_fp8_big_cuda(x,w,scale,y,m,n,k,ldy,rounded,stages,at::cuda::getCurrentCUDAStream());
}
extern "C" int glm53_spin_ns_cuda(long long,cudaStream_t);
extern "C" int glm53_l2_prefetch_cuda(const void*,long long,cudaStream_t);
extern "C" int rs_spin_ns(long long ns){return glm53_spin_ns_cuda(ns,at::cuda::getCurrentCUDAStream());}
extern "C" int rs_l2_prefetch(const void* p,long long bytes){return glm53_l2_prefetch_cuda(p,bytes,at::cuda::getCurrentCUDAStream());}
// Benchmark telemetry only; called outside timed intervals on device zero.
extern "C" int rs_memory_stats(int reset,int64_t* out){
    try {
        if(reset)c10::cuda::CUDACachingAllocator::resetPeakStats(0);
        auto s=c10::cuda::CUDACachingAllocator::getDeviceStats(0);
        size_t free=0,total=0;auto rc=cudaMemGetInfo(&free,&total);if(rc!=cudaSuccess)return int(rc);
        out[0]=s.allocated_bytes[0].current;out[1]=s.allocated_bytes[0].peak;
        out[2]=s.reserved_bytes[0].current;out[3]=s.reserved_bytes[0].peak;
        out[4]=s.num_alloc_retries;out[5]=s.num_ooms;out[6]=free;out[7]=total;
        return 0;
    }catch(const std::exception& e){fprintf(stderr,"[memory stats] %s\n",e.what());return -1;}
}

// Append to engine-rs/shim/shim.cpp, which already imports ATen CUDA context.
extern "C" int glm53_dsa_index_mask_cuda(const float*,const int64_t*,float*,int,cudaStream_t);
extern "C" int glm53_dsa_index_expand_cuda(const int64_t*,const int64_t*,int64_t*,int,cudaStream_t);
extern "C" int rs_dsa_index_mask(const float* scores,const int64_t* pos,float* out,int pools) {
    return glm53_dsa_index_mask_cuda(scores,pos,out,pools,at::cuda::getCurrentCUDAStream());
}
extern "C" int rs_dsa_index_expand(const int64_t* selected,const int64_t* pos,int64_t* out,int k) {
    return glm53_dsa_index_expand_cuda(selected,pos,out,k,at::cuda::getCurrentCUDAStream());
}

// Append to shim.cpp; its existing includes provide int64_t and CUDA context.
extern "C" int glm53_draft_selector_cuda(const float*,const int64_t*,int64_t*,int,cudaStream_t);
extern "C" int rs_draft_selector(const float* edges,const int64_t* ids,int64_t* path,int steps) {
    return glm53_draft_selector_cuda(edges,ids,path,steps,at::cuda::getCurrentCUDAStream());
}

// Append only when integrating the separate candidate into shim.cpp.
extern "C" int glm53_draft_conv_cuda(const void*,const void*,const void*,void*,int,int,int64_t,int64_t,int64_t,cudaStream_t);
extern "C" int rs_draft_conv(const void* x,const void* delta,const void* base,void* out,
    int n,int side,int64_t ds0,int64_t ds1,int64_t ds2) {
    return glm53_draft_conv_cuda(x,delta,base,out,n,side,ds0,ds1,ds2,at::cuda::getCurrentCUDAStream());
}

// Native shared GU epilogue. stages is null on every production call.
extern "C" int glm53_shared_gu_cuda(const void*,const void*,void*,float*,int,cudaStream_t);
extern "C" int rs_shared_gu(const void* g,const void* u,void* out,float* stages,int count) {
    return glm53_shared_gu_cuda(g,u,out,stages,count,at::cuda::getCurrentCUDAStream());
}

extern "C" int glm53_shared_gu_packed_cuda(const void*,void*,int,int,cudaStream_t);
extern "C" int rs_shared_gu_packed(const void* gu,void* out,int rows,int n){return glm53_shared_gu_packed_cuda(gu,out,rows,n,at::cuda::getCurrentCUDAStream());}

// Captures pre-append length into independently owned storage while masking.
extern "C" int glm53_dsa_index_mask_capture_cuda(const float*,const int64_t*,float*,int64_t*,int,cudaStream_t);
extern "C" int rs_dsa_index_mask_capture(const float* scores,const int64_t* pos,float* out,int64_t* captured_pos,int pools) {
    return glm53_dsa_index_mask_capture_cuda(scores,pos,out,captured_pos,pools,at::cuda::getCurrentCUDAStream());
}

// P2a (GLM53_PREFILL_EXPERT_STREAMS=N): independent prefill experts run on a
// small stream pool. Fork records the main stream; every pool stream waits on
// it; join makes the main stream wait on every pool stream. Tensors shared
// across streams (input, packed routes, result) are owned by the caller and
// outlive the join; per-expert temporaries are allocated on their own stream.
namespace {
std::vector<c10::cuda::CUDAStream> g_pool;
c10::optional<c10::cuda::CUDAStream> g_main;
}
extern "C" int rs_stream_fork(int n){
    if(n<1||n>8)return int(cudaErrorInvalidValue);
    auto main=c10::cuda::getCurrentCUDAStream();g_main=main;
    while((int)g_pool.size()<n)g_pool.push_back(c10::cuda::getStreamFromPool(false));
    cudaEvent_t ev;if(cudaEventCreateWithFlags(&ev,cudaEventDisableTiming)!=cudaSuccess)return -1;
    cudaEventRecord(ev,main.stream());
    for(int i=0;i<n;++i)cudaStreamWaitEvent(g_pool[i].stream(),ev,0);
    cudaEventDestroy(ev);
    return int(cudaGetLastError());
}
extern "C" int rs_stream_set(int i){
    if(!g_main)return -1;
    c10::cuda::setCurrentCUDAStream(i<0?*g_main:g_pool.at(i));return 0;
}
extern "C" int rs_stream_join(int n){
    if(!g_main)return -1;
    for(int i=0;i<n;++i){cudaEvent_t ev;cudaEventCreateWithFlags(&ev,cudaEventDisableTiming);
        cudaEventRecord(ev,g_pool[i].stream());cudaStreamWaitEvent(g_main->stream(),ev,0);cudaEventDestroy(ev);}
    c10::cuda::setCurrentCUDAStream(*g_main);g_main.reset();
    return int(cudaGetLastError());
}
extern "C" int glm53_kda_decay_rows_cuda(const float*,const float*,const float*,float*,long long,int,cudaStream_t);
extern "C" int rs_kda_decay_rows(const float* g1,const float* a,const float* dt,float* out,long long total,int heads){
    return glm53_kda_decay_rows_cuda(g1,a,dt,out,total,heads,at::cuda::getCurrentCUDAStream());
}

// Storage version counter of a tensor (in-place writes bump it); used by the
// large-row Half input cache to prove a cached conversion is still current.
extern "C" int64_t rs_tensor_version(const void* t){return reinterpret_cast<const at::Tensor*>(t)->_version();}
extern "C" int glm53_shared_swiglu_hh_cuda(const void*,const void*,void*,long long,float,cudaStream_t);
extern "C" int rs_shared_swiglu_hh(const void* g,const void* u,void* out,long long n,float lim){
    return glm53_shared_swiglu_hh_cuda(g,u,out,n,lim,at::cuda::getCurrentCUDAStream());}
extern "C" int glm53_kda_conv_silu_l2_h_cuda(const float*,const void*,const void*,const void*,const float*,float*,float*,float*,int,int,float,cudaStream_t);
extern "C" int rs_kda_conv_silu_l2_h(const float* base,const void* p0,const void* p1,const void* p2,const float* wall,float* q,float* k,float* v,int t,int h,float scale){
    return glm53_kda_conv_silu_l2_h_cuda(base,p0,p1,p2,wall,q,k,v,t,h,scale,at::cuda::getCurrentCUDAStream());}
extern "C" int glm53_kda_decay_rows_h_cuda(const void*,const float*,const float*,float*,long long,int,cudaStream_t);
extern "C" int rs_kda_decay_rows_h(const void* g1,const float* a,const float* dt,float* out,long long total,int heads){
    return glm53_kda_decay_rows_h_cuda(g1,a,dt,out,total,heads,at::cuda::getCurrentCUDAStream());}
extern "C" int glm53_kda_onorm_gate_sig_hh_cuda(const float*,const float*,int,const void*,void*,int,cudaStream_t);
extern "C" int rs_kda_onorm_gate_sig_hh(const float* o,const float* onorm,int len,const void* g2,void* out,int groups){
    return glm53_kda_onorm_gate_sig_hh_cuda(o,onorm,len,g2,out,groups,at::cuda::getCurrentCUDAStream());}
extern "C" int glm53_kda_conv_silu_l2_cuda(const float*,const float*,const float*,float*,float*,float*,int,int,float,cudaStream_t);
extern "C" int rs_kda_conv_silu_l2(const float* base,const float* projected,const float* wall,float* q,float* k,float* v,int t,int h,float scale){
    return glm53_kda_conv_silu_l2_cuda(base,projected,wall,q,k,v,t,h,scale,at::cuda::getCurrentCUDAStream());
}
extern "C" int glm53_mhc_post_pre_fused_cuda(const float*,const float*,const float*,const float*,float*,const void*,int,const float*,const float*,const float*,float*,float*,float*,float*,int,int,cudaStream_t);
extern "C" int rs_mhc_post_pre_fused(const float* x,const float* residual,const float* comb,const float* post,float* out,const void* fn,int fn_bf16,const float* scale,const float* base,const float* ln,float* partial,float* z,float* post_next,float* comb_next,int rows,int h){
    return glm53_mhc_post_pre_fused_cuda(x,residual,comb,post,out,fn,fn_bf16,scale,base,ln,partial,z,post_next,comb_next,rows,h,at::cuda::getCurrentCUDAStream());
}

extern "C" int glm53_latent_attention_fp8_cuda(const float*,const void*,const long long*,float*,float*,int,int,int,int,int,cudaStream_t);
extern "C" int rs_latent_attention_fp8(const float* q,const void* latent,const int64_t* ids,float* out,float* scratch,int heads,int queries,int slots,int stride,int splits){
    return glm53_latent_attention_fp8_cuda(q,latent,reinterpret_cast<const long long*>(ids),out,scratch,heads,queries,slots,stride,splits,at::cuda::getCurrentCUDAStream());
}
extern "C" int glm53_latent_fp8_store_cuda(const void*,int,const long long*,long long,void*,cudaStream_t);
extern "C" int rs_latent_fp8_store(const void* rows,int n,const int64_t* pos_dev,int64_t pos_host,void* latent){
    return glm53_latent_fp8_store_cuda(rows,n,reinterpret_cast<const long long*>(pos_dev),pos_host,latent,at::cuda::getCurrentCUDAStream());
}
extern "C" int glm53_dsa_prefill_scores_cuda(const float*,const float*,const float*,float*,int,int,cudaStream_t);
extern "C" int glm53_dsa_prefill_scores2_cuda(const float*,const float*,const float*,float*,int,int,long long,cudaStream_t);
extern "C" int rs_dsa_prefill_scores_masked(const float* q,const float* mixing,const float* pools,float* out,int n,int active,int64_t first_pos){
    return glm53_dsa_prefill_scores2_cuda(q,mixing,pools,out,n,active,first_pos,at::cuda::getCurrentCUDAStream());
}
extern "C" int rs_dsa_prefill_scores(const float* q,const float* mixing,const float* pools,float* out,int n,int active){
    const char* v=std::getenv("GLM53_DSA_PREFILL_SCORE_TILED");
    if(v&&v[0]=='1')return glm53_dsa_prefill_scores2_cuda(q,mixing,pools,out,n,active,-1,at::cuda::getCurrentCUDAStream());
    return glm53_dsa_prefill_scores_cuda(q,mixing,pools,out,n,active,at::cuda::getCurrentCUDAStream());
}
extern "C" int glm53_gumbel_candidates_cuda(const long long*,const float*,const long long*,float*,int,cudaStream_t);
extern "C" int rs_gumbel_candidates(const int64_t* ids,const float* temps,const int64_t* keys,float* out,int positions){
    return glm53_gumbel_candidates_cuda((const long long*)ids,temps,(const long long*)keys,out,positions,at::cuda::getCurrentCUDAStream());
}
extern "C" int glm53_gumbel_noise_cuda(float*,long long,int,int,long long,const float*,const unsigned long long*,cudaStream_t);
extern "C" int rs_gumbel_noise(float* out,int64_t stride,int rows,int cols,int64_t col_offset,const float* temps,const uint64_t* keys){
    return glm53_gumbel_noise_cuda(out,stride,rows,cols,col_offset,temps,reinterpret_cast<const unsigned long long*>(keys),at::cuda::getCurrentCUDAStream());
}

extern "C" int glm53_gumbel_params_cuda(float*,unsigned long long*,int,const float*,const unsigned long long*,cudaStream_t);
extern "C" int rs_gumbel_params(float* temps,uint64_t* keys,int rows,const float* t,const uint64_t* k){
    return glm53_gumbel_params_cuda(temps,reinterpret_cast<unsigned long long*>(keys),rows,t,reinterpret_cast<const unsigned long long*>(k),at::cuda::getCurrentCUDAStream());
}
extern "C" int glm53_gumbel_argmax_packet_cuda(const float*,int,int,long long,const float*,const unsigned long long*,float*,int*,float*,int,int,cudaStream_t);
extern "C" int rs_gumbel_argmax_packet(const float* values,int rows,int width,int64_t start,const float* temps,const uint64_t* keys,float* part_v,int* part_i,float* packet,int world,int rank){
    return glm53_gumbel_argmax_packet_cuda(values,rows,width,start,temps,reinterpret_cast<const unsigned long long*>(keys),part_v,part_i,packet,world,rank,at::cuda::getCurrentCUDAStream());
}

// Profiling ranges (GLM53_NVTX=1 at the call sites): header-only NVTX3, no-ops without a profiler.
extern "C" void rs_nvtx_push(const char* name){nvtxRangePushA(name);}
extern "C" void rs_nvtx_pop(){nvtxRangePop();}
