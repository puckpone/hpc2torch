#include <cuda.h>
#include <cub/cub.cuh>
#include <algorithm>    // 用于std::min

template <typename T, typename Tind>
__global__ void blockGatherKernel(T const *input, Tind const *indices, T *output, int stride, int indSize)
{
    int tid = blockIdx.x % stride + (blockIdx.x - blockIdx.x % stride) * indSize; 
    
    // 添加循环展开指令（修正语法）
    #pragma unroll 4
    for (int index = threadIdx.x; index < indSize; index += blockDim.x) {
        const Tind src_offset = indices[index] * stride;
        output[tid + index * stride] = input[tid + src_offset];
    }
}

template <typename T, typename Tind>
__global__ void warpGatherKernel(T const *input, Tind const *indices, T *output, int stride, int indSize)
{
    int otherIdx = blockIdx.x * blockDim.y + threadIdx.y;
    int tid = otherIdx % stride + (otherIdx - otherIdx % stride) * indSize;
    
    // 添加循环展开指令（修正语法）
    #pragma unroll 4
    for (int index = threadIdx.x; index < indSize; index += blockDim.x) {
        const Tind src_offset = indices[index] * stride;
        output[tid + index * stride] = input[tid + src_offset];
    }
}

template <typename T, typename Tind>
void gatherLaunch(void const *input, void const *indices, void *output, int stride, int indSize, int othersize)
{
    if (indSize > 1024)
    {
        // 优化线程块配置（保证warp对齐）
        int blockDim = (stride % 32 == 0) ? std::min(1024, stride) : ((stride/32)+1)*32;
        blockDim = std::min(blockDim, 1024);
        blockGatherKernel<T, Tind>
            <<<othersize, blockDim>>>((T *)input, (Tind *)indices, (T *)output, stride, indSize);
    }
    else if (indSize > 31)
    {
        dim3 block_dim(32, 32);
        dim3 grid_dim((othersize + 31)/32, 1);
        warpGatherKernel<T, Tind>
            <<<grid_dim, block_dim>>>((T *)input, (Tind *)indices, (T *)output, stride, indSize);
    }
    else if (indSize > 15)
    {
        int BLOCK_DIM_y = std::min(256, othersize);
        dim3 block_dim(16, BLOCK_DIM_y);
        dim3 grid_dim(1);
        warpGatherKernel<T, Tind>
            <<<grid_dim, block_dim>>>((T *)input, (Tind *)indices, (T *)output, stride, indSize);
    }
    else
    {
        int BLOCK_DIM_y = std::min(256, othersize);
        dim3 block_dim(4, BLOCK_DIM_y);
        dim3 grid_dim(1);
        warpGatherKernel<T, Tind>
            <<<grid_dim, block_dim>>>((T *)input, (Tind *)indices, (T *)output, stride, indSize);
    }
}

extern "C" void gather_nv_f32(void const *input, void const *indices, void *output, int stride, int indSize, int othersize) {
    gatherLaunch<float, uint64_t>(input, indices, output, stride, indSize, othersize);
}

extern "C" void gather_nv_f16(void const *input, void const *indices, void *output, int stride, int indSize, int othersize) {
    gatherLaunch<half, uint64_t>(input, indices, output, stride, indSize, othersize);
}