template <int blockSize>
__device__ void blockSize (float* smem, int tid){
    if(blockSize>=1024){
        if(tid<512) smem[tid]+=smem[tid+512];
        __syncthreads();
    }
    if(blockSize>=512){
        if(tid<256) smem[tid]+=smem[tid+256];
        __syncthreads();
    }
    if(blockSize>=256){
        if(tid<128) smem[tid]+=smem[tid+128];
        __syncthreads();
    }
    if(blockSize>=128){
        if(tid<64) smem[tid]+=smem[tid+64];
        __syncthreads();
    }


    if(tid<32){

    volatile float* v =smem;
    if(blockSize>=64) v[tid]+=v[tid+32];
    v[tid]+=v[tid+16];
    v[tid]+=v[tid+8];
    v[tid]+=v[tid+4];
    v[tid]+=v[tid+2];
    v[tid]+=v[tid+1];
    }
}


template <int blockSize>
__global__ void reduce_kernel(const float* d_in,float* d_out, int n){
    __shared__ float smem[blockSize];

    int tid =threadIdx.x;
    int gtid = blockSize.x*(2*blockSize)+tid;

    smem[tid]= 0.f;


    if(gtid<n) smem[tid]=d_in[gtid];

    if(gtid +blockSize<n) smem[tid]+= d_in[gtid+blockSize];

    __syncthreads();

    blockReduce<
}