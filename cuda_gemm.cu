
#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <sstream>
#include <vector>

#define BASE_THREAD_NUM 256
#define BASE_THREAD_NUM_2D 16

#define VOLUME 4
#define BLOCK_TILE_ROW 2
#define BLOCK_TILE_COL 1

#define TILE 4
#define S (BASE_THREAD_NUM_2D * TILE)
#define L (BLOCK_TILE_COL * TILE)
typedef float scalar_t;
const size_t ELEM_SIZE = sizeof(scalar_t);

struct CudaArray {
  CudaArray(const size_t size) {
    cudaError_t err = cudaMalloc(&ptr, size * ELEM_SIZE);
    if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
    this->size = size;
  }
  ~CudaArray() { cudaFree(ptr); }
  size_t ptr_as_int() { return (size_t)ptr; }
  
  scalar_t* ptr;
  size_t size;
};

struct CudaDims {
  dim3 block, grid;
};

CudaDims CudaOneDim(size_t size) {
  /**
   * Utility function to get cuda dimensions for 1D call
   */
  CudaDims dim;
  size_t num_blocks = (size + BASE_THREAD_NUM - 1) / BASE_THREAD_NUM;
  dim.block = dim3(BASE_THREAD_NUM, 1, 1);
  dim.grid = dim3(num_blocks, 1, 1);
  return dim;
}

CudaDims CudaTwoDim(size_t size_x, size_t size_y) {
  CudaDims dim;
  size_t num_blocks_x = (size_x + BASE_THREAD_NUM_2D - 1) / BASE_THREAD_NUM_2D;
  size_t num_blocks_y = (size_y + BASE_THREAD_NUM_2D - 1) / BASE_THREAD_NUM_2D;
  dim.block = dim3(BASE_THREAD_NUM_2D, BASE_THREAD_NUM_2D, 1);
  dim.grid = dim3(num_blocks_x, num_blocks_y, 1);
  return dim;
}

#define MAX_VEC_SIZE 8
struct CudaVec {
  uint32_t size;
  int32_t data[MAX_VEC_SIZE];
};

CudaVec VecToCuda(const std::vector<int32_t>& x) {
  CudaVec shape;
  if (x.size() > MAX_VEC_SIZE) throw std::runtime_error("Exceeded CUDA supported max dimesions");
  shape.size = x.size();
  for (size_t i = 0; i < x.size(); i++) {
    shape.data[i] = x[i];
  }
  return shape;
}


__global__ void MatmulKernel(const scalar_t *A, const scalar_t *B, scalar_t *out, uint32_t M, uint32_t N, uint32_t P) {
    // buffer in shared memory
    __shared__ float sA[S][L];
    __shared__ float sB[S][L]; // transpose shared buffer of matrix B
    // buffer in local memory
    float c[TILE][TILE] = {0};
    float a[TILE], b[TILE]; 
    size_t block_x = blockIdx.x;
    size_t block_y = blockIdx.y;
    for (int k = 0; k < N; k += L) {
        __syncthreads();
        // cooperative fetching
        int nthreads = blockDim.x * blockDim.y;
        int tid = blockDim.x * threadIdx.y + threadIdx.x;
        size_t item_id = tid; 
        while (item_id < S * L) {
            int x = item_id / L;
            int y = item_id % L;
            sA[x][y] = *(A + (block_x * S + x) * N + (k + y));
            sB[x][y] = *(B + (y + k) * P + (block_y * S + x));
          item_id += nthreads;
        }
        __syncthreads();
        
        for (int ki = 0; ki < L; ki++) {
            // copy to thread local vector
            for (int j = 0; j < TILE; j++) a[j] = sA[threadIdx.x * TILE + j][ki];
            for (int j = 0; j < TILE; j++) b[j] = sB[threadIdx.y * TILE + j][ki];
            // outer dots
            for (int x = 0; x < TILE; x++) {
                for (int y = 0; y < TILE; y++) {
                    c[x][y] += a[x] * b[y];
                }
            }
        }
    }

    size_t x_base = block_x * blockDim.x * TILE + threadIdx.x * TILE;
    size_t y_base = block_y * blockDim.y * TILE + threadIdx.y * TILE;

    // copy to out matrix if in range
    for (int x = 0; x < TILE; x++) {
        for (int y = 0; y < TILE; y++) {
            if (x_base + x < M && y_base + y < P)
            *(out + (x_base + x) * P + (y_base + y)) = c[x][y];
        }
    }
}

int main() {
    // init test matrix
    const size_t M = 128;
    const size_t N = 128;
    const size_t P = 128;
    
    scalar_t A[M][N], B[N][P], C[M][P];
    size_t counter = 0;
    // init A with 1
    for (size_t i = 0; i < M; i++) {
        for (size_t j = 0; j < N; j++) {
            // A[i][j] = 1;
            A[i][j] = i * N + j;
            // printf("%f ", A[i][j]);
        }
    }
    // init B with 1 to N*P
    for (size_t i = 0; i < N; i++) {
        for (size_t j = 0; j < P; j++) {
            B[i][j] = i * P + j;
            // B[i][j] = 1;
            counter++;
        }
    }

    // copy to device
    CudaArray dA(M * N), dB(N * P), dC(M * P);
    cudaMemcpy(dA.ptr, A, M * N * ELEM_SIZE, cudaMemcpyHostToDevice);
    cudaMemcpy(dB.ptr, B, N * P * ELEM_SIZE, cudaMemcpyHostToDevice);
    // set out to 0
    cudaMemset(dC.ptr, 0, M * P * ELEM_SIZE);

    // call kernel
    CudaDims dim = CudaTwoDim(M, P);
    MatmulKernel<<<dim.grid, dim.block>>>(dA.ptr, dB.ptr, dC.ptr, M, N, P);

    // print result
    cudaMemcpy(C, dC.ptr, M * P * ELEM_SIZE, cudaMemcpyDeviceToHost);
    for (size_t i = 0; i < M; i++) {
        if (i < M / 2) continue;
        for (size_t j = 0; j < P; j++) {
            if (j < P / 2) continue;
            std::cout << C[i][j] << " ";
        }
        std::cout << std::endl;
    }
    return 0;
    
}