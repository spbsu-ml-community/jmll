extern "C"
__global__ void fMatrixExp(
    const float* arguments,
    float* results,
    const int states
) {
  const int X = gridDim.x;
  const int col = gridDim.y * X * threadIdx.x + X * blockIdx.y + blockIdx.x;

  if (col < states) {
    float sum = 0;
    for (int j = 0; j < states - 1; j++) {
      results[col * states + j] = expf(arguments[col * (states - 1) + j]);
      sum = sum + results[col * states + j];
    }
    sum = sum + 1;
    results[col * states + states - 1] = 1;
    for (int j = 0; j < states; j++) {
      results[col * states + j] = results[col * states + j] / sum;
    }
  }
}

extern "C"
#define BLOCK_DIM 1024
__global__ void fMatrixReduce(
    const float* arguments,
    float* results
) {
  const int col = blockIdx.x;
  const int states = blockDim.x;
  const int tid = threadIdx.x;
  const int index = states * col + tid;
  __shared__ float sdata[BLOCK_DIM];
  __shared__ float res[BLOCK_DIM];
  if (tid < (states - 1)) {
    const float f = expf(arguments[col * (states - 1) + tid]);
    sdata[tid] = f;
    res[tid] = f;
  } else {
    sdata[tid] = 1;
    res[tid] = 1;
  }

  __syncthreads();

  for (int s = BLOCK_DIM / 2; s > 0; s>>=1) {
    if (tid < s) {
        sdata[tid] += sdata[tid + s];
    }
    __syncthreads();
  }

  results[index] = res[tid] / sdata[0];
}

extern "C"
#define BLOCK_SIZE 32
__global__ void reduce5(const float* arguments, float* results, const int n) {
    extern __shared__ float sdata[];
    const int tid = threadIdx.x;
    const int i = blockIdx.x*blockDim.x + tid;

    if (i < n) {
        sdata[tid] = arguments[i];
    }

    for (int s = BLOCK_SIZE / 2; s > 0; s>>=1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        results[blockIdx.x] = sdata[0];
    }
}


extern "C"
__global__ void fFill(
    float* arguments,
    const float value,
    const int size
) {
  const int index = blockDim.x * blockIdx.x + threadIdx.x;

  if(index < size) {
    arguments[index] = value;
  }
}

extern "C"
__global__ void fMatrixKernel1(
    const int states,
    const float weight,
    const float diff,
    const float* distribution,
    const float* expectedValue,
    float* betaGrad,
    const int to,
    const float* weights
) {
  const int index = blockDim.x * blockIdx.x + threadIdx.x;

  if (index < states * (states - 1)) {
    const int i = index / (states - 1);
    const int j = index % (states - 1);

    const float curW = weights[i * states + to];
    const float grad = 2 * weight * diff * distribution[i] * expectedValue[to];

    if (j == to) {
      betaGrad[index] += grad * curW * (1 - curW);
    } else {
      betaGrad[index] += -grad * curW * weights[i * states + j];
    }
  }
}


extern "C"
__global__ void fMatrixKernel2(
    const int states,
    const float lambda,
    float* betaGrad,
    const int to,
    const float* weights
) {
  const int index = blockDim.x * blockIdx.x + threadIdx.x;

  if (index < (states - 1) * states) {
    const int from = index / (states - 1);
    const int j = index % (states - 1);

    const float curW = weights[from * states + to];
    const float grad = lambda * curW;

    if (j == to) {
      betaGrad[index] += grad * curW * (1 - curW);
    } else {
      betaGrad[index] += -grad * curW * weights[from * states + j];
    }
  }
}

extern "C"
__global__ void fVectorKernel1(
    const float* lastGrad,
    const float* gradCoordinate,
    const float* totalGrad,
    const float step,
    const int sumSize,
    float* result,
    const int size
) {
  const int index = blockDim.x * blockIdx.x + threadIdx.x;

  if(index < size && lastGrad[index] != 0) {
      result[index] += -step * gradCoordinate[index] * totalGrad[index] / sumSize;
  }
}

/*
extern "C"
#define STATES 6
#define SIZE 15
__global__ void getSeqValue(
    const float* params,
    const float* seq,
    const int len,
    float result
) {
  const int tid = threadIdx.x;
  const int dim = STATES * (STATES - 1) * SIZE + STATES;

  __shared__ float sdata[dim];
  __shared__ float res[BLOCK_DIM];
  if (tid < dim) {
    sdata[tid] = params[tid];
  }
  __syncthreads();

  for (int i = 0; i < len; i++) {
    const int offset = seq[i] * STATES * (STATES - 1);
    if (tid >= offset && tid < offset + STATES * (STATES - 1)) {
      const int row = (tid - offset) / STATES;

    }
    __syncthreads();
  }

  if (tid < (states - 1)) {
    for (int s = 1; s < states - 1; s *= 2) {
      if (tid % (2 * s) == 0) {
        sdata[tid] += sdata[tid + s];
      }
     __syncthreads();
    }
  }

  const float sum = sdata[0] + 1;
  results[index] = res[tid] / sum;
}*/