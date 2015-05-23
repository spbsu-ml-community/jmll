extern "C"
__global__ void fSigmoid(
    const float* arguments,
    float* results,
    const long size
) {
  const int X = gridDim.x;
  const int index = gridDim.y * X * threadIdx.x + X * blockIdx.y + blockIdx.x;

  if(index < size) {
    results[index] = 1.f / (1.f + expf(-arguments[index]));
  }
}

extern "C"
__global__ void fDerSigmoid(
    const float* arguments,
    float* results,
    const long size
) {
  const int X = gridDim.x;
  const int index = gridDim.y * X * threadIdx.x + X * blockIdx.y + blockIdx.x;

  if(index < size) {
    const float argument = arguments[index];
    results[index] = argument - argument * argument;
  }
}

extern "C"
__global__ void fExp(
    const float* arguments,
    float* results,
    const long size
) {
  const int X = gridDim.x;
  const int index = gridDim.y * X * threadIdx.x + X * blockIdx.y + blockIdx.x;

  if(index < size) {
    results[index] = expf(arguments[index]);
  }
}

extern "C"
__global__ void fTanh(
    const float* arguments,
    float* results,
    const long size
) {
  const int X = gridDim.x;
  const int index = gridDim.y * X * threadIdx.x + X * blockIdx.y + blockIdx.x;

  if(index < size) {
    results[index] = tanh(arguments[index]);
  }
}

extern "C"
__global__ void fNegation(
    const float* arguments,
    float* results,
    const long size
) {
  const int X = gridDim.x;
  const int index = gridDim.y * X * threadIdx.x + X * blockIdx.y + blockIdx.x;

  if(index < size) {
    results[index] = -arguments[index];
  }
}

extern "C"
__global__ void fHadamard(
    const float* argumentsA,
    const float* argumentsB,
    float* results,
    const long size
) {
  const int X = gridDim.x;
  const int index = gridDim.y * X * threadIdx.x + X * blockIdx.y + blockIdx.x;

  if(index < size) {
    results[index] = argumentsA[index] * argumentsB[index];
  }
}
