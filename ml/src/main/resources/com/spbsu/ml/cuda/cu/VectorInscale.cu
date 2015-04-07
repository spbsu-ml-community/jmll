extern "C"
__global__ void dSigmoid(double *original, int size) {
  const int X = gridDim.x;
  const int index = gridDim.y * X * threadIdx.x + X * blockIdx.y + blockIdx.x;

  if(index < size) {
    original[index] = 1. / (1. + exp(-original[index]));
  }
}

extern "C"
__global__ void dExp(double *original, int size) {
  const int X = gridDim.x;
  const int index = gridDim.y * X * threadIdx.x + X * blockIdx.y + blockIdx.x;

  if(index < size) {
    original[index] = exp(original[index]);
  }
}

extern "C"
__global__ void dTanh(double *original, int size) {
  const int X = gridDim.x;
  const int index = gridDim.y * X * threadIdx.x + X * blockIdx.y + blockIdx.x;

  if(index < size) {
    original[index] = tanh(original[index]);
  }
}

extern "C"
__global__ void dRndSigmoid(double *original, double *random, int size) {
  const int X = gridDim.x;
  const int index = gridDim.y * X * threadIdx.x + X * blockIdx.y + blockIdx.x;

  if(index < size) {
    original[index] = (1. / (1. + exp(-original[index]))) > random[index];
  }
}
