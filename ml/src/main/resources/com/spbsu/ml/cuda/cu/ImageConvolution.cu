extern "C"
__global__ void convolution(
        int *original,
        int *kernel,
        int *blurred,
        int kernelRadius,
        int originalWidth,
        int mirrorWidth
) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;

  const int kernelSize = kernelRadius * 2 + 1;

  int kernelValue = 0;
  int imageValue = 0;
  int sumR = 0;
  int sumG = 0;
  int sumB = 0;
  int r = 0;
  int g = 0;
  int b = 0;

  for(int i = 0; i <= kernelSize; i++) {
    for(int j = 0; j <= kernelSize; j++) {
      kernelValue = kernel[i * kernelSize + j];
      imageValue = original[(y + i) * mirrorWidth + (x + j)];

      r = ((imageValue >> 16) & 0xFF) * kernelValue;
      g = ((imageValue >> 8) & 0xFF) * kernelValue;
      b = ((imageValue >> 0) & 0xFF) * kernelValue;

      sumR += r;
      sumG += g;
      sumB += b;
    }
  }
  sumR = (sumR << 8) + sumG;
  sumR = (sumR << 8) + sumB;
  blurred[y * originalWidth + x] = sumR;
}