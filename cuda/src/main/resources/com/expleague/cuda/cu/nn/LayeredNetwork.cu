
__device__ float sigmoid(float x) {
    return 1 / (1 + expf(-x));
}

extern "C"
#define BLOCK_DIM 4
#define SWINDOW_DIM 8
#define WDATA_DIM 16
__global__ void produceState(const float* arguments, const int argsSize, const float* weights, 
                                const int* topology, const int topSize, float* outStates) {
    const int tid = threadIdx.x;
    const int dim = argsSize + topSize;  
    //extern __shared__ float s[];
    //float* states = s;
    //bool* ready = (bool*)&states[dim];
    //extern __shared__ bool ready[]; 
    __shared__ int counter[BLOCK_DIM];
    __shared__ float swindow[SWINDOW_DIM];
    __shared__ int tdata[BLOCK_DIM * 3];
    __shared__ float wdata[WDATA_DIM];
    int totalCount = argsSize;
    int offset = 0;

    for (int i = tid; i < argsSize; i += blockDim.x) {
        swindow[i] = arguments[i];
    }        
    counter[tid] = 0;
    __syncthreads();

    /*

    while(totalCount < dim) {
        const int t = totalCount - argsSize;
        int topCount = blockDim.x * 3;
        if (t + blockDim.x > topSize) {
            topCount = (topSize - t) * 3;
        }
        for (int i = tid; i < topCount; i += blockDim.x) {
            tdata[i] = topology[t * 3 + i]
        }

        const int topIndex = t + tid;
        if (topIndex < topSize) {
            const int leftBorder = tdata[tid];
            const int rightBorder = tdata[tid + 1];
            const int weightsStart = tdata[tid + 2];

            if (rightBorder <= counter[0]) {
                float sum = 0;
                for (int i = leftBorder; i < rightBorder; i++) {
                    sum += outStates[i] * weights[weightsStart + i - leftBorder];
                }

                outStates[index] = sigmoid(sum);
                ready[index] = true;
            }
        }
        __syncthreads();

        if (tid == 0) {
            int total = counter[0];
            for (int i = total; i < total + blockDim.x && i < dim; i++) {
                if (ready[i]) {
                    counter[0]++;
                }
            }
        } 
        __syncthreads();
    }
    */

    //int n = tid;
    //while(n < dim) {
    //    outStates[n] = states[n];
    //    n += blockDim.x;
    //}
}


extern "C"
__global__ void produceState3(const float* arguments, const int argsSize, const float* weights, 
                                const int* topology, const int topSize, float* outStates) {
    const int tid = threadIdx.x;
    const int dim = argsSize + topSize;  
    extern __shared__ float s[];
    float* states = s;
    int* iters = (int*)&states[dim];      

    iters[tid] = 0;
    int r = tid;
    while (r < argsSize) {
        states[r] = arguments[r];
        iters[tid]++;
        r += blockDim.x;
    }
    __syncthreads();

    while(iters[tid] * blockDim.x + tid < dim) {
        const int index = iters[tid] * blockDim.x + tid;
        const int topIndex = index - argsSize;
        const int leftBorder = topology[topIndex*3];
        const int rightBorder = topology[topIndex*3 + 1];
        const int weightsStart = topology[topIndex*3 + 2];

        bool canStart = true;
        for (int i = leftBorder; i < rightBorder; i++) {
            int threadId = i % blockDim.x;
            int mustCounted = i / blockDim.x + 1;
            if (iters[threadId] < mustCounted) {
                canStart = false;
                break;
            }
        }

        if (canStart) {
            float sum = 0;
            for (int i = leftBorder; i < rightBorder; i++) {
                sum += states[i] * weights[weightsStart + i - leftBorder];
            }
            states[index] = sigmoid(sum);
            iters[tid]++;
        }
        __syncthreads();
    }

    __syncthreads();

    int n = tid;
    while(n < dim) {
        outStates[n] = states[n];
        n += blockDim.x;
    }
}