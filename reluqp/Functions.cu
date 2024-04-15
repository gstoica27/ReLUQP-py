#include <stdlib.h>
#include <stdio.h>
#include <cfloat>
#include <cuda_runtime.h>
#include <vector>
#include <curand.h>
#include <curand_kernel.h>
#include <float.h>
#include <math.h>
#include <cusolverDn.h>
#include <cuda_runtime.h>
#include <iostream>



// Function to concatenate two matrices in C
int** concatenate_matrices(int** mat1, int rows1, int cols1, int** mat2, int rows2, int cols2, int dim, int* out_rows, int* out_cols) {
    int **result;

    if (dim == 1) { // Horizontal concatenation
        if (rows1 != rows2) {
            return NULL; // Incompatible dimensions
        }
        *out_rows = rows1;
        *out_cols = cols1 + cols2;
        result = (int**)malloc(*out_rows * sizeof(int*));
        for (int i = 0; i < *out_rows; i++) {
            result[i] = (int*)malloc(*out_cols * sizeof(int));
            for (int j = 0; j < cols1; j++) {
                result[i][j] = mat1[i][j];
            }
            for (int j = 0; j < cols2; j++) {
                result[i][j + cols1] = mat2[i][j];
            }
        }
    } else if (dim == 0) { // Vertical concatenation
        if (cols1 != cols2) {
            return NULL; // Incompatible dimensions
        }
        *out_rows = rows1 + rows2;
        *out_cols = cols1;
        result = (int**)malloc(*out_rows * sizeof(int*));
        for (int i = 0; i < rows1; i++) {
            result[i] = (int*)malloc(*out_cols * sizeof(int));
            for (int j = 0; j < *out_cols; j++) {
                result[i][j] = mat1[i][j];
            }
        }
        for (int i = 0; i < rows2; i++) {
            result[i + rows1] = (int*)malloc(*out_cols * sizeof(int));
            for (int j = 0; j < *out_cols; j++) {
                result[i + rows1][j] = mat2[i][j];
            }
        }
    } else {
        return NULL; // Invalid dimension
    }

    return result;
}


// Function to free dynamically allocated matrix
void free_matrix(int** matrix, int rows) {
    for (int i = 0; i < rows; i++) {
        free(matrix[i]);
    }
    free(matrix);
}




// Direction: 0 for vertical, 1 for horizontal
__global__ void concatenateMatricesKernel(int* mat1, int* mat2, int* result, int rows1, int cols1, int rows2, int cols2, int totalRows, int totalCols, int direction) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (direction == 1) { // Horizontal concatenation
        if (row < rows1 && col < totalCols) {
            if (col < cols1) {
                result[row * totalCols + col] = mat1[row * cols1 + col];
            } else if (col >= cols1) {
                result[row * totalCols + col] = mat2[row * cols2 + (col - cols1)];
            }
        }
    } else { // Vertical concatenation
        if (col < cols1 && row < totalRows) {
            if (row < rows1) {
                result[row * cols1 + col] = mat1[row * cols1 + col];
            } else if (row >= rows1) {
                result[row * cols1 + col] = mat2[(row - rows1) * cols2 + col];
            }
        }
    }
}

void printMatrix(const char* desc, int* matrix, int rows, int cols) {
    printf("%s:\n", desc);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%d ", matrix[i * cols + j]);
        }
        printf("\n");
    }
}


/* Example of concatenation
int main() {
    // Example for horizontal concatenation
    const int rows1 = 2, cols1 = 3, rows2 = 2, cols2 = 2;
    int h_mat1[rows1 * cols1] = {1, 2, 3, 4, 5, 6};
    int h_mat2[rows2 * cols2] = {7, 8, 9, 10};
    int totalRows = rows1 + rows2, totalCols = cols1 + cols2;
    int* h_result = new int[totalRows * totalCols];

    int *d_mat1, *d_mat2, *d_result;
    cudaMalloc(&d_mat1, rows1 * cols1 * sizeof(int));
    cudaMalloc(&d_mat2, rows2 * cols2 * sizeof(int));
    cudaMalloc(&d_result, totalRows * totalCols * sizeof(int));

    cudaMemcpy(d_mat1, h_mat1, rows1 * cols1 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_mat2, h_mat2, rows2 * cols2 * sizeof(int), cudaMemcpyHostToDevice);

    // Configure dimensions for horizontal concatenation
    dim3 blockDim(16, 16);
    dim3 gridDim((cols1 + cols2 + 15) / 16, (rows1 + 15) / 16);
    concatenateMatricesKernel<<<gridDim, blockDim>>>(d_mat1, d_mat2, d_result, rows1, cols1, rows2, cols2, rows1, cols1 + cols2, 1);
    cudaDeviceSynchronize();

    cudaMemcpy(h_result, d_result, rows1 * (cols1 + cols2) * sizeof(int), cudaMemcpyDeviceToHost);
    printMatrix("Horizontal Concatenation Result", h_result, rows1, cols1 + cols2);

    // Setup for vertical concatenation
    gridDim = dim3((cols1 + 15) / 16, (rows1 + rows2 + 15) / 16);
    concatenateMatricesKernel<<<gridDim, blockDim>>>(d_mat1, d_mat2, d_result, rows1, cols1, rows2, cols2, rows1 + rows2, cols1, 0);
    cudaDeviceSynchronize();

    cudaMemcpy(h_result, d_result, (rows1 + rows2) * cols1 * sizeof(int), cudaMemcpyDeviceToHost);
    printMatrix("Vertical Concatenation Result", h_result, rows1 + rows2, cols1);

    cudaFree(d_mat1);
    cudaFree(d_mat2);
    cudaFree(d_result);
    delete[] h_result;

    return 0;
}

*/







__global__ void blockArgmin(const float* array, int* minIndices, float* minValues, int n) {
    extern __shared__ float sdata[];
    int* sidx = (int*)&sdata[blockDim.x];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    float minVal = FLT_MAX;
    int minIdx = -1;

    // Each thread processes one element
    if (i < n) {
        minVal = array[i];
        minIdx = i;
        for (int offset = blockDim.x / 2; offset > 0; offset /= 2) {
            if (i + offset < n && array[i + offset] < minVal) {
                minVal = array[i + offset];
                minIdx = i + offset;
            }
        }
    }

    sdata[tid] = minVal;
    sidx[tid] = minIdx;
    __syncthreads();

    // Reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (sdata[tid + s] < sdata[tid]) {
                sdata[tid] = sdata[tid + s];
                sidx[tid] = sidx[tid + s];
            }
        }
        __syncthreads();
    }

    // Write the result for this block to global memory
    if (tid == 0) {
        minIndices[blockIdx.x] = sidx[0];
        minValues[blockIdx.x] = sdata[0];
    }
}




/*  Example for block Argmin nvcc Functions.cu -o executable -lcurand

int main() {
    const int size = 1024;  // Array size
    float *d_array, *minValues;
    int *minIndices;

    cudaMalloc(&d_array, size * sizeof(float));
    cudaMalloc(&minIndices, (size + 255) / 256 * sizeof(int));
    cudaMalloc(&minValues, (size + 255) / 256 * sizeof(float));

    // Setup CURAND
    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen, 1234ULL);  // Set random seed

    // Generate random numbers
    curandGenerateUniform(gen, d_array, size);

    // Launch kernel
    blockArgmin<<<(size + 255) / 256, 256, 256 * sizeof(float) * 2>>>(d_array, minIndices, minValues, size);
    cudaDeviceSynchronize();

    // Copy results back to host
    std::vector<int> h_minIndices((size + 255) / 256);
    std::vector<float> h_minValues((size + 255) / 256);
    cudaMemcpy(h_minIndices.data(), minIndices, h_minIndices.size() * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_minValues.data(), minValues, h_minValues.size() * sizeof(float), cudaMemcpyDeviceToHost);

    // Final reduction on the host
    int globalMinIndex = h_minIndices[0];
    float globalMinValue = h_minValues[0];
    for (int i = 1; i < h_minValues.size(); i++) {
        if (h_minValues[i] < globalMinValue) {
            globalMinValue = h_minValues[i];
            globalMinIndex = h_minIndices[i];
        }
    }

    printf("Minimum value: %f found at index %d\n", globalMinValue, globalMinIndex);

    // Cleanup
    cudaFree(d_array);
    cudaFree(minIndices);
    cudaFree(minValues);
    curandDestroyGenerator(gen);  // Destroy the generator

    return 0;
}
*/



int argmin(const float *array, int n) {
    if (array == NULL || n <= 0) return -1; // Error handling for invalid input

    int minIndex = 0;
    float minValue = FLT_MAX;

    for (int i = 0; i < n; i++) {
        if (array[i] < minValue) {
            minValue = array[i];
            minIndex = i;
        }
    }

    return minIndex;
}



int argmax(const float *array, int n) {
    if (array == NULL || n <= 0) return -1; // Error handling for invalid input

    int maxIndex = 0;
    float maxValue = FLT_MIN; // Initialize to the smallest possible float value

    for (int i = 0; i < n; i++) {
        if (array[i] > maxValue) {
            maxValue = array[i];
            maxIndex = i;
        }
    }

    return maxIndex;
}




__global__ void blockArgmax(const float* array, int* maxIndices, float* maxValues, int n) {
    extern __shared__ float sdata[];
    int* sidx = (int*)&sdata[blockDim.x];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    float maxVal = -FLT_MAX;  // Initialize to the smallest possible float value
    int maxIdx = -1;

    if (i < n) {
        maxVal = array[i];
        maxIdx = i;
        for (int offset = blockDim.x / 2; offset > 0; offset /= 2) {
            if (i + offset < n && array[i + offset] > maxVal) {
                maxVal = array[i + offset];
                maxIdx = i + offset;
            }
        }
    }

    sdata[tid] = maxVal;
    sidx[tid] = maxIdx;
    __syncthreads();

    // Perform reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (sdata[tid + s] > sdata[tid]) {
                sdata[tid] = sdata[tid + s];
                sidx[tid] = sidx[tid + s];
            }
        }
        __syncthreads();
    }

    // Write result for this block to global mem
    if (tid == 0) {
        maxIndices[blockIdx.x] = sidx[0];
        maxValues[blockIdx.x] = sdata[0];
    }
}

/* Max of two scalars already exist same for Min 
int main() {
    double num1 = 3.5;
    double num2 = 2.7;

    printf("The maximum of %.2f and %.2f is %.2f\n", num1, num2, max(num1, num2));
    return 0;
}

*/


__device__ float atomicMaxFloat(float* address, float val) {
    int* address_as_i = (int*) address;
    int old = *address_as_i, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_i, assumed, __float_as_int(fmaxf(val, __int_as_float(assumed))));
    } while (assumed != old);
    return __int_as_float(old);
}

// CUDA Kernel to find the inf norm (maximum absolute value in the array)
__global__ void maxNormKernel(float* array, float* maxNorm, int n) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int local_tid = threadIdx.x;

    // Load shared mem from global mem
    float myNumber = (tid < n) ? fabsf(array[tid]) : 0;
    sdata[local_tid] = myNumber;
    __syncthreads();

    // Perform reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (local_tid < s) {
            sdata[local_tid] = fmaxf(sdata[local_tid], sdata[local_tid + s]);
        }
        __syncthreads();
    }

    // Write result for this block to global mem
    if (local_tid == 0) {
        atomicMaxFloat(maxNorm, sdata[0]);
    }
}


/* Example of norm inf computation

int main() {
    float h_array[] = {1.0, -2.0, 3.0, -4.0, 5.0, 13.0, -14};
    int n = sizeof(h_array) / sizeof(h_array[0]);
    float *d_array, *d_maxNorm;
    float h_maxNorm = 0;

    cudaMalloc(&d_array, n * sizeof(float));
    cudaMalloc(&d_maxNorm, sizeof(float));
    cudaMemcpy(d_array, h_array, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_maxNorm, &h_maxNorm, sizeof(float), cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    maxNormKernel<<<blocksPerGrid, threadsPerBlock, threadsPerBlock * sizeof(float)>>>(d_array, d_maxNorm, n);
    cudaDeviceSynchronize();

    cudaMemcpy(&h_maxNorm, d_maxNorm, sizeof(float), cudaMemcpyDeviceToHost);
    printf("Maximum norm of the array is: %f\n", h_maxNorm);

    cudaFree(d_array);
    cudaFree(d_maxNorm);
    return 0;
}

*/


// In regular C can use sqrt(num)
__global__ void computeSqrt(float *input, float *result, int n) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < n) {
        result[index] = sqrt(input[index]);
    }
}






