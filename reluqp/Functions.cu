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
#include <windows.h>
//#include <sys/time.h>


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






double vector_inf_norm(double* vec, int length) {
    double max_val = 0.0;
    for (int i = 0; i < length; i++) {
        double abs_val = fabs(vec[i]);
        if (abs_val > max_val) {
            max_val = abs_val;
        }
    }
    return max_val;
}

void matrix_vector_mult(double* A, double* B, double* C, int m, int n) {
    for (int i = 0; i < m; i++) {
        C[i] = 0.0;
        for (int j = 0; j < n; j++) {
            C[i] += A[i * n + j] * B[j];
        }
    }
}

void compute_residuals(
    double** H, double** A, double* g, double* x, double* z, double* lam, 
    double* rho, double rho_min, double rho_max, int nx, int nc, 
    double* primal_res, double* dual_res
) {
    double *t1 = (double*)malloc(nc * sizeof(double));
    double *t2 = (double*)malloc(nx * sizeof(double));
    double *t3 = (double*)malloc(nx * sizeof(double));
    double *temp_primal = (double*)malloc(nc * sizeof(double));
    double *temp_dual = (double*)malloc(nx * sizeof(double));

    matrix_vector_mult(*A, x, t1, nc, nx);
    matrix_vector_mult(*H, x, t2, nx, nx);

    for (int i = 0; i < nx; i++) {
        t3[i] = 0.0;
        for (int j = 0; j < nc; j++) {
            t3[i] += A[j][i] * lam[j];
        }
    }

    for (int i = 0; i < nc; i++) {
        temp_primal[i] = t1[i] - z[i];
    }
    *primal_res = vector_inf_norm(temp_primal, nc);

    for (int i = 0; i < nx; i++) {
        temp_dual[i] = t2[i] + t3[i] + g[i];
    }
    *dual_res = vector_inf_norm(temp_dual, nx);

    double numerator = *primal_res / fmax(vector_inf_norm(t1, nc), vector_inf_norm(z, nc));
    double denom = *dual_res / fmax(fmax(vector_inf_norm(t2, nx), vector_inf_norm(t3, nx)), vector_inf_norm(g, nx));

    *rho = fmax(fmin(sqrt(numerator / denom) * (*rho), rho_max), rho_min);

    free(t1);
    free(t2);
    free(t3);
    free(temp_primal);
    free(temp_dual);
}


/* Example of Compute residuals
int main() {
    int nx = 3;
    int nc = 2;

    double** H = (double**)malloc(nx * sizeof(double*));
    double** A = (double**)malloc(nc * sizeof(double*));
    double* g = (double*)malloc(nx * sizeof(double));
    double* x = (double*)malloc(nx * sizeof(double));
    double* z = (double*)malloc(nc * sizeof(double));
    double* lam = (double*)malloc(nc * sizeof(double));
    double rho = 1.0, rho_min = 0.1, rho_max = 10.0;

    for (int i = 0; i < nx; i++) {
        H[i] = (double*)malloc(nx * sizeof(double));
        for (int j = 0; j < nx; j++) {
            H[i][j] = (i == j) ? 2.0 : 0.0;
        }
        g[i] = 1.0;
        x[i] = 1.0;
    }
    for (int i = 0; i < nc; i++) {
        A[i] = (double*)malloc(nx * sizeof(double));
        for (int j = 0; j < nx; j++) {
            A[i][j] = 1.0;
        }
        z[i] = 1.5;
        lam[i] = 0.5;
    }

    double primal_res, dual_res;

    compute_residuals(H, A, g, x, z, lam, &rho, rho_min, rho_max, nx, nc, &primal_res, &dual_res);

    printf("Primal Residual: %f\n", primal_res);
    printf("Dual Residual: %f\n", dual_res);
    printf("Updated Rho: %f\n", rho);

    for (int i = 0; i < nx; i++) {
        free(H[i]);
    }
    for (int i = 0; i < nc; i++) {
        free(A[i]);
    }
    free(H);
    free(A);
    free(g);
    free(x);
    free(z);
    free(lam);

    return 0;
}

*/






/*
typedef struct  {
    time_t      tv_sec;     
    suseconds_t tv_usec;    
} timeval;

*/



typedef struct {
    bool verbose;
    bool warm_starting;
    bool scaling;
    double rho;
    double rho_min;
    double rho_max;
    double sigma;
    bool adaptive_rho;
    int adaptive_rho_interval;
    double adaptive_rho_tolerance;
    int max_iter;
    double eps_abs;
    double eq_tol;
    int check_interval;
} Settings;



typedef struct {
    int iter;
    double obj_val;
    double pri_res;
    double dua_res;
    double setup_time;
    double solve_time;
    double update_time;
    double run_time;
    double rho_estimate;
} Info;



typedef struct {
    double x;
    double z;
    Info* info;
} Results;

typedef struct {
    double** H;
    double* g;
    double** A;
    double* l;
    double* u;
    int nx;
    int nc;
    int nf;
} QP;



typedef struct
{
    QP* qp;
    Settings* settings;
    double* rhos;
    int rhos_len;
    double*** W_ks;
    double*** B_ks;
    double** b_ks;
    int clamp_left;
    int clamp_right;
} ReLU_Layer;




typedef struct
{
    Info* info;
    Results* results;
    Settings* settings;
    ReLU_Layer* layers;
    QP* qp;
    //struct timeval start;
    //struct timeval end;
    // double* x;
    // double* z;
    // double* lam;
    LARGE_INTEGER start;  // For timing the solve process
    LARGE_INTEGER end;    // For timing the solve process
    double* output;
    int rho_ind;
} ReLU_QP;








double dot_product(double* a, double* b, int n) {
    double sum = 0.0;
    for (int i = 0; i < n; i++) {
        sum += a[i] * b[i];
    }
    return sum;
}



double vector_dot(double* vec1, double* vec2, int dim) {
    double dot = 0;
    for (int i = 0; i < dim; i++) {
        dot += (vec1[i] * vec2[i]);
    }
    return dot;
}



double* create_vector(int dim) {
    double* vector = (double*)malloc(dim * sizeof(double));
    return vector;
}



void matvecmul(double** matrix, double* vector, double* result, int left, int nelem) {
    for (int i = 0; i < left; i++) {
        for (int j = 0; j < nelem; j++) {
            result[i] += matrix[i][j] * vector[j];
        }
    }
}



double compute_J(double** H, double* g, double* x, int nx) {
    double* Hx = create_vector(nx);
    matvecmul(H, x, Hx, nx, nx);
    double Hx_dot_x = vector_dot(Hx, x, nx);
    double gx = vector_dot(g, x, nx);
    return 0.5 * Hx_dot_x + gx;
}






void update_results(ReLU_QP* relu_qp, int iter, double pri_res, double dua_res, double rho_estimate) {
    // gettimeofday(&relu_qp->start, NULL);
    relu_qp->results->info->iter = iter;
    relu_qp->results->info->pri_res = pri_res;
    relu_qp->results->info->dua_res = dua_res;
    relu_qp->results->info->rho_estimate = rho_estimate;

    int nx = relu_qp->qp->nx;
    int nc = relu_qp->qp->nc;
    double* x = (double*)malloc(relu_qp->qp->nx * sizeof(double));
    for (int i = 0; i < relu_qp->qp->nx; i++) {
        x[i] = relu_qp->output[i];
    }
    double* z = (double*)malloc(relu_qp->qp->nc * sizeof(double));
    for (int i = nx; i < nx + nc; i++) {
        z[i - nx] = relu_qp->output[i];
    }
    relu_qp->results->info->obj_val = compute_J(relu_qp->qp->H, relu_qp->qp->g, x, nx);
    gettimeofday(&relu_qp->end, NULL);
    double elapsedTime = (double)(relu_qp->end.tv_sec - relu_qp->start.tv_sec) * 1000.0;
    elapsedTime += (((double)(relu_qp->end.tv_usec - relu_qp->start.tv_usec)) / 1000.0) / 1000.;
    relu_qp->results->info->run_time = elapsedTime;
    relu_qp->results->info->solve_time = relu_qp->results->info->update_time + elapsedTime;
    
    double* lam = create_vector(relu_qp->qp->nc);
    // TODO: Need to add the warm_starting check and then the clear_primal_dual function.
}












void solve(ReLU_QP* problem) {
    LARGE_INTEGER frequency;
    LARGE_INTEGER t1, t2;
    double elapsedTime;

    // Start the performance counter
    QueryPerformanceFrequency(&frequency);
    QueryPerformanceCounter(&t1);

    Settings* stng = problem->settings;
    int nx = problem->qp->nx;
    int nc = problem->qp->nc;
    double rho = problem->settings->rho;  // Starting rho, adjust as per rho_ind if needed

    for (int k = 1; k <= stng->max_iter; k++) {
        // Assuming an operation to update x, z, lam from output
        memcpy(problem->x, problem->output, nx * sizeof(double));
        memcpy(problem->z, problem->output + nx, nc * sizeof(double));
        memcpy(problem->lam, problem->output + nx + nc, nc * sizeof(double));

        // Perform computations as required
        if (k % stng->check_interval == 0) {
            double primal_res, dual_res;
            compute_residuals(problem->qp->H, problem->qp->A, problem->qp->g, problem->x, problem->z, problem->lam, &rho, stng->rho_min, stng->rho_max, nx, nc, &primal_res, &dual_res);

            // Log details if verbose
            if (stng->verbose) {
                printf("Iter: %d, rho: %.2e, res_p: %.2e, res_d: %.2e\n", k, rho, primal_res, dual_res);
            }

            // Check for convergence
            if (primal_res < stng->eps_abs && dual_res < stng->eps_abs) {
                update_results(problem->info, k, "solved", primal_res, dual_res, rho);
                break;
            }
        }
    }

    // Stop the performance counter
    QueryPerformanceCounter(&t2);
    elapsedTime = (double)(t2.QuadPart - t1.QuadPart) * 1000.0 / frequency.QuadPart;
    problem->info->solve_time = elapsedTime;
}


