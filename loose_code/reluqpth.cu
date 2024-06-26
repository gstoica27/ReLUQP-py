#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdbool.h> 
#include <signal.h>
#include <float.h>
#include <sys/time.h>


/// @brief Declare all structs ///
typedef struct {
    float** H;
    float* g;
    float** A;
    float* l;
    float* u;
    int nx;
    int nc;
    // int nf;
} QP;


QP* InitializeQP(
    float** H, float* g, float**A, float* l, float* u, int nx, int nc//, int nf
) {
    QP* qp = (QP*)malloc(sizeof(QP));
    qp->H = H;
    qp->g = g;
    qp->A = A;
    qp->l = l;
    qp->u = u;
    qp->nx = nx;
    qp->nc = nc;
    // qp->nf = nf;
    return qp;
}


typedef struct {
    bool verbose;
    bool warm_starting;
    bool scaling;
    float rho;
    float rho_min;
    float rho_max;
    float sigma;
    bool adaptive_rho;
    int adaptive_rho_interval;
    float adaptive_rho_tolerance;
    int max_iter;
    float eps_abs;
    float eq_tol;
    int check_interval;
} Settings;

Settings* InitializeSettings(
    bool verbose, 
    bool warm_starting,
    bool scaling,
    float rho,
    float rho_min,
    float rho_max,
    float sigma,
    bool adaptive_rho,
    int adaptive_rho_interval,
    float adaptive_rho_tolerance,
    int max_iter,
    float eps_abs,
    float eq_tol,
    int check_interval
) {
    Settings* settings = (Settings*)malloc(sizeof(Settings));
    settings->verbose = verbose;
    settings->warm_starting = warm_starting;
    settings->scaling = scaling;
    settings->rho = rho;
    settings->rho_min = rho_min;
    settings->rho_max = rho_max;
    settings->sigma = sigma;
    settings->adaptive_rho = adaptive_rho;
    settings->adaptive_rho_interval = adaptive_rho_interval;
    settings->adaptive_rho_tolerance = adaptive_rho_tolerance;
    settings->max_iter = max_iter;
    settings->eps_abs = eps_abs;
    settings->eq_tol = eq_tol;
    settings->check_interval = check_interval;
    return settings;
}


typedef struct {
    int iter;
    float obj_val;
    float pri_res;
    float dua_res;
    float setup_time;
    float solve_time;
    float update_time;
    float run_time;
    float rho_estimate;
} Info;


Info* InitializeInfo(
    int iter,
    float obj_val,
    float pri_res,
    float dua_res,
    float setup_time,
    float solve_time,
    float update_time,
    float run_time,
    float rho_estimate
) {
    Info* info = (Info*)malloc(sizeof(Info));
    info->iter = iter;
    info->obj_val = obj_val;
    info->pri_res = pri_res;
    info->dua_res = dua_res;
    info->setup_time = setup_time;
    info->solve_time = solve_time;
    info->update_time = update_time;
    info->run_time = run_time;
    info->rho_estimate = rho_estimate;
    return info;
}


typedef struct {
    float* x;
    float* z;
    Info* info;
} Results;

Results* InitializeResults(
    float* x,
    float* z,
    Info* info
) {
    Results* results = (Results*) malloc(sizeof(Results));
    results->x = x;
    results->z = z;
    results->info = info;
    return results;
}


/// Create variable setting functions

float** get_H(int nx) {
    float** H = (float**)malloc(nx * sizeof(float*));
    for (int i=0; i< nx; i++) {
        H[i] = (float*)malloc(nx * sizeof(float));
    }
    
    H[0][0] = 6;
    H[0][1] = 2;
    H[0][2] = 1;
    H[1][0] = 2;
    H[1][1] = 5;
    H[1][2] = 2;
    H[2][0] = 1;
    H[2][1] = 2;
    H[2][2] = 4.0;
    return H;
}

float* get_g(int nx) {
    float* g = (float*)malloc(nx * sizeof(float));
    g[0] = -8.;
    g[1] = -3.;
    g[2] = -3.;
    return g;
}

float** get_A(int nc, int nx) {
    float** A = (float**)malloc(nc * sizeof(float*));
    for (int i=0; i < nc; i++) {
        A[i] = (float*)malloc(nx * sizeof(float));
    }
    A[0][0] = 1;
    A[0][1] = 0;
    A[0][2] = 1;

    A[1][0] = 0;
    A[1][1] = 1;
    A[1][2] = 1;

    A[2][0] = 1;
    A[2][1] = 0;
    A[2][2] = 0;

    A[3][0] = 0;
    A[3][1] = 1;
    A[3][2] = 0;

    A[4][0] = 0;
    A[4][1] = 0;
    A[4][2] = 1;
    
    return A;
}

float* get_l(int nc) {
    float* l = (float*) malloc(nc * sizeof(float));
    l[0] = 3;
    l[1] = 0;
    l[2] = -10;
    l[3] = -10;
    l[4] = -10;
    return l;
}

float* get_u(int nc) {
    float* u = (float*) malloc(nc * sizeof(float));
    u[0] = 3;
    u[1] = 0;
    u[2] = INFINITY;
    u[3] = INFINITY;
    u[4] = INFINITY;
    return u;
}


float** create_matrix(int num_row, int num_col) {
    float** matrix = (float**)malloc(num_row * sizeof(float*));
    for (int i=0; i< num_row; i++) {
        matrix[i] = (float*)malloc(num_col * sizeof(float));
        for (int j = 0; j < num_col; j++) {
            matrix[i][j] = 0;
        }
    }
    return matrix;
}

float* create_vector(int dim) {
    float* vector = (float*)malloc(dim * sizeof(float));
    for (int i = 0; i < dim; i++) {
        vector[i] = 0;
    }
    return vector;
}

void vector_subtract(float* source, float* amount, float* dest, int dim) {
    for (int i = 0; i < dim; i++) {
        dest[i] = source[i] - amount[i];
    }
}

void vector_add(float* source, float* amount, float* dest, int dim) {
    for (int i = 0; i < dim; i++) {
        dest[i] = source[i] + amount[i];
    }
}


void vector_where(bool* conditional, float* vector, float when_true, float when_false, int dim) {
    for (int i = 0; i < dim; i++) {
        if (conditional[i]) {
            vector[i] = when_true;
        }
        else {
            vector[i] = when_false;
        }
    }
}

float** create_diagonal_matrix(float* vector, int dim) {
    float** matrix = create_matrix(dim, dim);
    for (int i = 0; i < dim; i++) {
        // matrix[i] = (float*)calloc(dim, sizeof(float));
        matrix[i][i] = vector[i];
    }
    return matrix;
}


void populate_diagonal_matrix(float* vector, float** matrix, int dim) {
    for (int i = 0; i < dim; i++) {
        matrix[i][i] = vector[i];
    }
}


float** create_scalar_diagonal_matrix(float w, int dim) {
    float** matrix = create_matrix(dim, dim);
    for (int i = 0; i < dim; i++) {
        // matrix[i] = (float*)calloc(dim, sizeof(float));
        matrix[i][i] = w;
    }
    return matrix;
}


void populate_diagonal_matrix_with_scalar(float w, float** matrix, int dim) {
    for (int i = 0; i < dim; i++) {
        matrix[i][i] = w;
    }
}


float** transpose_matrix(float** matrix, int num_row, int num_col) {
    float** transpose = create_matrix(num_col, num_row);
    for (int i = 0; i < num_col; i++) {
        // transpose[i] = (float*)malloc(num_row * sizeof(float));
        for (int j = 0; j < num_row; j++) {
            transpose[i][j] = matrix[j][i];
        }
    }
    return transpose;
}

void add_value_to_matrix_inplace(float** matrix, float value, int num_row, int num_col) {
    for (int i = 0; i < num_row; i++) {
        for (int j = 0; j < num_col; j++) {
            matrix[i][j] = matrix[i][j] + value;
        }
    }
}

void add_value_to_matrix(float** matrix, float** dest, float value, int num_row, int num_col) {
    for (int i = 0; i < num_row; i++) {
        for (int j = 0; j < num_col; j++) {
            dest[i][j] = matrix[i][j] + value;
        }
    }
}

float* copy_vector(float* vector, int dim) {
    float* copy = (float*)malloc(dim * sizeof(float));
    for (int i = 0; i < dim; i++) {
        copy[i] = vector[i];
    }
    return copy;

}

float** copy_matrix(float** matrix, int num_row, int num_col) {
    float** copy = create_matrix(num_row, num_col);
    for (int i = 0; i < num_row; i++) {
        copy[i] = (float*)calloc(num_col, sizeof(float));
        for (int j = 0; j < num_col; j++) {
            copy[i][j] = matrix[i][j];
        }
    }
    return copy;
}


void copy_matrix_inplace(float** matrix, float** copy, int num_row, int num_col) {
    for (int i = 0; i < num_row; i++) {
        copy[i] = (float*)calloc(num_col, sizeof(float));
        for (int j = 0; j < num_col; j++) {
            copy[i][j] = matrix[i][j];
        }
    }
}


void matmul(float** matrix1, float** matrix2, float** result, int left, int nelem, int right)
{
    for (int i = 0; i < left; i++) {
        for (int j = 0; j < right; j++) {
            result[i][j] = 0;
            for (int k = 0; k < nelem; k++) {
                result[i][j] += matrix1[i][k] * matrix2[k][j];
            }
        }
    }
}

void matvecmul(float** matrix, float* vector, float* result, int left, int nelem) {
    for (int i = 0; i < left; i++) {
        result[i] = 0;
        for (int j = 0; j < nelem; j++) {
            result[i] += matrix[i][j] * vector[j];
        }
    }
}

float vector_dot(float* vec1, float* vec2, int dim) {
    float dot = 0;
    for (int i = 0; i < dim; i++) {
        dot += (vec1[i] * vec2[i]);
    }
    return dot;
}


void add_matrices(float** A, float** B, float** result, int num_row, int num_col) {
    for (int i = 0; i < num_row; i++) {
        for (int k = 0; k < num_col; k++) {
            result[i][k] = A[i][k] + B[i][k];
        }
    }
}

void subtract_matrices(float** A, float** B, float** result, int num_row, int num_col) {
    for (int i = 0; i < num_row; i++) {
        for (int k = 0; k < num_col; k++) {
            result[i][k] = A[i][k] - B[i][k];
        }
    }
}

void scalar_multiply_matrix(float** matrix, float scalar, float** dest, int num_row, int num_col) {
    for (int i = 0; i < num_row; i++) {
        for (int j = 0; j < num_col; j++) {
            dest[i][j] = matrix[i][j] * scalar;
        }
    }
}

void flatten_matrix(float** matrix, float* dest, int num_row, int num_col) {
    for (int i = 0; i < num_row; i++) {
        for (int j = 0; j < num_col; j++) {
            dest[i*num_col + j] = matrix[i][j];
        }
    }
}

void divide_matrices(float** A, float** B, float** result, int num_row, int num_col) {
    for (int i = 0; i < num_row; i++) {
        for (int j = 0; j < num_col; j++) {
            result[i][j] = A[i][j] / B[i][j];
        }
    }
}

void divide_diag(float** A, float** B, float** result, int num_row) {
    for (int i = 0; i < num_row; i++) {
        result[i][i] = A[i][i] / B[i][i];
    }
}

// //////////////////////////////////////////////////////////////////////////////////////////////////////////
// Copied from https://en.wikipedia.org/wiki/LU_decomposition#Code_examples
/* INPUT: A - array of pointers to rows of a square matrix having dimension N
 *        Tol - small tolerance number to detect failure when the matrix is near degenerate
 * OUTPUT: Matrix A is changed, it contains a copy of both matrices L-E and U as A=(L-E)+U such that P*A=L*U.
 *        The permutation matrix is not stored as a matrix, but in an integer vector P of size N+1 
 *        containing column indexes where the permutation matrix has "1". The last element P[N]=S+N, 
 *        where S is the number of row exchanges needed for determinant computation, det(P)=(-1)^S    
 */
int LUPDecompose(float **A, int N, float Tol, int *P) {

    int i, j, k, imax; 
    float maxA, *ptr, absA;

    for (i = 0; i <= N; i++)
        P[i] = i; //Unit permutation matrix, P[N] initialized with N

    for (i = 0; i < N; i++) {
        maxA = 0.0;
        imax = i;

        for (k = i; k < N; k++)
            if ((absA = fabs(A[k][i])) > maxA) { 
                maxA = absA;
                imax = k;
            }

        if (maxA < Tol) return 0; //failure, matrix is degenerate

        if (imax != i) {
            //pivoting P
            j = P[i];
            P[i] = P[imax];
            P[imax] = j;

            //pivoting rows of A
            ptr = A[i];
            A[i] = A[imax];
            A[imax] = ptr;

            //counting pivots starting from N (for determinant)
            P[N]++;
        }

        for (j = i + 1; j < N; j++) {
            A[j][i] /= A[i][i];

            for (k = i + 1; k < N; k++)
                A[j][k] -= A[j][i] * A[i][k];
        }
    }

    return 1;  //decomposition done 
}

/* INPUT: A,P filled in LUPDecompose; b - rhs vector; N - dimension
 * OUTPUT: x - solution vector of A*x=b
 */
void LUPSolve(float **A, int *P, float *b, int N, float *x) {

    for (int i = 0; i < N; i++) {
        x[i] = b[P[i]];

        for (int k = 0; k < i; k++)
            x[i] -= A[i][k] * x[k];
    }

    for (int i = N - 1; i >= 0; i--) {
        for (int k = i + 1; k < N; k++)
            x[i] -= A[i][k] * x[k];

        x[i] /= A[i][i];
    }
}

/* INPUT: A,P filled in LUPDecompose; N - dimension
 * OUTPUT: IA is the inverse of the initial matrix
 */
void LUPInvert(float **A, int *P, int N, float **IA) {
  
    for (int j = 0; j < N; j++) {
        for (int i = 0; i < N; i++) {
            IA[i][j] = P[i] == j ? 1.0 : 0.0;

            for (int k = 0; k < i; k++)
                IA[i][j] -= A[i][k] * IA[k][j];
        }

        for (int i = N - 1; i >= 0; i--) {
            for (int k = i + 1; k < N; k++)
                IA[i][j] -= A[i][k] * IA[k][j];

            IA[i][j] /= A[i][i];
        }
    }
}

/* INPUT: A,P filled in LUPDecompose; N - dimension. 
 * OUTPUT: Function returns the determinant of the initial matrix
 */
float LUPDeterminant(float **A, int *P, int N) {

    float det = A[0][0];

    for (int i = 1; i < N; i++)
        det *= A[i][i];

    return (P[N] - N) % 2 == 0 ? det : -det;
}

void compute_matrix_inverse(float** A, float** IA, int N) {
    int *P = (int*)malloc(N * sizeof(int));
    LUPDecompose(A, N, 0.0001, P);
    LUPInvert(A, P, N, IA);
    // free(P);
}

// //////////////////////////////////////////////////////////////////////////////////////////////////////////

/// @brief Create all free functions ///
void freeQP(QP* qp) {
    free(qp);
}

void free_Settings(Settings* settings) {
    free(settings);
}


void free_Info(Info* info) {
    free(info);
}

void free_Results(Results* results) {
    free(results);
}


void free_tensor(float** tensor, int num_rows) {
    for (int i = 0; i < num_rows; i++) {
        free(tensor[i]);
    }
}

// //////////////////////////////////////////////////////////////////////////////////////////////////////////
// Copied from: https://github.com/rbga/CUDA-Merge-and-Bitonic-Sort/blob/master/BitonicMerge/kernel.cu
void merge(float* arr, float* temp, int left, int mid, int right) 
{
    int i = left;
    int j = mid + 1;
    int k = left;

    while (i <= mid && j <= right) 
    {
        if (arr[i] <= arr[j])
            temp[k++] = arr[i++];
        else
            temp[k++] = arr[j++];
    }

    while (i <= mid)
        temp[k++] = arr[i++];

    while (j <= right)
        temp[k++] = arr[j++];

    for (int idx = left; idx <= right; ++idx)
        arr[idx] = temp[idx];
}

//CPU Implementation of Merge Sort
void mergeSortCPU(float* arr, float* temp, int left, int right) 
{
    if (left >= right)
        return;

    int mid = left + (right - left) / 2;

    mergeSortCPU(arr, temp, left, mid);
    mergeSortCPU(arr, temp, mid + 1, right);

    merge(arr, temp, left, mid, right);
}


// //////////////////////////////////////////////////////////////////////////////////////////////////////////

void print_matrix(int r, int c, float** matrix)
{
    printf("[\n");
    for (int i = 0; i < r; i++)
    {
        printf("[");
        for (int j = 0; j < c; j++) {
            printf("%f, ", matrix[i][j]);
        }
        printf("],");
        printf("\n");
    }
    printf("]\n");
}

void print_vector(int dim, float* vector)
{
    printf("[");
    for (int i = 0; i < dim; i++) {
        printf("%f, ", vector[i]);
    }
    printf("]\n");
}

int argmin(float* array, int n) {
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


float* vector_abs(float* arr, int dim) {
    float* abs_arr = (float*)malloc(dim * sizeof(float));
    for (int i = 0; i < dim; i++) {
        abs_arr[i] = fabs(arr[i]);
    }
    return abs_arr;

}


float* vector_subtract_scalar(float* vector, float scalar, int dim) {
    float* dest = (float*)malloc(dim * sizeof(float));
    for (int i = 0; i < dim; i++) {
        dest[i] = vector[i] - scalar;
    }
    return dest;
}

// Function to concatenate two matrices in C
float** concatenate_matrices(float** mat1, int rows1, int cols1, float** mat2, int rows2, int cols2, int dim, int out_rows, int out_cols) {
    float** result;

    if (dim == 1) { // Horizontal concatenation
        if (rows1 != rows2) {
            return NULL; // Incompatible dimensions
        }
        out_rows = rows1;
        out_cols = cols1 + cols2;
        result = (float**)malloc(out_rows * sizeof(float*));
        for (int i = 0; i < out_rows; i++) {
            result[i] = (float*)malloc(out_cols * sizeof(float));
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
        out_rows = rows1 + rows2;
        out_cols = cols1;
        result = (float**)malloc(out_rows * sizeof(float*));
        for (int i = 0; i < rows1; i++) {
            result[i] = (float*)malloc(out_cols * sizeof(float));
            for (int j = 0; j < out_cols; j++) {
                result[i][j] = mat1[i][j];
            }
        }
        for (int i = 0; i < rows2; i++) {
            result[i + rows1] = (float*)malloc(out_cols * sizeof(float));
            for (int j = 0; j < out_cols; j++) {
                result[i + rows1][j] = mat2[i][j];
            }
        }
    } else {
        return NULL; // Invalid dimension
    }

    return result;
}

void concatenate_matrices_inplace(
float** mat1, int rows1, int cols1, float** mat2, int rows2, int cols2, int dim, float** result, int out_rows, int out_cols
) {
    if (dim == 1) { // Horizontal concatenation
        // if (rows1 != rows2) {
        //     return 0; // Incompatible dimensions
        // }
        out_rows = rows1;
        out_cols = cols1 + cols2;
        for (int i = 0; i < out_rows; i++) {
            for (int j = 0; j < cols1; j++) {
                result[i][j] = mat1[i][j];
            }
            for (int j = 0; j < cols2; j++) {
                result[i][j + cols1] = mat2[i][j];
            }
        }
    } else if (dim == 0) { // Vertical concatenation
        // if (cols1 != cols2) {
        //     return 0; // Incompatible dimensions
        // }
        out_rows = rows1 + rows2;
        out_cols = cols1;
        for (int i = 0; i < rows1; i++) {
            for (int j = 0; j < out_cols; j++) {
                result[i][j] = mat1[i][j];
            }
        }
        for (int i = 0; i < rows2; i++) {
            for (int j = 0; j < out_cols; j++) {
                result[i + rows1][j] = mat2[i][j];
            }
        }
    } //else {
    //     return 0; // Invalid dimension
    // }
}


///////// New Performing stuff Kernel + C functions using it
__global__ void matrixMulKernel(float *d_matrix1, float *d_matrix2, float *d_result, int left, int nelem, int right) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < left && col < right) {
        float sum = 0.0;
        for (int k = 0; k < nelem; k++) {
            sum += d_matrix1[row * nelem + k] * d_matrix2[k * right + col];
        }
        d_result[row * right + col] = sum;
    }
}

void matmul_c(float **matrix1, float **matrix2, float **result, int left, int nelem, int right) {
    size_t size_matrix1 = left * nelem * sizeof(float);
    size_t size_matrix2 = nelem * right * sizeof(float);
    size_t size_result = left * right * sizeof(float);

    float *h_matrix1 = (float*)malloc(size_matrix1);
    float *h_matrix2 = (float*)malloc(size_matrix2);
    float *h_result = (float*)malloc(size_result);

    // Initialize input matrices with valid numbers to avoid NaN results
    for (int i = 0; i < left; i++) {
        for (int j = 0; j < nelem; j++) {
            h_matrix1[i * nelem + j] = matrix1[i][j];
        }
    }
    for (int i = 0; i < nelem; i++) {
        for (int j = 0; j < right; j++) {
            h_matrix2[i * right + j] = matrix2[i][j];
        }
    }

    float *d_matrix1, *d_matrix2, *d_result;
    cudaMalloc(&d_matrix1, size_matrix1);
    cudaMalloc(&d_matrix2, size_matrix2);
    cudaMalloc(&d_result, size_result);

    cudaMemcpy(d_matrix1, h_matrix1, size_matrix1, cudaMemcpyHostToDevice);
    cudaMemcpy(d_matrix2, h_matrix2, size_matrix2, cudaMemcpyHostToDevice);

    dim3 blockSize(32, 32);
    dim3 gridSize((right + blockSize.x - 1) / blockSize.x, (left + blockSize.y - 1) / blockSize.y);

    matrixMulKernel<<<gridSize, blockSize>>>(d_matrix1, d_matrix2, d_result, left, nelem, right);

    cudaMemcpy(h_result, d_result, size_result, cudaMemcpyDeviceToHost);

    for (int i = 0; i < left; i++) {
        for (int j = 0; j < right; j++) {
            result[i][j] = h_result[i * right + j];
        }
    }

    free(h_matrix1);
    free(h_matrix2);
    free(h_result);
    cudaFree(d_matrix1);
    cudaFree(d_matrix2);
    cudaFree(d_result);
}

__global__ void matvecMulKernel(float* d_matrix, float* d_vector, float* d_result, int left, int nelem) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < left) {
        float sum = 0.0;
        for (int j = 0; j < nelem; j++) {
            sum += d_matrix[row * nelem + j] * d_vector[j];
        }
        d_result[row] = sum;
    }
}


void matvecmul_c(float** matrix, float* vector, float* result, int left, int nelem) {
    size_t size_matrix = left * nelem * sizeof(float);
    size_t size_vector = nelem * sizeof(float);
    size_t size_result = left * sizeof(float);

    float* d_matrix;
    float* d_vector;
    float* d_result;

    // Allocate device memory
    cudaMalloc((void**)&d_matrix, size_matrix);
    cudaMalloc((void**)&d_vector, size_vector);
    cudaMalloc((void**)&d_result, size_result);

    // Flatten the 2D matrix into a 1D array for easy CUDA memory handling
    float* flat_matrix = (float*)malloc(size_matrix);
    for (int i = 0; i < left; i++) {
        for (int j = 0; j < nelem; j++) {
            flat_matrix[i * nelem + j] = matrix[i][j];
        }
    }

    // Copy data from host to device
    cudaMemcpy(d_matrix, flat_matrix, size_matrix, cudaMemcpyHostToDevice);
    cudaMemcpy(d_vector, vector, size_vector, cudaMemcpyHostToDevice);

    // Calculate grid and block sizes
    int blockSize = 256; // This can be tuned to best suit hardware capabilities
    int numBlocks = (left + blockSize - 1) / blockSize;

    // Launch the kernel
    matvecMulKernel<<<numBlocks, blockSize>>>(d_matrix, d_vector, d_result, left, nelem);

    // Copy the result back to the host
    cudaMemcpy(result, d_result, size_result, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_matrix);
    cudaFree(d_vector);
    cudaFree(d_result);

    // Free host memory
    free(flat_matrix);
}

__global__ void vectorDotKernel(float *d_vec1, float *d_vec2, float *d_result, int dim) {
    extern __shared__ float cache[]; // Dynamic shared memory allocation
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int cacheIndex = threadIdx.x;

    float tempSum = 0;
    while (tid < dim) {
        tempSum += d_vec1[tid] * d_vec2[tid];
        tid += blockDim.x * gridDim.x;
    }

    // Set the cache values
    cache[cacheIndex] = tempSum;

    // Synchronize threads in this block
    __syncthreads();

    // Reduction within a block
    int i = blockDim.x / 2;
    while (i != 0) {
        if (cacheIndex < i) {
            cache[cacheIndex] += cache[cacheIndex + i];
        }
        __syncthreads();
        i /= 2;
    }

    // Store the result from this block to global memory
    if (cacheIndex == 0) {
        d_result[blockIdx.x] = cache[0];
    }
}


float vector_dot_c(float* vec1, float* vec2, int dim) {
    float *d_vec1, *d_vec2, *d_result;
    float final_result = 0.0;

    int threadsPerBlock = 256;
    int blocksPerGrid = (dim + threadsPerBlock - 1) / threadsPerBlock;

    // Allocate memory for partial sums after blocksPerGrid is calculated
    float *partial_sums = (float *)malloc(blocksPerGrid * sizeof(float));
    
    cudaMalloc((void **)&d_vec1, dim * sizeof(float));
    cudaMalloc((void **)&d_vec2, dim * sizeof(float));
    cudaMalloc((void **)&d_result, blocksPerGrid * sizeof(float));

    cudaMemcpy(d_vec1, vec1, dim * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vec2, vec2, dim * sizeof(float), cudaMemcpyHostToDevice);

    // Launch the kernel with dynamically allocated shared memory
    vectorDotKernel<<<blocksPerGrid, threadsPerBlock, threadsPerBlock * sizeof(float)>>>(d_vec1, d_vec2, d_result, dim);

    // Copy results back to the host
    cudaMemcpy(partial_sums, d_result, blocksPerGrid * sizeof(float), cudaMemcpyDeviceToHost);

    // Accumulate results from all blocks
    for (int i = 0; i < blocksPerGrid; i++) {
        final_result += partial_sums[i];
    }

    // Free device memory
    cudaFree(d_vec1);
    cudaFree(d_vec2);
    cudaFree(d_result);
    
    // Free host memory
    free(partial_sums);

    return final_result;
}


__global__ void concatenateMatricesKernel(float* d_mat1, int rows1, int cols1, float* d_mat2, int rows2, int cols2, float* d_result, int out_rows, int out_cols, int dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    if (dim == 1) { // Horizontal concatenation
        if (idy < out_rows && idx < out_cols) {
            if (idx < cols1) {
                d_result[idy * out_cols + idx] = d_mat1[idy * cols1 + idx];
            } else if (idx >= cols1 && idx < out_cols) {
                d_result[idy * out_cols + idx] = d_mat2[idy * cols2 + (idx - cols1)];
            }
        }
    } else if (dim == 0) { // Vertical concatenation
        if (idx < out_cols && idy < out_rows) {
            if (idy < rows1) {
                d_result[idy * out_cols + idx] = d_mat1[idy * cols1 + idx];
            } else if (idy >= rows1 && idy < out_rows) {
                d_result[idy * out_cols + idx] = d_mat2[(idy - rows1) * cols2 + idx];
            }
        }
    }
}


float** concatenate_matrices_c(float** mat1, int rows1, int cols1, float** mat2, int rows2, int cols2, int dim, int out_rows, int out_cols) {
    size_t size_mat1 = rows1 * cols1 * sizeof(float);
    size_t size_mat2 = rows2 * cols2 * sizeof(float);
    size_t size_result = out_rows * out_cols * sizeof(float);

    float* d_mat1, *d_mat2, *d_result;
    cudaMalloc((void **)&d_mat1, size_mat1);
    cudaMalloc((void **)&d_mat2, size_mat2);
    cudaMalloc((void **)&d_result, size_result);

    // Aplatir et copier les données sur le GPU
    cudaMemcpy(d_mat1, mat1[0], size_mat1, cudaMemcpyHostToDevice);
    cudaMemcpy(d_mat2, mat2[0], size_mat2, cudaMemcpyHostToDevice);

    dim3 blockSize(32, 32);
    dim3 gridSize((out_cols + blockSize.x - 1) / blockSize.x, (out_rows + blockSize.y - 1) / blockSize.y);

    // Lancer le kernel
    concatenateMatricesKernel<<<gridSize, blockSize>>>(d_mat1, rows1, cols1, d_mat2, rows2, cols2, d_result, out_rows, out_cols, dim);

    // Allouer la matrice résultat sur l'hôte et copier les données depuis le GPU
    float** result = (float**)malloc(out_rows * sizeof(float*));
    for (int i = 0; i < out_rows; i++) {
        result[i] = (float*)malloc(out_cols * sizeof(float));
    }
    cudaMemcpy(result[0], d_result, size_result, cudaMemcpyDeviceToHost);

    cudaFree(d_mat1);
    cudaFree(d_mat2);
    cudaFree(d_result);

    return result;
}


__global__ void addMatricesKernel(float* d_A, float* d_B, float* d_result, int num_row, int num_col) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < num_row && col < num_col) {
        int index = row * num_col + col;
        d_result[index] = d_A[index] + d_B[index];
    }
}


void add_matrices_c(float** A, float** B, float** result, int num_row, int num_col) {
    size_t size = num_row * num_col * sizeof(float);

    // Flatten 2D arrays into 1D arrays for easier GPU memory handling
    float* flatA = (float*)malloc(size);
    float* flatB = (float*)malloc(size);
    float* flatResult = (float*)malloc(size);

    for (int i = 0; i < num_row; i++) {
        for (int j = 0; j < num_col; j++) {
            flatA[i * num_col + j] = A[i][j];
            flatB[i * num_col + j] = B[i][j];
        }
    }

    float *d_A, *d_B, *d_result;

    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);
    cudaMalloc((void**)&d_result, size);

    cudaMemcpy(d_A, flatA, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, flatB, size, cudaMemcpyHostToDevice);

    dim3 blockSize(32, 32);
    dim3 gridSize((num_col + blockSize.x - 1) / blockSize.x, (num_row + blockSize.y - 1) / blockSize.y);

    addMatricesKernel<<<gridSize, blockSize>>>(d_A, d_B, d_result, num_row, num_col);

    // Copy the result back to the host
    cudaMemcpy(flatResult, d_result, size, cudaMemcpyDeviceToHost);

    // Unflatten result back to 2D array
    for (int i = 0; i < num_row; i++) {
        for (int j = 0; j < num_col; j++) {
            result[i][j] = flatResult[i * num_col + j];
        }
    }

    // Free resources
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_result);
    free(flatA);
    free(flatB);
    free(flatResult);
}

__global__ void subtractMatricesKernel(float* d_A, float* d_B, float* d_result, int num_row, int num_col) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < num_row && col < num_col) {
        int index = row * num_col + col;
        d_result[index] = d_A[index] - d_B[index];
    }
}


void subtract_matrices_c(float** A, float** B, float** result, int num_row, int num_col) {
    size_t size = num_row * num_col * sizeof(float);

    // Flatten 2D arrays into 1D arrays for easier GPU memory handling
    float* flatA = (float*)malloc(size);
    float* flatB = (float*)malloc(size);
    float* flatResult = (float*)malloc(size);

    for (int i = 0; i < num_row; i++) {
        for (int j = 0; j < num_col; j++) {
            flatA[i * num_col + j] = A[i][j];
            flatB[i * num_col + j] = B[i][j];
        }
    }

    float *d_A, *d_B, *d_result;

    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);
    cudaMalloc((void**)&d_result, size);

    cudaMemcpy(d_A, flatA, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, flatB, size, cudaMemcpyHostToDevice);

    dim3 blockSize(32, 32);
    dim3 gridSize((num_col + blockSize.x - 1) / blockSize.x, (num_row + blockSize.y - 1) / blockSize.y);

    subtractMatricesKernel<<<gridSize, blockSize>>>(d_A, d_B, d_result, num_row, num_col);

    // Copy the result back to the host
    cudaMemcpy(flatResult, d_result, size, cudaMemcpyDeviceToHost);

    // Unflatten result back to 2D array
    for (int i = 0; i < num_row; i++) {
        for (int j = 0; j < num_col; j++) {
            result[i][j] = flatResult[i * num_col + j];
        }
    }

    // Free resources
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_result);
    free(flatA);
    free(flatB);
    free(flatResult);
}


typedef struct
{
    QP* qp;
    Settings* settings;
    float* rhos;
    int rhos_len;
    float*** W_ks;
    float*** B_ks;
    float** b_ks;
    int clamp_left;
    int clamp_right;
} ReLU_Layer;

void free_ReLU_Layer(ReLU_Layer* layer) {
    free(layer);
}

ReLU_Layer* Initialize_ReLU_Layer (
    QP* qp,
    Settings* settings
) {
    ReLU_Layer* relu_layer = (ReLU_Layer*)malloc(sizeof(ReLU_Layer));
    relu_layer->qp = qp;
    relu_layer->settings = settings;

    // setup rhos
    // compute the number of rhos needed
    int increasing_amount = (int)((1. + (log(settings->rho) - log(settings->rho_min)) / log(settings->adaptive_rho_tolerance)));
    int decreasing_amount = (int)((log(settings->rho_max) - log(settings->rho)) / log(settings->adaptive_rho_tolerance));
    int total_amount = increasing_amount + decreasing_amount;
    // printf("The total amount is: %d\n", total_amount);
    float* rhos = (float*)malloc(total_amount * sizeof(float));
    rhos[0] = settings->rho;
    relu_layer->rhos = rhos;
    relu_layer->rhos_len = 1;
    if (settings->adaptive_rho) {
        float rho = settings->rho / settings->adaptive_rho_tolerance;
        // float* rhos;
        int i = 1;
        // printf("Increasing rho:\n");
        while (rho >= settings->rho_min) {
            // rhos = (float*)realloc(rhos, (i+1) * sizeof(float));
            rhos[i] = rho;
            // printf("i: %d, rho: %f\n", i, rho);
            rho = rho / settings->adaptive_rho_tolerance;
            i++;
        }
        rho = settings->rho*settings->adaptive_rho_tolerance;
        // printf("Decreasing rho: \n");
        while (rho <= settings->rho_max) {
            // rhos = (float*)realloc(rhos, (i+1) * sizeof(float));
            rhos[i] = rho;
            // printf("i: %d, rho: %f\n", i, rho);
            rho = rho * settings->adaptive_rho_tolerance;
            i++;
        }
        // printf("Finished rho\n");
        relu_layer->rhos = rhos;
        relu_layer->rhos_len = i;
        // Sort
        float* temp = (float*)malloc(relu_layer->rhos_len * sizeof(float));
        mergeSortCPU(relu_layer->rhos, temp, 0, relu_layer->rhos_len - 1);
    }
    
    // setup matrices
    float** H = qp->H; 
    float** A = qp->A;
    float* g = qp->g;
    float* l = qp->l;
    float* u = qp->u;
    // int nx = qp->nx;
    int nc = qp->nc;
    // int nf = qp->nf; 
    int nf = qp->nx;
    float sigma = settings->sigma;

    float*** kkts_rhs_invs = (float***)malloc(relu_layer->rhos_len * sizeof(float**));
    for (int i = 0; i < relu_layer->rhos_len; i++) {
        // kkts_rhs_invs[i] = (float**)malloc(nf * sizeof(float*));
        // for (int j = 0; j < nf; j++) {
        //     kkts_rhs_invs[i][j] = (float*)malloc(nf * sizeof(float));
        //     for ()
        // }
        kkts_rhs_invs[i] = create_matrix(nf, nf);
    }
    // float kkts_rhs_invs[relu_layer->rhos_len][nf][nf];
    float* flag_checks = (float*)calloc(nc, sizeof(float));
    vector_subtract(u, l, flag_checks, nc);
    bool* conditional = (bool*)calloc(nc, sizeof(bool));
    for (int i = 0; i < nc; i++) {
        conditional[i] = flag_checks[i] <= settings->eq_tol;
    }
    free(flag_checks);
    
    for (int i = 0; i < relu_layer->rhos_len; i++) {
        float rho_scalar = relu_layer->rhos[i];
        float* rho = (float*)calloc(nc, sizeof(float));
        for (int j = 0; j < nc; j++) {
            rho[j] = rho_scalar;
        }
        vector_where(conditional, rho, rho_scalar * 1e3, rho_scalar, nc);
        float** rho_mat = create_diagonal_matrix(rho, nc);
        float* sigma_vector = (float*)calloc(nf, sizeof(float));
        for (int j = 0; j < nf; j++) {
            sigma_vector[j] = sigma;
        }
        float** sigma_mat = create_diagonal_matrix(sigma_vector, nf);

        float** A_transpose = transpose_matrix(A, nc, nf);
        float** rho_A = create_matrix(nc, nf);
        matmul(rho_mat, A, rho_A, nc, nc, nf);
        float** AT_rho_A = create_matrix(nf, nf);
        matmul(A_transpose, rho_A, AT_rho_A, nf, nc, nf);

        float** summed_mat = create_matrix(nf, nf);
        add_matrices(H, sigma_mat, summed_mat, nf, nf);
        add_matrices(summed_mat, AT_rho_A, summed_mat, nf, nf);
        // need to take inverse now.... of summed mats
        float** summed_mat_inv = create_matrix(nf, nf);
        compute_matrix_inverse(summed_mat, summed_mat_inv, nf);
        for (int a = 0; a < nf; a++) {
            for (int b = 0; b < nf; b++) {
                kkts_rhs_invs[i][a][b] = summed_mat_inv[a][b];
                // printf("a: %d, b: %d, val: %f\n", a, b, summed_mat_inv[a][b]);
            }
        }
        // kkts_rhs_invs[i] = summed_mat_inv;
        // // free variables
        free(sigma_vector);
        free(rho);
        free_tensor(rho_A, nc);
        free_tensor(A_transpose, nf);
        free_tensor(summed_mat, nf);
        free_tensor(sigma_mat, nf);
        // printf("Freed them MFs\n");
    }

    // Define W_ks, B_ks, b_ks
    int W_Row = nf + 2 * nc;
    float*** W_ks = (float***)malloc(relu_layer->rhos_len * sizeof(float**));
    float*** B_ks = (float***)malloc(relu_layer->rhos_len * sizeof(float**));
    float** b_ks = (float**)malloc(relu_layer->rhos_len * sizeof(float*));
    for (int i = 0; i < relu_layer->rhos_len; i++) {
        // W_ks[i] = (float**)malloc(W_Row * sizeof(float*));
        // B_ks[i] = (float**)malloc(W_Row * sizeof(float*));
        // b_ks[i] = (float*)malloc(W_Row * sizeof(float));
        // for (int j = 0; j < nf + 2 * nc; j++) {
        //     W_ks[i][j] = (float*)malloc(W_Row * sizeof(float));
        // }
        // for (int j = 0; j < nf; j++) {
        //     B_ks[i][j] = (float*)malloc(nf * sizeof(float));
        // }
        W_ks[i] = create_matrix(W_Row, W_Row);
        B_ks[i] = create_matrix(W_Row, nf);
        b_ks[i] = create_vector(W_Row);
    }
    // printf("I'm here\n");
    // Define variables
    float* rho = create_vector(nc);
    float** rho_A = create_matrix(nc, nf);
    float** AT_rho_A = create_matrix(nf, nf);
    float** summed_mat = create_matrix(nf, nf);
    float** AT_rho = create_matrix(nf, nc);
    float** K_AT_rho = create_matrix(nf, nc);
    float** neg_K = create_matrix(nf, nf);
    float** A_elem00 = create_matrix(nc, nf);
    float** A_K_AT_rho = create_matrix(nc, nc);
    float** K_AT = create_matrix(nf, nc);
    float** A_K_AT = create_matrix(nc, nc);
    float** neg_AK = create_matrix(nc, nf);
    float** neg_A = create_matrix(nc, nf);
    float** zeros = create_matrix(nc, nf);
    float** A_transpose = transpose_matrix(A, nc, nf);
    float* ones_vector = (float*)calloc(nc, sizeof(float));
    for (int j = 0; j < nc; j++) {
        ones_vector[j] = 1;
    }
    float** Ic = create_diagonal_matrix(ones_vector, nc);
    float** rho_mat = create_matrix(nc, nc);
    float* sigma_vector = create_vector(nf);
    for (int j = 0; j < nf; j++) {
        sigma_vector[j] = sigma;
    }
    float** Ix = create_diagonal_matrix(sigma_vector, nf);
    float** neg_I = create_scalar_diagonal_matrix(-1, nf);
    float** rho_inv = create_matrix(nc, nc);
    // Define elements involved with concatenation
    float** elem_00 = create_matrix(nf, nf);
    float** elem_01 = create_matrix(nf, nc);
    float** elem_02 = create_matrix(nf, nc);
    float** elem_10 = create_matrix(nc, nf);
    float** elem_11 = create_matrix(nc, nc);
    float** elem_12 = create_matrix(nc, nc);
    float** elem_20 = create_matrix(nc, nf);
    float** elem_21 = create_matrix(nc, nc);
    float** elem_22 = create_matrix(nc, nc);



    // float** elem_00_01 = create_matrix(nf, nf+nc);

    float* b_k = create_vector(nf + 2*nc);

    for (int rho_ind = 0; rho_ind < relu_layer->rhos_len; rho_ind++) {
        // printf("##################################################\n");
        // printf("rho_ind is: %d\n", rho_ind);
        float rho_scalar = relu_layer->rhos[rho_ind];
        for (int j = 0; j < nc; j++) {
            rho[j] = rho_scalar;
        }
        // printf("created rho vector\n");
        vector_where(conditional, rho, rho_scalar * 1e3, rho_scalar, nc);
        // float** rho_mat = create_diagonal_matrix(rho, nc);
        populate_diagonal_matrix(rho, rho_mat, nc);
        // free(sigma_vector);
        // float** K = kkts_rhs_invs[rho_ind];
        // printf("created Ic, K\n");
        // CREATING W_ks elements!!!!
        // Create (0, 0) element
        // K @ (sigma * Ix - A.T @ (rho @ A))
        matmul(rho_mat, A, rho_A, nc, nc, nf);
        matmul(A_transpose, rho_A, AT_rho_A, nf, nc, nf);
        subtract_matrices(Ix, AT_rho_A, summed_mat, nf, nf);
        matmul(kkts_rhs_invs[rho_ind], summed_mat, elem_00, nf, nf, nf);
        // printf("Created elem 00\n");
        // Create (0, 1) element
        // 2 * K @ A.T @ rho
        matmul(A_transpose, rho_mat, AT_rho, nf, nc, nc); // [nf,nc]
        matmul(kkts_rhs_invs[rho_ind], AT_rho, K_AT_rho, nf, nf, nc); // [nf, nc]
        scalar_multiply_matrix(K_AT_rho, 2, K_AT_rho, nf, nc);
        // float** elem_01 = copy_matrix(K_AT_rho, nf, nc);
        copy_matrix_inplace(K_AT_rho, elem_01, nf, nc);
        // printf("Created elem 01\n");
        // Create (0, 2) element
        // -K @ A.T
        matmul(neg_I, kkts_rhs_invs[rho_ind], neg_K, nf, nf, nf);
        matmul(neg_K, A_transpose, elem_02, nf, nf, nc);
        // printf("Created elem 02\n");
        // Create (1, 0) element
        // A @ K @ (sigma * Ix - A.T @ (rho @ A)) + A
        // A @ elem_00 + A
        matmul(A, elem_00, A_elem00, nc, nf, nf);
        add_matrices(A_elem00, A, elem_10, nc, nf);
        // printf("Created elem 10\n");
        // Create (1, 1) element
        // 2 * A @ K @ A.T @ rho - Ic
        // 2 * partial_elem01 - Ic
        scalar_multiply_matrix(K_AT_rho, .5, K_AT_rho, nf, nc);
        matmul(A, K_AT_rho, A_K_AT_rho, nc, nf, nc);
        scalar_multiply_matrix(A_K_AT_rho, 2, A_K_AT_rho, nc, nc);
        subtract_matrices(A_K_AT_rho, Ic, elem_11, nc, nc);
        // printf("Created elem 11\n");
        // Create (1, 2) element
        // -A @ K @ A.T + rho_inv
        matmul(kkts_rhs_invs[rho_ind], A_transpose, K_AT, nf, nf, nc);
        matmul(A, K_AT, A_K_AT, nc, nf, nc);
        scalar_multiply_matrix(A_K_AT, -1, A_K_AT, nc, nc);
        // float** rho_inv = create_scalar_diagonal_matrix(1.0, nc);
        // divide_diag(rho_inv, rho_mat, rho_inv, nc);
        divide_diag(Ic, rho_mat, rho_inv, nc);
        add_matrices(A_K_AT, rho_inv, elem_12, nc, nc);
        // printf("Created elem 12\n");
        // Create (2, 0) element
        // rho @ A
        matmul(rho_mat, A, elem_20, nc, nc, nf);
        // printf("Created elem 20\n");
        // Create (2, 1) element                                                    [nc, nf]
        // -rho
        scalar_multiply_matrix(rho_mat, -1., elem_21, nc, nc);
        // printf("Created elem 21\n");
        // Create (2, 2) element                                                    [nc, nc]
        // Ic
        // float** elem_22 = copy_matrix(Ic, nc, nc); //                              [nc, nc]
        copy_matrix_inplace(Ic, elem_22, nc, nc);
        // printf("Created elem 22\n");
        // W_ks is of size:
        // [nf, nf + nc + nc]
        // [nc, nf + nc + nc]
        // [nc, nf + nc + nc]
        // [nf + 2*nc, nf + 2*nc]
        // [elem_00, elem_01, elem_02]
        // [elem_10, elem_11, elem_12]
        // [elem_20, elem_21, elem_22]
        float** elem_00_01 = concatenate_matrices(elem_00, nf, nf, elem_01, nf, nc, 1, nf, nf+nc);
        // concatenate_matrices_inplace(elem_00_01, nf, nf+nc, elem_02, nf, nc, 1, elem_00_01, nf, nf+2*nc);
        float** row_0 = concatenate_matrices(elem_00_01, nf, nf+nc, elem_02, nf, nc, 1, nf, nf+2*nc);
        float** elem_10_11 = concatenate_matrices(elem_10, nc, nf, elem_11, nc, nc, 1, nc, nf+nc);
        float** row_1 = concatenate_matrices(elem_10_11, nc, nf+nc, elem_12, nc, nc, 1, nc, nf+2*nc);
        float** elem_20_21 = concatenate_matrices(elem_20, nc, nf, elem_21, nc, nc, 1, nc, nf+nc);
        float** row_2 = concatenate_matrices(elem_20_21, nc, nf+nc, elem_22, nc, nc, 1, nc, nf+2*nc);
        float** rows_01 = concatenate_matrices(row_0, nf, nf+2*nc, row_1, nc, nf+2*nc, 0, nf+nc, nf+2*nc);
        float** rows_012 = concatenate_matrices(rows_01, nf+nc, nf+2*nc, row_2, nc, nf+2*nc, 0, nf+2*nc, nf+2*nc);
        W_ks[rho_ind] = copy_matrix(rows_012, nf+2*nc, nf+2*nc);
        // FREE TENSORS
        // free_tensor(rho_A, nc);
        // free_tensor(A_transpose, nf);
        // Creating the B_ks
        scalar_multiply_matrix(A, -1, neg_A, nc, nf);
        matmul(neg_A, kkts_rhs_invs[rho_ind], neg_AK, nc, nf, nf);
        float** elems_01 = concatenate_matrices(neg_K, nf, nf, neg_AK, nc, nf, 0, nf+nc, nf);
        float** elems_012 = concatenate_matrices(elems_01, nf+nc, nf, zeros, nc, nf, 0, nf+2*nc, nf);
        B_ks[rho_ind] = copy_matrix(elems_012, nf+2*nc, nf);
        // Creating the b_ks
        matvecmul(B_ks[rho_ind], g, b_k, nf+2*nc, nf);
        // matmul(B_ks[rho_ind], g, b_k, nf+2*nc, nf, 1);
        b_ks[rho_ind] = copy_vector(b_k, nf+2*nc);
        // FREE Tensors
        free_tensor(elem_00_01, nf);
        free_tensor(row_0, nf);
        free_tensor(elem_10_11, nc);
        free_tensor(row_1, nc);
        free_tensor(elem_20_21, nc);
        free_tensor(row_2, nc);
        free_tensor(rows_01, nf+nc);
        free_tensor(rows_012, nf+2*nc);
        free_tensor(elems_01, nf+nc);
        free_tensor(elems_012, nf+2*nc);
    }
    relu_layer->W_ks = W_ks;
    relu_layer->B_ks = B_ks;
    relu_layer->b_ks = b_ks;
    relu_layer->clamp_left = qp->nx;
    relu_layer->clamp_right = qp->nx + qp->nc;
    
    // Free remaining variables
    free(rho);
    free_tensor(rho_A, nc);
    free_tensor(AT_rho_A, nf);
    free_tensor(summed_mat, nf);
    free_tensor(AT_rho, nf);
    free_tensor(K_AT_rho, nf);
    free_tensor(neg_K, nf);
    free_tensor(A_elem00, nc);
    free_tensor(A_K_AT_rho, nc);
    free_tensor(K_AT, nf);
    free_tensor(A_K_AT, nc);
    free_tensor(neg_AK, nc);
    free_tensor(neg_A, nc);
    free_tensor(zeros, nc);
    free_tensor(A_transpose, nf);
    free_tensor(Ic, nc);
    free_tensor(rho_mat, nc);
    free_tensor(Ix, nf);
    free_tensor(neg_I, nf);
    free_tensor(rho_inv, nc);
    free_tensor(elem_00, nf);
    free_tensor(elem_01, nf);
    free_tensor(elem_02, nf);
    free_tensor(elem_10, nc);
    free_tensor(elem_11, nc);
    free_tensor(elem_12, nc);
    free_tensor(elem_20, nc);
    free_tensor(elem_21, nc);
    free_tensor(elem_22, nc);
    free(b_k);
    
    // printf("All set\n");
    return relu_layer;
}

float* ReLU_Layer_Forward(ReLU_Layer* layer, float* x, int idx) {
    int idx1 = layer->clamp_left;
    int idx2 = layer->clamp_right;
    int nx = layer->qp->nx;
    int nc = layer->qp->nc;

    int D = nx + 2*nc;
    float* temp = create_vector(D);
    float* intermediate = create_vector(D);
    matvecmul(layer->W_ks[idx], x, temp, D, D);
    vector_add(temp, layer->b_ks[idx], intermediate, D);

    for (int i = idx1; i < idx2; i++) {
        float val = intermediate[i];
        if (val < layer->qp->l[i - idx1]) {
            intermediate[i] = layer->qp->l[i - idx1];
        } else if (val > layer->qp->u[i - idx1]) {
            intermediate[i] = layer->qp->u[i - idx1];
        }
    }
    free(temp);
    return intermediate;
}

typedef struct  {
    time_t      tv_sec;     /* seconds */
    suseconds_t tv_usec;    /* microseconds */
} timeval;

typedef struct
{
    Info* info;
    Results* results;
    Settings* settings;
    ReLU_Layer* layers;
    QP* qp;
    struct timeval start;
    struct timeval end;
    // float* x;
    // float* z;
    // float* lam;
    float* output;
    int rho_ind;
} ReLU_QP;


ReLU_QP* Initialize_ReLU_QP(
    float** H, float* g, float** A, float* l, float* u,
    bool warm_starting, 
    bool scaling,
    float rho,
    float rho_min,
    float rho_max,
    float sigma, 
    bool adaptive_rho,
    int adaptive_rho_interval,
    int adaptive_rho_tolerance,
    int max_iter,
    float eps_abs,
    int check_interval,
    bool verbose,
    float eq_tol,
    int nc,
    // int nf,
    int nx
) {

    ReLU_QP* relu_qp = (ReLU_QP*)malloc(sizeof(ReLU_QP));

    // relu_qp->start = clock();
    gettimeofday(&relu_qp->start, NULL);
    relu_qp->settings = InitializeSettings(
        verbose,
        warm_starting,
        scaling,
        rho,
        rho_min,
        rho_max,
        sigma,
        adaptive_rho,
        adaptive_rho_interval,
        adaptive_rho_tolerance,
        max_iter,
        eps_abs,
        eq_tol,
        check_interval
    );
    relu_qp->info = InitializeInfo(0,0,0,0,0,0,0,0,0);
    relu_qp->results = InitializeResults(
        create_vector(nx), 
        create_vector(nc), 
        relu_qp->info
    );
    relu_qp->qp = InitializeQP(H, g, A, l, u, nx, nc);
    relu_qp->layers = Initialize_ReLU_Layer(relu_qp->qp, relu_qp->settings);
    // float* x = create_vector(nx);
    // float* z = create_vector(nc);
    // float* lam = create_vector(nc);
   relu_qp->output = create_vector(nx + 2*nc);
   for (int i =0; i < nx + 2*nc; i++) {
        relu_qp->output[i] = 0;
   }
    // self.rho_ind = np.argmin(np.abs(self.layers.rhos.cpu().detach().numpy() - self.settings.rho))
    float* rhos_minus_rho = vector_subtract_scalar(relu_qp->layers->rhos, relu_qp->settings->rho, relu_qp->layers->rhos_len);
    relu_qp->rho_ind = argmin(vector_abs(rhos_minus_rho, relu_qp->layers->rhos_len), relu_qp->layers->rhos_len);

    gettimeofday(&relu_qp->end, NULL);
    float elapsedTime = (float)(relu_qp->end.tv_sec - relu_qp->start.tv_sec);
    elapsedTime += (((float)(relu_qp->end.tv_usec - relu_qp->start.tv_usec)) / 1000000.0);
    relu_qp->info->setup_time = elapsedTime;

    free(rhos_minus_rho);
    return relu_qp;
}

float compute_J(float** H, float* g, float* x, int nx) {
    float* Hx = create_vector(nx);
    matvecmul(H, x, Hx, nx, nx);
    float Hx_dot_x = vector_dot(Hx, x, nx);
    float gx = vector_dot(g, x, nx);
    free(Hx);
    return 0.5 * Hx_dot_x + gx;
}


void clear_primal_dual(ReLU_QP* relu_qp) {
    for (int i = 0; i < relu_qp->qp->nx + 2 * relu_qp->qp->nc; i++) {
        relu_qp->output[i] = 0;
    }
    // self.rho_ind = np.argmin(np.abs(self.layers.rhos.cpu().detach().numpy() - self.settings.rho))
    float* rhos_minus_rho = vector_subtract_scalar(relu_qp->layers->rhos, relu_qp->settings->rho, relu_qp->layers->rhos_len);
    relu_qp->rho_ind = argmin(vector_abs(rhos_minus_rho, relu_qp->layers->rhos_len), relu_qp->layers->rhos_len);
    free(rhos_minus_rho);
}


void update_results(ReLU_QP* relu_qp, int iter, float pri_res, float dua_res, float rho_estimate) {
    // gettimeofday(&relu_qp->start, NULL);
    relu_qp->results->info->iter = iter;
    relu_qp->results->info->pri_res = pri_res;
    relu_qp->results->info->dua_res = dua_res;
    relu_qp->results->info->rho_estimate = rho_estimate;

    int nx = relu_qp->qp->nx;
    int nc = relu_qp->qp->nc;
    float* x = (float*)malloc(relu_qp->qp->nx * sizeof(float));
    for (int i = 0; i < relu_qp->qp->nx; i++) {
        x[i] = relu_qp->output[i];
    }
    float* z = (float*)malloc(relu_qp->qp->nc * sizeof(float));
    for (int i = nx; i < nx + nc; i++) {
        z[i - nx] = relu_qp->output[i];
    }
    relu_qp->results->x = x;
    relu_qp->results->z = z;
    relu_qp->results->info->obj_val = compute_J(relu_qp->qp->H, relu_qp->qp->g, x, nx);
    gettimeofday(&relu_qp->end, NULL);
    float elapsedTime = (float)(relu_qp->end.tv_sec - relu_qp->start.tv_sec);
    elapsedTime += (((float)(relu_qp->end.tv_usec - relu_qp->start.tv_usec)) / 1000000.0);
    relu_qp->results->info->run_time = elapsedTime;
    relu_qp->results->info->solve_time = relu_qp->results->info->update_time + elapsedTime;
    
    // float* lam = create_vector(relu_qp->qp->nc);
    // TODO: Need to add the warm_starting check and then the clear_primal_dual function.
    if (relu_qp->settings->warm_starting) {
        clear_primal_dual(relu_qp);
    }
    
    // free(x);
    // free(z);
    // free(lam);
}


float vector_inf_norm(float* vec, int length) {
    float max_val = 0.0;
    for (int i = 0; i < length; i++) {
        float abs_val = fabs(vec[i]);
        if (abs_val > max_val) {
            max_val = abs_val;
        }
    }
    return max_val;
}


void compute_residuals(
    float** H, float** A, float* g, float* x, float* z, float* lam, 
    float* rho, float rho_min, float rho_max, int nx, int nc, 
    float* primal_res, float* dual_res
) {
    float *t1 = (float*)malloc(nc * sizeof(float));
    float *t2 = (float*)malloc(nx * sizeof(float));
    float *t3 = (float*)malloc(nx * sizeof(float));
    float *temp_primal = (float*)malloc(nc * sizeof(float));
    float *temp_dual = (float*)malloc(nx * sizeof(float));
    matvecmul(A, x, t1, nc, nx);
    matvecmul(H, x, t2, nx, nx);
    // float** AT = transpose_matrix(A, nc, nx);
    // matvecmul(AT, lam, t3, nx, nc);
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
    float numerator = *primal_res / fmax(vector_inf_norm(t1, nc), vector_inf_norm(z, nc));
    float denom = *dual_res / fmax(fmax(vector_inf_norm(t2, nx), vector_inf_norm(t3, nx)), vector_inf_norm(g, nx));

    *rho = fmax(fmin(sqrt(numerator / denom) * (*rho), rho_max), rho_min);
    // float denom = *dual_res / fmax(fmax(vector_inf_norm(t2, nx), vector_inf_norm(t3, nx)), vector_inf_norm(g, nx));

    // *rho = fmax(fmin(sqrt(numerator / denom) * (*rho), rho_max), rho_min);

    free(t1);
    free(t2);
    free(t3);
    free(temp_primal);
    free(temp_dual);
}


Results* solve(ReLU_QP* relu_qp) {
    // LARGE_INTEGER frequency, t1, t2;
    // QueryPerformanceFrequency(&frequency); // Get the frequency for timing calculations
    // QueryPerformanceCounter(&t1); // Start the performance counter
    gettimeofday(&relu_qp->start, NULL);

    Settings* settings = relu_qp->settings;
    int nx = relu_qp->qp->nx;
    int nc = relu_qp->qp->nc;
    float rho = relu_qp->layers->rhos[relu_qp->rho_ind]; // Starting rho from adaptive rho array

    float* x = (float*)malloc(relu_qp->qp->nx * sizeof(float));
    for (int i = 0; i < relu_qp->qp->nx; i++) {
        x[i] = relu_qp->output[i];
    }
    float* z = (float*)malloc(relu_qp->qp->nc * sizeof(float));
    for (int i = nx; i < nx + nc; i++) {
        z[i - nx] = relu_qp->output[i];
    }
    float* lam = create_vector(relu_qp->qp->nc);
    for (int i = nx+nc; i < nx + nc * 2; i++) {
        lam[i - (nx + nc)] = relu_qp->output[i];
    }
    float primal_res, dual_res;

    for (int k = 1; k <= settings->max_iter; k++) {
        // Assume a function to update output based on current state
        relu_qp->output = ReLU_Layer_Forward(relu_qp->layers, relu_qp->output, relu_qp->rho_ind);
        // Perform computations as required
        if (k % settings->check_interval ==  0 && settings->adaptive_rho) {
            for (int i = 0; i < relu_qp->qp->nx; i++) {
                x[i] = relu_qp->output[i];
            }
            float* z = (float*)malloc(relu_qp->qp->nc * sizeof(float));
            for (int i = nx; i < nx + nc; i++) {
                z[i - nx] = relu_qp->output[i];
            }
            float* lam = create_vector(relu_qp->qp->nc);
            for (int i = nx+nc; i < nx + nc * 2; i++) {
                lam[i - (nx + nc)] = relu_qp->output[i];
            }

            compute_residuals(
                relu_qp->qp->H, relu_qp->qp->A, relu_qp->qp->g, x, z, lam, &rho, 
                settings->rho_min, settings->rho_max, nx, nc, &primal_res, &dual_res
            );

            // Adaptive rho adjustment
            if ((rho > relu_qp->layers->rhos[relu_qp->rho_ind] * settings->adaptive_rho_tolerance) && (relu_qp->rho_ind < relu_qp->layers->rhos_len - 1)) {
                relu_qp->rho_ind++;
                relu_qp->rho_ind = fmin(relu_qp->rho_ind, relu_qp->layers->rhos_len - 1);
                
            } else if ((rho < (relu_qp->layers->rhos[relu_qp->rho_ind] / settings->adaptive_rho_tolerance)) && (relu_qp->rho_ind > 0)) {
                relu_qp->rho_ind--;
            }

            // Verbose output
            if (settings->verbose) {
                printf("Iter: %d, rho: %.2e, res_p: %.2e, res_d: %.2e\n", k, rho, primal_res, dual_res);
            }

            // Check for convergence
            // printf("Priting x: ")
            if (primal_res < settings->eps_abs * sqrt(nc) && dual_res < settings->eps_abs * sqrt(nx)) {
                update_results(relu_qp, k, primal_res, dual_res, rho);
                return relu_qp->results;
            }
        }
    }

    compute_residuals(
        relu_qp->qp->H, relu_qp->qp->A, relu_qp->qp->g, x, z, lam, &rho, 
        settings->rho_min, settings->rho_max, nx, nc, &primal_res, &dual_res
    );
    update_results(relu_qp, settings->max_iter, primal_res, dual_res, rho);
    return relu_qp->results;

    // QueryPerformanceCounter(&t2); // Stop the performance counter
    // float elapsedTime = (float)(t2.QuadPart - t1.QuadPart) * 1000.0 / frequency.QuadPart;
    // problem->info->solve_time = elapsedTime;
}

// Benchmarking Functions
float return_bounded_random_number(float upper_bound) 
// Taken from Hw5.
{
    float random_number = ((float) rand() / (float)RAND_MAX) * upper_bound;
    // float rounded_number = roundf(random_number * 1000) / 1000; 
    return random_number;
}


float** generate_random_matrix(int rows, int cols, float upper_bound) {
    float** matrix = create_matrix(rows, cols);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            matrix[i][j] = return_bounded_random_number(upper_bound);
        }
    }
    return matrix;

}

float* generate_random_vector(int num_elems, float upper_bound) {
    float* vector = create_vector(num_elems);
    for (int i = 0; i < num_elems; i++) {
        vector[i] = return_bounded_random_number(upper_bound);
    }
    return vector;

}


void generate_random_H(int nx, float** H_actual) {
    // Generate a random H
    float** H = generate_random_matrix(nx, nx, 1);
    float** HT = transpose_matrix(H, nx, nx);
    float** HTH = create_matrix(nx, nx);
    matmul(HT, H, HTH, nx, nx, nx);
    float** Ix = create_scalar_diagonal_matrix(1., nx);
    // float** HTH_plus_Ix = create_matrix(nx, nx);
    add_matrices(HTH, Ix, H_actual, nx, nx);
    // Free variables
    free(H);
    free(HT);
    free(HTH);
    free(Ix);
    // return HTH_plus_Ix;
}

void generate_random_QP(
    int nx, int n_eq, int n_ineq, 
    float** H, float* g, float** A_actual, float* l, float*u
) {
    generate_random_H(nx, H);
    float** A = generate_random_matrix(n_eq, nx, 1.);
    float** C = generate_random_matrix(n_ineq, nx, 1.);
    float* active_ineq_temp = generate_random_vector(n_ineq, 1.);
    float* active_ineq = create_vector(n_ineq);
    for (int i = 0; i < n_ineq; i++) {
        if (active_ineq_temp[i] > 0.5) {
            active_ineq[i] = 1.;
        }
        else {
            active_ineq[i] = 0.;
        }
    }
    float* mu = generate_random_vector(n_eq, 1.);
    float* lam = generate_random_vector(n_ineq, 1.);
    for (int i = 0; i < n_ineq; i++) {
        lam[i] = lam[i] * active_ineq[i];
    }
    float* x = generate_random_vector(nx, 1.);
    float* b = create_vector(n_eq);
    matvecmul(A, x, b, n_eq, nx);
    float* Cx = create_vector(nx);
    matvecmul(C, x, Cx, n_ineq, nx);
    float* temp = generate_random_vector(n_ineq, 1.);
    for (int i=0; i < n_ineq; i++) {
        temp[i] = temp[i] * (1. - active_ineq[i]);
    }
    float* d = create_vector(n_ineq);
    vector_subtract(Cx, temp, d, n_ineq);
    // Let's get g
    float** AT = transpose_matrix(A, n_eq, nx);
    float* ATmu = create_vector(nx);
    matvecmul(AT, mu, ATmu, nx, n_eq);
    
    float* Hx = create_vector(nx);
    matvecmul(H, x, Hx, nx, nx);

    float** CT = transpose_matrix(C, n_ineq, nx);
    float* CTlam = create_vector(nx);
    matvecmul(CT, lam, CTlam, nx, n_ineq);

    for (int i = 0; i < nx; i++) {
        g[i] = -Hx[i] - ATmu[i] - CTlam[i];
    }

    float** A_temp = concatenate_matrices(A, n_eq, nx, C, n_ineq, nx, 0, n_eq + n_ineq, nx);
    float** zeros = create_matrix( n_eq + n_ineq, nx);
    add_matrices(A_temp, zeros, A_actual, n_eq + n_ineq, nx);
    for (int i = 0; i < n_eq; i++) {
        l[i] = b[i];
    }
    for (int i = n_eq; i < n_eq + n_ineq; i++) {
        l[i] = d[i - n_eq];
    }
    for (int i = 0; i < n_eq; i++) {
        u[i] = b[i];
    }
    for (int i = n_eq; i < n_eq + n_ineq; i++) {
        u[i] = INFINITY;
    }

    // printf("Printing l: \n");
    // for (int i = 0; i < n_eq + n_ineq; i++) {
    //     printf("%f\t", l[i]);
    // }
    // printf("\n");
    // printf("Printing u: \n");
    // for (int i = 0; i < n_eq + n_ineq; i++) {
    //     printf("%f\t", u[i]);
    // }
    // printf("\n");

    free(x);
    free(active_ineq_temp);
    free_tensor(A, n_eq);
    free_tensor(C, n_ineq);
    free(active_ineq);
    free(mu);
    free(lam);
    free(b);
    free(Cx);
    free(temp);
    free(d);
    free(ATmu);
    free(Hx);
    free(CTlam);
    free_tensor(AT, nx);
    free_tensor(CT, nx);
    // free_tensor(zeros, n_eq + n_ineq);
    free_tensor(A_temp, n_eq + n_ineq);
}

int main()
{
    // int nx = 100;
    // int n_eq = 8;
    // int n_ineq = 5;
    int num_seeds = 4;
    unsigned int* seeds = (unsigned int*)malloc(num_seeds * sizeof(unsigned int));
    for (int i = 0; i < num_seeds; i++) {
        seeds[i] = i;
    }

    float nx_list[10] = {
        25.0, 
        34.87376984906053, 
        48.64719293940974, 
        67.86044041487268, 
        94.6619752353662, 
        132.04879751262516, 
        184.2015749320193, 
        256.9521332005488, 
        358.4355822184434, 
        500.0
    };

    bool verbose = false;
    bool warm_starting = true;
    bool scaling = false;
    float rho = 0.1;
    float rho_min = 1e-6;
    float rho_max = 1000000.0;
    float sigma = 1e-6;
    bool adaptive_rho = true;
    int adaptive_rho_interval = 1;
    float adaptive_rho_tolerance = 5;
    int max_iter = 4000;
    float eps_abs = 1e-3;
    float eq_tol = 1e-6;
    int check_interval = 25;

    float** solve_times = create_matrix(10, num_seeds);
    Results* solve_results = (Results*) malloc(sizeof(Results));
    for (int nx_idx = 0; nx_idx < 10; nx_idx++) {
        printf("On nx_idx: %d\n", nx_idx);
        int nx = (int) nx_list[nx_idx];
        int n_eq = (int) (nx_list[nx_idx] / 4.);
        int n_ineq = (int) (nx_list[nx_idx] / 4.);

        int nc = n_eq + n_ineq;
        float** H = create_matrix(nx, nx);
        float** A = create_matrix(nc, nx);
        float* g = create_vector(nx);
        float* u = create_vector(nc);
        float* l = create_vector(nc);

        for (int seed_idx = 0; seed_idx < num_seeds; seed_idx++) {
            printf("On seed_idx: %d\n", seed_idx);
            srand(seeds[seed_idx]);
            generate_random_QP(nx, n_eq, n_ineq, H, g, A, l, u);
            ReLU_QP* relu_qp = Initialize_ReLU_QP(
                H, g, A, l, u,
                warm_starting,
                scaling,
                rho,
                rho_min,
                rho_max,
                sigma,
                adaptive_rho,
                adaptive_rho_interval,
                adaptive_rho_tolerance,
                max_iter,
                eps_abs,
                check_interval,
                verbose,
                eq_tol,
                nc,
                nx
            );
            solve_results = solve(relu_qp);

            // model.results.info.run_time/1e6
            // solve_times[seed_idx + nx_idx * 10] = solve_results->info->run_time;
            solve_times[nx_idx][seed_idx] = solve_results->info->run_time;
            printf("The solve time was: %f\n", solve_results->info->run_time);
        }

        free_tensor(H, nx);
        free_tensor(A, nc);
        free(g);
        free(u);
        free(l);
    }

    printf("Printing solve times...\n");
    print_matrix(10, num_seeds, solve_times);

    




    // // srand(seed);
    
    // // int nc = 5;
    // // float** H = get_H(nx);
    // // float** A = get_A(nc, nx);
    // // float* g = get_g(nx);
    // // float* u = get_u(nc);
    // // float* l = get_l(nc);

    // int nc = n_eq + n_ineq;
    // float** H = create_matrix(nx, nx);
    // float** A = create_matrix(nc, nx);
    // float* g = create_vector(nx);
    // float* u = create_vector(nc);
    // float* l = create_vector(nc);
    // generate_random_QP(nx, n_eq, n_ineq, H, g, A, l, u);

    // // printf("Printing Variables: \n");
    // // print_matrix(nx, nx, H);
    // // print_matrix(nc, nx, A);
    // // print_vector(nx, g);
    // // print_vector(nc, l);
    // // print_vector(nc, u);
    // // printf("----------------------\n");
    
    // bool verbose = false;
    // bool warm_starting = true;
    // bool scaling = false;
    // float rho = 0.1;
    // float rho_min = 1e-6;
    // float rho_max = 1000000.0;
    // float sigma = 1e-6;
    // bool adaptive_rho = true;
    // int adaptive_rho_interval = 1;
    // float adaptive_rho_tolerance = 5;
    // int max_iter = 4000;
    // float eps_abs = 1e-3;
    // float eq_tol = 1e-6;
    // int check_interval = 25;
    
    // ReLU_QP* relu_qp = (ReLU_QP*)malloc(sizeof(ReLU_QP));
    // Results* solve_results = (Results*) malloc(sizeof(Results));

    // for (int i = 0; i < 10; i++) {
    //     generate_random_QP(nx, n_eq, n_ineq, H, g, A, l, u);
    //     relu_qp = Initialize_ReLU_QP(
    //         H, g, A, l, u,
    //         warm_starting,
    //         scaling,
    //         rho,
    //         rho_min,
    //         rho_max,
    //         sigma,
    //         adaptive_rho,
    //         adaptive_rho_interval,
    //         adaptive_rho_tolerance,
    //         max_iter,
    //         eps_abs,
    //         check_interval,
    //         verbose,
    //         eq_tol,
    //         nc,
    //         nx
    //     );
    //     solve_results = solve(relu_qp);
    //     printf("The setup time taken is: %f\n", solve_results->info->setup_time);
    //     printf("The solve time taken is: %f\n", solve_results->info->solve_time);
    // }

    // generate_random_QP(nx, n_eq, n_ineq, H, g, A, l, u);
    // relu_qp = Initialize_ReLU_QP(
    //     H, g, A, l, u,
    //     warm_starting,
    //     scaling,
    //     rho,
    //     rho_min,
    //     rho_max,
    //     sigma,
    //     adaptive_rho,
    //     adaptive_rho_interval,
    //     adaptive_rho_tolerance,
    //     max_iter,
    //     eps_abs,
    //     check_interval,
    //     verbose,
    //     eq_tol,
    //     nc,
    //     nx
    // );
    // solve_results = solve(relu_qp);
    // printf("The result x is: ");
    // for (int i = 0; i < nx; i++) {
    //     printf("%f ", solve_results->x[i]);
    // }
    // printf("\n");
    // printf("The solve time was: %f\n", solve_results->info->solve_time);
    // printf("The setup time was: %f\n", solve_results->info->setup_time);
    // printf("the number of iterations was: %d\n", solve_results->info->iter);

    // // Results* solve_results = solve(relu_qp);
    // // // printf("The solve time taken is: %f\n", solve_results->info->solve_time);
    // // printf("The result x is: ");
    // // for (int i = 0; i < nx; i++) {
    // //     printf("%f ", solve_results->x[i]);
    // // }
    // struct timeval start_time;
    // struct timeval end_time;
    // gettimeofday(&start_time, NULL);
    // for (int i = 0; i < 1000; i++) {
    //     solve_results = solve(relu_qp);
    // }
    // gettimeofday(&end_time, NULL);
    // float elapsedTime = (float)(end_time.tv_sec - start_time.tv_sec);
    // elapsedTime += (((float)(end_time.tv_usec - start_time.tv_usec)) / 1000000.0);
    // float avg_time = elapsedTime / 1000.;
    // printf("The average time taken is: %f\n", avg_time);
    
    // // freeQP(qp);
    // free_tensor(H, nx);
    // free_tensor(A, nc);
    // free(g);
    // free(l);
    // free(u);
    // // free_Settings(settings);
    // // free_Info(info);
    // // free_Results(results);
    // // free_ReLU_Layer(layers);

    // // for(int loop = 0; loop < 10; loop++)
    // //   printf("%f ", array[loop]);

    // printf("Hello World");

    return 0;
}
