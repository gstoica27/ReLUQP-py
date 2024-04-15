#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdbool.h> 
// #include "/home/hice1/gstoica3/courses/6679/project/ReLUQP-py/reluqp/Functions.cu"
// You will need to download https://github.com/troydhanson/uthash/archive/master.zip and then just 
// unzip and replace my basedir path with your path to the uthash-master directory.
#include "/home/hice1/gstoica3/courses/6679/project/ReLUQP-py/packages/uthash-master/include/uthash.h"
#include <signal.h>
#include <float.h>
#include <sys/time.h>


/// @brief Declare all structs ///
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


QP* InitializeQP(
    double** H, double* g, double**A, double* l, double* u, int nx, int nc, int nf
) {
    QP* qp = (QP*)malloc(sizeof(QP));
    qp->H = H;
    qp->g = g;
    qp->A = A;
    qp->l = l;
    qp->u = u;
    qp->nx = nx;
    qp->nc = nc;
    qp->nf = nf;
    return qp;
}


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

Settings* InitializeSettings(
    bool verbose, 
    bool warm_starting,
    bool scaling,
    double rho,
    double rho_min,
    double rho_max,
    double sigma,
    bool adaptive_rho,
    int adaptive_rho_interval,
    double adaptive_rho_tolerance,
    int max_iter,
    double eps_abs,
    double eq_tol,
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
    double obj_val;
    double pri_res;
    double dua_res;
    double setup_time;
    double solve_time;
    double update_time;
    double run_time;
    double rho_estimate;
} Info;


Info* InitializeInfo(
    int iter,
    double obj_val,
    double pri_res,
    double dua_res,
    double setup_time,
    double solve_time,
    double update_time,
    double run_time,
    double rho_estimate
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
    double x;
    double z;
    Info* info;
} Results;

Results* InitializeResults(
    double x,
    double z,
    Info* info
) {
    Results* results = (Results*) malloc(sizeof(Results));
    results->x = x;
    results->z = z;
    results->info = info;
    return results;
}


/// Create variable setting functions

double** get_H(int nx, int nf) {
    double** H = (double**)malloc(nx * sizeof(double*));
    for (int i=0; i< nx; i++) {
        H[i] = (double*)malloc(nf * sizeof(double));
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

double* get_g(int nx) {
    double* g = (double*)malloc(nx * sizeof(double));
    g[0] = -8.;
    g[1] = -3.;
    g[2] = -3.;
    return g;
}

double** get_A(int nc, int nf) {
    double** A = (double**)malloc(nc * sizeof(double*));
    for (int i=0; i < nc; i++) {
        A[i] = (double*)malloc(nf * sizeof(double));
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

double* get_l(int nc) {
    double* l = (double*) malloc(nc * sizeof(double));
    l[0] = 3;
    l[1] = 0;
    l[2] = -10;
    l[3] = -10;
    l[4] = -10;
    return l;
}

double* get_u(int nc) {
    double* u = (double*) malloc(nc * sizeof(double));
    u[0] = 3;
    u[1] = 0;
    u[2] = INFINITY;
    u[3] = INFINITY;
    u[4] = INFINITY;
    return u;
}


void vector_subtract(double* source, double* amount, double* dest, int dim) {
    for (int i = 0; i < dim; i++) {
        dest[i] = source[i] - amount[i];
    }
}

void vector_add(double* source, double* amount, double* dest, int dim) {
    for (int i = 0; i < dim; i++) {
        dest[i] = source[i] + amount[i];
    }
}


void vector_where(bool* conditional, double* vector, double when_true, double when_false, int dim) {
    for (int i = 0; i < dim; i++) {
        if (conditional[i]) {
            vector[i] = when_true;
        }
        else {
            vector[i] = when_false;
        }
    }
}

double** create_diagonal_matrix(double* vector, int dim) {
    double** matrix = (double**)calloc(dim, dim * sizeof(double*));
    for (int i = 0; i < dim; i++) {
        matrix[i] = (double*)calloc(dim, sizeof(double));
        matrix[i][i] = vector[i];
    }
    return matrix;
}

double** create_scalar_diagonal_matrix(double w, int dim) {
    double** matrix = (double**)calloc(dim, dim * sizeof(double*));
    for (int i = 0; i < dim; i++) {
        matrix[i] = (double*)calloc(dim, sizeof(double));
        matrix[i][i] = w;
    }
    return matrix;
}

double** transpose_matrix(double** matrix, int num_row, int num_col) {
    double** transpose = (double**)malloc(num_col * sizeof(double*));
    for (int i = 0; i < num_col; i++) {
        transpose[i] = (double*)malloc(num_row * sizeof(double));
        for (int j = 0; j < num_row; j++) {
            transpose[i][j] = matrix[j][i];
        }
    }
    return transpose;
}

void add_value_to_matrix_inplace(double** matrix, double value, int num_row, int num_col) {
    for (int i = 0; i < num_row; i++) {
        for (int j = 0; j < num_col; j++) {
            matrix[i][j] = matrix[i][j] + value;
        }
    }
}

void add_value_to_matrix(double** matrix, double** dest, double value, int num_row, int num_col) {
    for (int i = 0; i < num_row; i++) {
        for (int j = 0; j < num_col; j++) {
            dest[i][j] = matrix[i][j] + value;
        }
    }
}

double** copy_matrix(double** matrix, int num_row, int num_col) {
    double** copy = (double**)calloc(num_row, num_col * sizeof(double*));
    for (int i = 0; i < num_row; i++) {
        copy[i] = (double*)calloc(num_col, sizeof(double));
        for (int j = 0; j < num_col; j++) {
            copy[i][j] = matrix[i][j];
        }
    }
    return copy;
}

double** create_matrix(int num_row, int num_col) {
    double** H = (double**)malloc(num_row * sizeof(double*));
    for (int i=0; i< num_row; i++) {
        H[i] = (double*)malloc(num_col * sizeof(double));
    }
    return H;
}

double* create_vector(int dim) {
    double* vector = (double*)malloc(dim * sizeof(double));
    return vector;
}


void matmul(double** matrix1, double** matrix2, double** result, int left, int nelem, int right)
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

void matvecmul(double** matrix, double* vector, double* result, int left, int nelem) {
    for (int i = 0; i < left; i++) {
        for (int j = 0; j < nelem; j++) {
            result[i] += matrix[i][j] * vector[j];
        }
    }
}


void add_matrices(double** A, double** B, double** result, int num_row, int num_col) {
    for (int i = 0; i < num_row; i++) {
        for (int k = 0; k < num_col; k++) {
            result[i][k] =A[i][k] + B[i][k];
        }
    }
}

void subtract_matrices(double** A, double** B, double** result, int num_row, int num_col) {
    for (int i = 0; i < num_row; i++) {
        for (int k = 0; k < num_col; k++) {
            result[i][k] = A[i][k] - B[i][k];
        }
    }
}

void scalar_multiply_matrix(double** matrix, double scalar, double** dest, int num_row, int num_col) {
    for (int i = 0; i < num_row; i++) {
        for (int j = 0; j < num_col; j++) {
            dest[i][j] = matrix[i][j] * scalar;
        }
    }
}

void flatten_matrix(double** matrix, double* dest, int num_row, int num_col) {
    for (int i = 0; i < num_row; i++) {
        for (int j = 0; j < num_col; j++) {
            dest[i*num_col + j] = matrix[i][j];
        }
    }
}

void divide_matrices(double** A, double** B, double** result, int num_row, int num_col) {
    for (int i = 0; i < num_row; i++) {
        for (int j = 0; j < num_col; j++) {
            result[i][j] = A[i][j] / B[i][j];
        }
    }
}

void divide_diag(double** A, double** B, double** result, int num_row) {
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
int LUPDecompose(double **A, int N, double Tol, int *P) {

    int i, j, k, imax; 
    double maxA, *ptr, absA;

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
void LUPSolve(double **A, int *P, double *b, int N, double *x) {

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
void LUPInvert(double **A, int *P, int N, double **IA) {
  
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
double LUPDeterminant(double **A, int *P, int N) {

    double det = A[0][0];

    for (int i = 1; i < N; i++)
        det *= A[i][i];

    return (P[N] - N) % 2 == 0 ? det : -det;
}

void compute_matrix_inverse(double** A, double** IA, int N) {
    int *P = (int*)malloc((N+1) * sizeof(int));
    LUPDecompose(A, N, 0.0001, P);
    LUPInvert(A, P, N, IA);
    free(P);
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


void free_tensor(double** tensor, int num_rows) {
    for (int i = 0; i < num_rows; i++) {
        free(tensor[i]);
    }
}


void merge(double* arr, double* temp, int left, int mid, int right) 
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
void mergeSortCPU(double* arr, double* temp, int left, int right) 
{
    if (left >= right)
        return;

    int mid = left + (right - left) / 2;

    mergeSortCPU(arr, temp, left, mid);
    mergeSortCPU(arr, temp, mid + 1, right);

    merge(arr, temp, left, mid, right);
}

void print_matrix(int r, int c, float matrix[r][c])
{
    for (int i = 0; i < r; i++)
    {
        for (int j = 0; j < c; j++) {
            printf("%.3f ", matrix[i][j]);
        }
        printf("\n");
    }
}


int argmin(double* array, int n) {
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


double* vector_abs(double* arr, int dim) {
    double* abs_arr = (double*)malloc(dim * sizeof(double));
    for (int i = 0; i < dim; i++) {
        abs_arr[i] = fabs(arr[i]);
    }
    return abs_arr;

}


double* vector_subtract_scalar(double* vector, double scalar, int dim) {
    double* dest = (double*)malloc(dim * sizeof(double));
    for (int i = 0; i < dim; i++) {
        dest[i] = vector[i] - scalar;
    }
    return dest;
}


void MatrixInverse3x3(double** m, double** minv) {
    // computes the inverse of a matrix m
    double det = m[0][0] * (m[1][1] * m[2][2] - m[2][1] * m[1][2]) -
                m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0]) +
                m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);

    double invdet = 1 / det;
    minv[0][0] = (m[1][1] * m[2][2] - m[2][1] * m[1][2]) * invdet;
    minv[0][1] = (m[0][2] * m[2][1] - m[0][1] * m[2][2]) * invdet;
    minv[0][2] = (m[0][1] * m[1][2] - m[0][2] * m[1][1]) * invdet;
    minv[1][0] = (m[1][2] * m[2][0] - m[1][0] * m[2][2]) * invdet;
    minv[1][1] = (m[0][0] * m[2][2] - m[0][2] * m[2][0]) * invdet;
    minv[1][2] = (m[1][0] * m[0][2] - m[0][0] * m[1][2]) * invdet;
    minv[2][0] = (m[1][0] * m[2][1] - m[2][0] * m[1][1]) * invdet;
    minv[2][1] = (m[2][0] * m[0][1] - m[0][0] * m[2][1]) * invdet;
    minv[2][2] = (m[0][0] * m[1][1] - m[1][0] * m[0][1]) * invdet;
}

// Function to concatenate two matrices in C
double** concatenate_matrices(double** mat1, int rows1, int cols1, double** mat2, int rows2, int cols2, int dim, int out_rows, int out_cols) {
    double** result;

    if (dim == 1) { // Horizontal concatenation
        if (rows1 != rows2) {
            return NULL; // Incompatible dimensions
        }
        out_rows = rows1;
        out_cols = cols1 + cols2;
        result = (double**)malloc(out_rows * sizeof(double*));
        for (int i = 0; i < out_rows; i++) {
            result[i] = (double*)malloc(out_cols * sizeof(double));
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
        result = (double**)malloc(out_rows * sizeof(double*));
        for (int i = 0; i < rows1; i++) {
            result[i] = (double*)malloc(out_cols * sizeof(double));
            for (int j = 0; j < out_cols; j++) {
                result[i][j] = mat1[i][j];
            }
        }
        for (int i = 0; i < rows2; i++) {
            result[i + rows1] = (double*)malloc(out_cols * sizeof(double));
            for (int j = 0; j < out_cols; j++) {
                result[i + rows1][j] = mat2[i][j];
            }
        }
    } else {
        return NULL; // Invalid dimension
    }

    return result;
}



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
    double* rhos = (double*)malloc(1 * sizeof(double));
    rhos[0] = settings->rho;
    relu_layer->rhos = rhos;
    relu_layer->rhos_len = 1;
    if (settings->adaptive_rho) {
        double rho = settings->rho / settings->adaptive_rho_tolerance;
        // double* rhos;
        int i = 1;
        while (rho >= settings->rho_min) {
            rhos = (double*)realloc(rhos, (i+1) * sizeof(double));
            rhos[i] = rho;
            rho = rho / settings->adaptive_rho_tolerance;
            i++;
        }
        rho = rho*settings->adaptive_rho_tolerance;
        while (rho <= settings->rho_max) {
            rhos = (double*)realloc(rhos, (i+1) * sizeof(double));
            rhos[i] = rho;
            rho = rho * settings->adaptive_rho_tolerance;
            i++;
        }
        relu_layer->rhos = rhos;
        relu_layer->rhos_len = i;
        // Sort
        double* temp = (double*)malloc(relu_layer->rhos_len * sizeof(double));
        mergeSortCPU(relu_layer->rhos, temp, 0, relu_layer->rhos_len - 1);
    }
    
    // setup matrices
    double** H = qp->H; 
    double** A = qp->A;
    double* g = qp->g;
    double* l = qp->l;
    double* u = qp->u;
    int nx = qp->nx;
    int nc = qp->nc;
    int nf = qp->nf; 
    double sigma = settings->sigma;

    double*** kkts_rhs_invs = (double***)malloc(relu_layer->rhos_len * sizeof(double**));
    for (int i = 0; i < relu_layer->rhos_len; i++) {
        kkts_rhs_invs[i] = (double**)malloc(nf * sizeof(double*));
        for (int j = 0; j < nf; j++) {
            kkts_rhs_invs[i][j] = (double*)malloc(nf * sizeof(double));
        }
    }
    // double kkts_rhs_invs[relu_layer->rhos_len][nf][nf];
    double* flag_checks = (double*)calloc(nc, sizeof(double));
    vector_subtract(u, l, flag_checks, nc);
    bool* conditional = (bool*)calloc(nc, sizeof(bool));
    for (int i = 0; i < nc; i++) {
        conditional[i] = flag_checks[i] <= settings->eq_tol;
        printf("flag_checks: %d\n", conditional[i]);
    }
    free(flag_checks);
    

    for (int i = 0; i < relu_layer->rhos_len; i++) {
        double rho_scalar = relu_layer->rhos[i];
        double* rho = (double*)calloc(nc, sizeof(double));
        for (int j = 0; j < nc; j++) {
            rho[j] = rho_scalar;
        }
        vector_where(conditional, rho, rho_scalar * 1e3, rho_scalar, nc);
        double** rho_mat = create_diagonal_matrix(rho, nc);
        double* sigma_vector = (double*)calloc(nf, sizeof(double));
        for (int j = 0; j < nf; j++) {
            sigma_vector[j] = sigma;
        }
        double** sigma_mat = create_diagonal_matrix(sigma_vector, nf);
        
        free(sigma_vector);
        double** A_transpose = transpose_matrix(A, nc, nf);
        double** rho_A = create_matrix(nc, nf);
        matmul(rho_mat, A, rho_A, nc, nf, nf);
        double** AT_rho_A = create_matrix(nf, nf);
        matmul(A_transpose, rho_A, AT_rho_A, nf, nc, nf);
        // // free variables
        free_tensor(rho_A, nc);
        free_tensor(A_transpose, nf);

        double** summed_mat = create_matrix(nf, nf);
        add_matrices(H, sigma_mat, summed_mat, nf, nf);
        add_matrices(summed_mat, AT_rho_A, summed_mat, nf, nf);
        // need to take inverse now.... of summed mats
        double** summed_mat_inv = create_matrix(nf, nf);
        // compute_matrix_inverse(summed_mat, summed_mat_inv, nf);
        MatrixInverse3x3(summed_mat, summed_mat_inv);
        for (int a = 0; a < nf; a++) {
            for (int b = 0; b < nf; b++) {
                kkts_rhs_invs[i][a][b] = summed_mat_inv[a][b];
                // printf("a: %d, b: %d, val: %lf\n", a, b, summed_mat[a][b]);
            }
        }
    }

    // Define W_ks, B_ks, b_ks
    int W_Row = nf + 2 * nc;
    double*** W_ks = (double***)malloc(relu_layer->rhos_len * sizeof(double**));
    double*** B_ks = (double***)malloc(relu_layer->rhos_len * sizeof(double**));
    double** b_ks = (double**)malloc(relu_layer->rhos_len * sizeof(double*));
    for (int i = 0; i < relu_layer->rhos_len; i++) {
        W_ks[i] = (double**)malloc(W_Row * sizeof(double*));
        B_ks[i] = (double**)malloc(W_Row * sizeof(double*));
        b_ks[i] = (double*)malloc(W_Row * sizeof(double));
        for (int j = 0; j < nf + 2 * nc; j++) {
            W_ks[i][j] = (double*)malloc(W_Row * sizeof(double));
        }
        for (int j = 0; j < nf; j++) {
            B_ks[i][j] = (double*)malloc(nf * sizeof(double));
        }
    }

    for (int rho_ind = 0; rho_ind < relu_layer->rhos_len; rho_ind++) {
        double rho_scalar = relu_layer->rhos[rho_ind];
        double* rho = (double*)calloc(nc, sizeof(double));
        for (int j = 0; j < nc; j++) {
            rho[j] = rho_scalar;
        }
        vector_where(conditional, rho, rho_scalar * 1e3, rho_scalar, nc);
        double** rho_mat = create_diagonal_matrix(rho, nc);
        double* sigma_vector = (double*)calloc(nf, sizeof(double));
        for (int j = 0; j < nf; j++) {
            sigma_vector[j] = sigma;
        }
        double** Ix = create_diagonal_matrix(sigma_vector, nf);
        free(sigma_vector);
        double* ones_vector = (double*)calloc(nf, sizeof(double));
        for (int j = 0; j < nf; j++) {
            ones_vector[j] = 1;
        }
        double** Ic = create_diagonal_matrix(ones_vector, nc);

        // CREATING W_ks elements!!!!

        // Create (0, 0) element
        // K @ (sigma * Ix - A.T @ (rho @ A))
        double** A_transpose = transpose_matrix(A, nc, nf);
        double** rho_A = create_matrix(nc, nf);
        matmul(rho_mat, A, rho_A, nc, nf, nf);
        double** AT_rho_A = create_matrix(nf, nf);
        matmul(A_transpose, rho_A, AT_rho_A, nf, nc, nf);
        double** summed_mat = create_matrix(nf, nf);
        subtract_matrices(Ix, AT_rho_A, summed_mat, nf, nf);
        double** elem_00 = create_matrix(nf, nf);
        matmul(summed_mat, kkts_rhs_invs[rho_ind], elem_00, nf, nf, nf);
        // Create (0, 1) element
        // 2 * K @ A.T @ rho
        double** AT_rho = create_matrix(nf, nc);
        matmul(A_transpose, rho_mat, AT_rho, nf, nc, nc); // [nf,nc]
        double** K_AT_rho = create_matrix(nf, nc);
        matmul(kkts_rhs_invs[rho_ind], AT_rho, K_AT_rho, nf, nf, nc); // [nf, nc]
        scalar_multiply_matrix(K_AT_rho, 2, K_AT_rho, nf, nc);
        double** elem_01 = copy_matrix(K_AT_rho, nf, nc);
        // Create (0, 2) element
        // -K @ A.T
        double** neg_I = create_scalar_diagonal_matrix(-1, nf);
        double** neg_K = create_matrix(nf, nf);
        matmul(neg_I, kkts_rhs_invs[rho_ind], neg_K, nf, nf, nf);
        double** elem_02 = create_matrix(nf, nc);
        matmul(neg_K, A_transpose, elem_02, nf, nf, nc);
        // Create (1, 0) element
        // A @ K @ (sigma * Ix - A.T @ (rho @ A)) + A
        // A @ elem_00 + A
        double** A_elem00 = create_matrix(nc, nf);
        matmul(A, elem_00, A_elem00, nc, nf, nf);
        double** elem_10 = create_matrix(nc, nf);
        add_matrices(A_elem00, A, elem_10, nc, nf);
        // Create (1, 1) element
        // 2 * A @ K @ A.T @ rho - Ic
        // 2 * partial_elem01 - Ic
        double** A_K_AT_rho = create_matrix(nc, nc);
        matmul(A, K_AT_rho, A_K_AT_rho, nc, nf, nc);
        scalar_multiply_matrix(A_K_AT_rho, 2, A_K_AT_rho, nc, nc);
        double** elem_11 = create_matrix(nc, nc);
        subtract_matrices(A_K_AT_rho, Ic, elem_11, nc, nc);
        // Create (1, 2) element
        // -A @ K @ A.T + rho_inv
        double** K_AT = create_matrix(nf, nc);
        matmul(kkts_rhs_invs[rho_ind], A_transpose, K_AT, nf, nf, nc);
        double** A_K_AT = create_matrix(nc, nc);
        matmul(A, K_AT, A_K_AT, nc, nf, nc);
        scalar_multiply_matrix(A_K_AT, -1, A_K_AT, nc, nc);
        double** rho_inv = create_scalar_diagonal_matrix(1.0, nc);
        divide_diag(rho_inv, rho_mat, rho_inv, nc);
        double** elem_12 = create_matrix(nc, nc);
        add_matrices(A_K_AT, rho_inv, elem_12, nc, nc);
        // Create (2, 0) element
        // rho @ A
        double** elem_20 = create_matrix(nc, nf);
        matmul(rho_mat, A, elem_20, nc, nc, nf);
        // Create (2, 1) element                                                    [nc, nf]
        // -rho
        double** elem_21 = create_matrix(nc, nc);
        scalar_multiply_matrix(rho_mat, -1., elem_21, nc, nf);
        // Create (2, 2) element                                                    [nc, nc]
        // Ic
        double** elem_22 = copy_matrix(Ic, nc, nc); //                              [nc, nc]
        // W_ks is of size:
        // [nf, nf + nc + nc]
        // [nc, nf + nc + nc]
        // [nc, nf + nc + nc]
        // [nf + 2*nc, nf + 2*nc]
        // [elem_00, elem_01, elem_02]
        // [elem_10, elem_11, elem_12]
        // [elem_20, elem_21, elem_22]
        double** elem_00_01 = concatenate_matrices(elem_00, nf, nf, elem_01, nf, nc, 1, nf, nf+nc);
        double** row_0 = concatenate_matrices(elem_00_01, nf, nf+nc, elem_02, nf, nc, 1, nf, nf+2*nc);
        double** elem_10_11 = concatenate_matrices(elem_10, nc, nf, elem_11, nc, nc, 1, nc, nf+nc);
        double** row_1 = concatenate_matrices(elem_10_11, nc, nf+nc, elem_12, nc, nc, 1, nc, nf+2*nc);
        double** elem_20_21 = concatenate_matrices(elem_20, nc, nf, elem_21, nc, nc, 1, nc, nf+nc);
        double** row_2 = concatenate_matrices(elem_20_21, nc, nf+nc, elem_22, nc, nc, 1, nc, nf+2*nc);
        double** rows_01 = concatenate_matrices(row_0, nf, nf+2*nc, row_1, nc, nf+2*nc, 0, nf+nc, nf+2*nc);
        double** rows_012 = concatenate_matrices(rows_01, nf+nc, nf+2*nc, row_2, nc, nf+2*nc, 0, nf+2*nc, nf+2*nc);
        W_ks[rho_ind] = rows_012;
        int D = nf + 2*nc;
        // for (int a = 0; a < D; a++) {
        //     for (int b = 0; b < D; b++) {
        //         printf("a: %d, b: %d, val: %lf\n", a, b,rows_012[a][b]);
        //     }
        // }
        // FREE TENSORS
        // free_tensor(rho_A, nc);
        // free_tensor(A_transpose, nf);
        // Creating the B_ks
        double** neg_AK = create_matrix(nc, nf);
        double** neg_A = create_matrix(nc, nf);
        scalar_multiply_matrix(A, -1, neg_A, nc, nf);
        matmul(neg_A, kkts_rhs_invs[rho_ind], neg_AK, nc, nf, nf);
        double** zeros = create_matrix(nc, nf);
        double** elems_01 = concatenate_matrices(neg_K, nf, nf, neg_AK, nc, nf, 0, nf+nc, nf);
        double** elems_012 = concatenate_matrices(elems_01, nf+nc, nf, zeros, nc, nf, 0, nf+2*nc, nf);
        B_ks[rho_ind] = elems_012;
        // Creating the b_ks
        double* b_k = create_vector(nf + 2*nc);
        matvecmul(B_ks[rho_ind], g, b_k, nf+2*nc, nf);
        // matmul(B_ks[rho_ind], g, b_k, nf+2*nc, nf, 1);
        b_ks[rho_ind] = b_k;
    }
    relu_layer->W_ks = W_ks;
    relu_layer->B_ks = B_ks;
    relu_layer->b_ks = b_ks;
    relu_layer->clamp_left = qp->nx;
    relu_layer->clamp_right = qp->nx + qp->nc;
    // // print kkts_rhs_invs
    // for (int i = 0; i < relu_layer->rhos_len; i++) {
    //     for (int a = 0; a < nf; a++) {
    //         for (int b = 0; b < nf; b++) {
    //             printf("a: %d, b: %d, val: %lf\n", a, b, kkts_rhs_invs[i][a][b]);
    //         }
    //     }
    // }

    return relu_layer;
}

double* ReLU_Layer_Forward(ReLU_Layer* layer, double* x, int idx) {
    double **W = layer->W_ks[idx];
    double *b = layer->b_ks[idx];
    double *l = layer->qp->l;
    double *u = layer->qp->u;
    int idx1 = layer->clamp_left;
    int idx2 = layer->clamp_right;
    int nx = layer->qp->nx;
    int nc = layer->qp->nc;

    int D = nx + 2*nc;
    double* intermediate = create_vector(D);
    matvecmul(W, x, intermediate, D, D);
    vector_add(intermediate, b, intermediate, D);

    for (int i = idx1; i < idx2; i++) {
        double val = intermediate[i];
        if (val < l[i - idx1]) {
            intermediate[i] = l[i - idx1];
        } else if (val > u[i - idx1]) {
            intermediate[i] = u[i - idx1];
        }
    }
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
    // double* x;
    // double* z;
    // double* lam;
    double* output;
    int rho_ind;
} ReLU_QP;


ReLU_QP* Initialize_ReLU_QP(
    double** H, double* g, double** A, double* l, double* u,
    bool warm_starting, 
    bool scaling,
    double rho,
    double rho_min,
    double rho_max,
    double sigma, 
    bool adaptive_rho,
    int adaptive_rho_interval,
    int adaptive_rho_tolerance,
    int max_iter,
    double eps_abs,
    int check_interval,
    bool verbose,
    double eq_tol,
    int nc,
    int nf,
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
    relu_qp->results = InitializeResults(0,0,relu_qp->info);
    relu_qp->qp = InitializeQP(H, g, A, l, u, nx, nc, nf);
    relu_qp->layers = Initialize_ReLU_Layer(relu_qp->qp, relu_qp->settings);
    // double* x = create_vector(nx);
    // double* z = create_vector(nc);
    // double* lam = create_vector(nc);
   relu_qp->output = create_vector(nx + 2*nc);
    // self.rho_ind = np.argmin(np.abs(self.layers.rhos.cpu().detach().numpy() - self.settings.rho))
    double* rhos_minus_rho = vector_subtract_scalar(relu_qp->layers->rhos, relu_qp->settings->rho, relu_qp->layers->rhos_len);
    relu_qp->rho_ind = argmin(vector_abs(rhos_minus_rho, relu_qp->layers->rhos_len), relu_qp->layers->rhos_len);
    gettimeofday(&relu_qp->end, NULL);
    double elapsedTime = (double)(relu_qp->end.tv_sec - relu_qp->start.tv_sec) * 1000.0;
    elapsedTime += (((double)(relu_qp->end.tv_usec - relu_qp->start.tv_usec)) / 1000.0) / 1000.;
    relu_qp->info->setup_time = elapsedTime;
    return relu_qp;
}








int main()
{
    int nx = 3;
    int nc = 5;
    int nf = 3;
    // double H[3][3] = {{6, 2, 1}, {2, 5, 2}, {1, 2, 4.0}};
    // double A[5][3] = { {1, 0, 1}, {0, 1, 1}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1} };
    // double l[5] = {3.0, 0, -10., -10, -10 };
    // double u[5] = {3., 0, INFINITY, INFINITY, INFINITY};
    // double g[3] = {-8.0, -3., -3.};
    
    double** H = get_H(nx, nf);
    double** A = get_A(nc, nf);
    double* g = get_g(nx);
    double* u = get_u(nc);
    double* l = get_l(nc);
    
    bool verbose = false;
    bool warm_starting = true;
    bool scaling = false;
    double rho = 0.1;
    double rho_min = 1e-6;
    double rho_max = 1e6;
    double sigma = 1e-6;
    bool adaptive_rho = true;
    int adaptive_rho_interval = 1;
    double adaptive_rho_tolerance = 5;
    int max_iter = 4000;
    double eps_abs = 1e-3;
    double eq_tol = 1e-6;
    int check_interval = 25;
    
    int iter = 500;
    double obj_val = 0.1;
    double pri_res = 0.2;
    double dua_res = 0.3;
    double setup_time = 1;
    double solve_time = 2;
    double update_time = 3;
    double run_time = 4;
    double rho_estimate = 5.;
    
    double x = 420;
    double z = 17;
    
    Settings* settings = InitializeSettings(
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
    
    Info* info = InitializeInfo(
        iter,
        obj_val,
        pri_res,
        dua_res,
        setup_time,
        solve_time,
        update_time,
        run_time,
        rho_estimate
    );
    
    Results* results = InitializeResults(
        x,
        z,
        info
    );

    QP* qp = InitializeQP(H, g, A, l, u, nx, nc, nf);
    printf("qp->nc is: %d\n", qp->nc);
    ReLU_Layer* layers = Initialize_ReLU_Layer(qp, settings);
    printf("Printing rhos array of length %d:\n", layers->rhos_len);
    for (int i = 0; i < layers->rhos_len; i++) {
        // if (i == layers->rhos_len - 1) {
        //     printf("%lf ", layers->rhos[i-1] * 5);
        //     printf("%lf ", layers->rhos[i] + 5);
        // }
        // else {
        //     printf("%lf ", layers->rhos[i]);
        // }
        printf("%lf ", layers->rhos[i]);
    }
    printf("\n");

    double* input = create_vector((nx + 2 * nc));
    for (int i = 0; i < nx + 2 * nc; i++) {
        input[i] = 1;
    }
    double* y = ReLU_Layer_Forward(layers, input, 0);
    for (int i = 0; i < nx + 2 * nc; i++) {
        printf("%lf ", y[i]);
    }
    printf("\n");

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
        nf,
        nx
    );

    printf("The RELU_QP IS:\n");
    printf("ReLU_QP rho_ind: %d\n", relu_qp->rho_ind);
    printf("The setup time taken is: %lf\n", relu_qp->info->setup_time);

    
    freeQP(qp);
    free_tensor(H, nx);
    free_tensor(A, nc);
    free(g);
    free(l);
    free(u);
    free_Settings(settings);
    free_Info(info);
    free_Results(results);
    free_ReLU_Layer(layers);

    // for(int loop = 0; loop < 10; loop++)
    //   printf("%f ", array[loop]);

    printf("Hello World");

    return 0;
}
