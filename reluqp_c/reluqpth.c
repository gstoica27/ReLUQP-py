#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdbool.h> 
// You will need to download https://github.com/troydhanson/uthash/archive/master.zip and then just 
// unzip and replace my basedir path with your path to the uthash-master directory.
#include "/home/hice1/gstoica3/courses/6679/project/ReLUQP-py/packages/uthash-master/include/uthash.h"



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
    qp->nx;
    qp->nc;
    qp->nf;
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


// Taken from https://stackoverflow.com/questions/58752824/uthash-adding-a-new-entry-to-a-struct-struct-hashmap
typedef struct hash_ptr {
    char* string;
    size_t len;
}hash_ptr;

typedef struct hash_map_entry {
    struct hash_ptr *key;
    struct hash_ptr *value;
    UT_hash_handle hh;
}hash_map_entry;

void add_entry(hash_map_entry **map, hash_ptr *key, hash_ptr *value) {
    hash_map_entry *entry;
    HASH_FIND(hh, *map, key->string, key->len, entry);
    if (entry == NULL) {
        entry = (hash_map_entry*) malloc(sizeof *entry);
        memset(entry, 0, sizeof *entry);
        entry->value = value;
        entry->key = key;
        HASH_ADD_KEYPTR(hh, *map, key->string, key->len, entry);
    }
}
///////////////////////////////////////////////////////////////////

void vector_subtract(double* source, double* amount, double* dest, int dim) {
    for (int i = 0; i < dim; i++) {
        dest[i] = source[i] - amount[i];
    }
}


void vector_where(double* conditional, double* vector, double when_true, double when_false, int dim) {
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

double** transpose_matrix(double** matrix, int num_row, int num_col) {
    double** transpose = (double**)calloc(num_col, num_row * sizeof(double*));
    for (int i = 0; i < num_col; i++) {
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
    double** mat = (double**)calloc(num_row, num_col * sizeof(double*));
    for (int i = 0; i < num_col; i ++ ) {
        mat[i] = (double*)calloc(num_col, sizeof(double));
    }
    return mat;
}


void matmul(double** matrix1, double** matrix2, double** result, int num_row, int num_col)
{
    for (int i = 0; i < num_row; i++) {
        for (int j = 0; j < num_row; j++) {
            result[i][j] = 0;
            for (int k = 0; k < num_col; k++) {
                result[i][j] += matrix1[i][k] * matrix2[k][j];
            }
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
// //////////////////////////////////////////////////////////////////////////////////////////////////////////

typedef struct
{
    QP* qp;
    Settings* settings;
    double* rhos;
    int rhos_len;
} ReLU_Layer;

ReLU_Layer* Initialize_ReLU_Layer (
    QP* qp,
    Settings* settings
) {
    ReLU_Layer* relu_layer = (ReLU_Layer*)malloc(sizeof(ReLU_Layer));
    relu_layer->qp = qp;
    relu_layer->settings = settings;

    // setup rhos
    if (!settings->adaptive_rho) {
        double* rhos = (double*)malloc(1 * sizeof(double));
        rhos[0] = settings->rho;
        relu_layer->rhos = rhos;
        relu_layer->rhos_len = 1;
    }
    else {
        double rho = settings->rho / settings->adaptive_rho_tolerance;
        double* rhos = (double*)calloc(1, sizeof(double));
        // double* rhos;
        int i = 0;
        while (rho >= settings->rho_min) {
            if (i != 0) {
                rhos = (double*)realloc(rhos, (i+1) * sizeof(double));
            }
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






    double** kkts_rhs_invs = (double**)calloc(relu_layer->rhos_len, (double*)malloc(nc * sizeof(double)));
    double* flag_checks = (double*)calloc(nc, sizeof(double));
    vector_subtract(u, l, flag_checks, nc);
    bool* conditional = (bool*)calloc(nc, sizeof(bool));
    for (int i = 0; i < nc; i++) {
        conditional[i] = flag_checks[i] <= settings->eq_tol;
    }
    free(flag_checks);

    for (int i = 0; i < relu_layer->rhos_len; i++) {
        double rho_scalar = relu_layer->rhos[i];
        double* rho = (double*)calloc(nc, sizeof(double));
        for (int j = 0; j < nc; j++) {
            rho[j] = rho_scalar;
        }
        vector_where(conditional, rho, rho_scalar * 1e3, rho_scalar, nc);
        free(conditional);

        double** rho_mat = create_diagonal_matrix(rho, nc);
        // double** H_sigma = copy_matrix(rho_mat, nc, nf);
        // add_value_to_matrix_inplace(H_sigma, sigma, nc, nf);
        double* sigma_vector = (double*)calloc(nf, sizeof(double));
        for (int j = 0; j < nf; j++) {
            sigma_vector[j] = sigma;
        }
        double** sigma_mat = create_diagonal_matrix(sigma_vector, nf);
        free(sigma_vector);
        double** A_transpose = transpose_matrix(A, nc, nf);
        double** rho_A = create_matrix(nc, nf);
        matmul(rho, A, rho_A, nc, nc);
        // TODO: This is completely buggy probably. I don't think I have the correct dimensions here.
        // Maybe need to change the matmul...
        double** AT_rho_A = create_matrix(nf, nf);
        matmul(A_transpose, rho_A, AT_rho_A, nc, nf);
        // free variables
        free_tensor(rho_A, nc);
        free_tensor(A_transpose, nf);

        double** summed_mat = create_matrix(nf, nf);
        add_matrices(H, sigma_mat, summed_mat, nf, nf);
        add_matrices(summed_mat, AT_rho_A, summed_mat, nf, nf);
        // need to take inverse now.... of summed mats
    }


    return relu_layer;
}


// typedef struct ReLU_QP
// {
//     Info* info;
//     Results* results;
//     Settings* settings;
//     QP* qp;
//     int start;
//     int end;
//     double* x;
//     double* z;
//     double* lam;
//     double* output;
//     int rho_ind;
// };


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

void free_ReLU_Layer(ReLU_Layer* layer) {
    free(layer);
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
    
    hash_map_entry *map = NULL;

    hash_ptr *key = (hash_ptr*) malloc(sizeof *key);
    memset(key, 0, sizeof *key);
    key->string = "Is this the Krusty Krab?";
    key->len = strlen(key->string);

    hash_ptr *value = (hash_ptr*) malloc(sizeof *value);
    memset(value, 0, sizeof *value);
    value->string = "No, this is Patrick!";
    value->len = strlen(value->string);

    add_entry(&map, key, value);

    hash_map_entry *find_me;
    HASH_FIND(hh, map, key->string, key->len, find_me);
    if (find_me)
    {
        printf("found key=\"%s\", val=\"%s\"\n", find_me->key->string, find_me->value->string);
    }
    else
    {
        printf("not found\n");
    }


    // for(int loop = 0; loop < 10; loop++)
    //   printf("%f ", array[loop]);

    printf("Hello World");

    return 0;
}
