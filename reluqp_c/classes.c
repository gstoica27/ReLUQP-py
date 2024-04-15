#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdbool.h> 


typedef struct {
    double** H;
    double* g;
    double** A;
    double* l;
    double* u;
    int nx;
    int nc;
} QP;


QP* InitializeQP(
    double** H, double* g, double**A, double* l, double* u, int nx, int nc
) {
    QP* qp = (QP*)malloc(sizeof(QP));
    qp->H = H;
    qp->g = g;
    qp->A = A;
    qp->l = l;
    qp->u = u;
    qp->nx;
    qp->nc;
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
    int adaptive_rho_tolerance;
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
    int adaptive_rho_tolerance,
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
}


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


void freeTensor(double** tensor, int num_rows) {
    for (int i = 0; i < num_rows; i++) {
        free(tensor[i]);
    }
}

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
    A[5][2] = 1;
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
    double* g = get_g(nx);
    double** A = get_A(nc, nf);
    double* u = get_u(nc);
    double* l = get_l(nc);
    
    
    QP* qp = InitializeQP(H, g, A, l, u, nx, nc);
    freeQP(qp);
    freeTensor(H, nx);
    freeTensor(A, nc);
    free(g);
    free(l);
    free(u);

    bool verbose = false;
    bool warm_starting = true;
    bool scaling = false;
    double rho = 0.1;
    double rho_min = 1e-6;
    double rho_max = 1e6;
    double sigma = 1e-6;
    bool adaptive_rho = true;
    int adaptive_rho_interval = 1;
    int adaptive_rho_tolerance = 5;
    int max_iter = 4000;
    double eps_abs = 1e-3;
    double eq_tol = 1e-6;
    int check_interval = 25;
    
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
    
    int iter = 500;
    double obj_val = 0.1;
    double pri_res = 0.2;
    double dua_res = 0.3;
    double setup_time = 1;
    double solve_time = 2;
    double update_time = 3;
    double run_time = 4;
    double rho_estimate = 5.;
    
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
    
    double x = 420;
    double z = 17;
    Results* results = InitializeResults(
        x,
        z,
        info
    );
    
    
    free_Settings(settings);
    free_Info(info);
    free_Results(results);
    
    printf("Hello World");

    return 0;
}
