#ifndef __CFASTPT_CFASTPT_H
#define __CFASTPT_CFASTPT_H
#ifdef __cplusplus
extern "C" {
#endif

typedef struct fastpt_config 
{
  double nu; // only used in scalar; in tensor, nu1,nu2 are computed by alpha,beta
  double c_window_width;
  long N_pad;
  long N_extrap_low;
  long N_extrap_high;
} fastpt_config;

typedef struct fastpt_todo 
{
  int isScalar;
  double* alpha;
  double* beta;
  double* ell;
  int* isP13type;
  double* coeff_ar;
  int Nterms;
} fastpt_todo;

typedef struct fastpt_todolist 
{
  fastpt_todo *fastpt_todo;
  int N_todo;
} fastpt_todolist;

void fastpt_scalar(int* alpha_ar, int* beta_ar, int* ell_ar, int* isP13type_ar, 
double* coeff_A_ar, int Nterms, double* Pout, double* k, double* Pin, int Nk);

void J_abl_ar(double* x, double* fx, long N, int* alpha, int* beta, int* ell, 
int* isP13type, int Nterms, fastpt_config* config, double** Fy);

void J_abl(double* x, double* fx, int alpha, int beta, long N, 
fastpt_config* config, int ell, double* Fy);

void J_abJ1J2Jk(
    const double *restrict x,      // input k grid, length N (log-spaced wavenumbers)
    const double *restrict fx,     // input power spectrum P(k), length N (changes each call)
    long N,                        // number of input k points (before padding)
    const int *restrict alpha,     // biasing exponent 1 per term: nu1 = -2 - alpha[i]
    const int *restrict beta,      // biasing exponent 2 per term: nu2 = -2 - beta[i]
    const int *restrict J1,        // angular momentum coupling 1 per term (indexes g_m cache)
    const int *restrict J2,        // angular momentum coupling 2 per term (indexes g_m cache)
    const int *restrict Jk,        // angular momentum coupling 3 per term (indexes g_m cache)
    int Nterms,                    // number of terms to compute
    const fastpt_config *restrict config, // padding/windowing config (N_pad, c_window_width, etc.)
    double **restrict Fy           // output: Fy[i][j] = result for term i at k-point j
  );

void Pd1d2(double* k, double* Pin, long Nk, double* Pout);

void Pd2d2(double* k, double* Pin, long Nk, double* Pout);

void Pd1s2(double* k, double* Pin, long Nk, double* Pout);

void Pd2s2(double* k, double* Pin, long Nk, double* Pout);

void Ps2s2(double* k, double* Pin, long Nk, double* Pout);

#ifdef __cplusplus
}
#endif
#endif // HEADER GUARD
