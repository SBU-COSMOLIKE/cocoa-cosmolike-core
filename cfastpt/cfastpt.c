#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <complex.h>
#include <string.h>
#include <time.h>
#include <fftw3.h>
#include <gsl/gsl_math.h>
#include "cfastpt.h"
#include "utils_cfastpt.h"
#include "utils_complex_cfastpt.h"

#include "../log.c/src/log.h"

#include <omp.h>

void fastpt_scalar(int *alpha_ar, int *beta_ar, int *ell_ar, int *isP13type_ar,
double *coeff_A_ar, int Nterms, double *Pout, double *k, double *Pin, int Nk)
{
  double **Fy;
  Fy = malloc(sizeof(double*) * Nterms);
  for(int i=0;i<Nterms;i++) 
  {
    Fy[i] = malloc(sizeof(double) * Nk);
  }

  fastpt_config config;
  config.nu = -2.;
  config.c_window_width = 0.65; //0.25;
  config.N_pad = 1500;
  config.N_extrap_low = 500;
  config.N_extrap_high = 500;

  J_abl_ar(k, Pin, Nk, alpha_ar, beta_ar, ell_ar, isP13type_ar, Nterms, &config, Fy);

  #pragma omp parallel for
  for(int j=0; j<Nk; j++)
  {
    Pout[j] = 0.;
    for(int i=0; i<Nterms; i++) 
    {
      Pout[j] += coeff_A_ar[i] * Fy[i][j];
    }
  }


  for(int i = 0; i < Nterms; i++) 
  {
    free(Fy[i]);
  }
  free(Fy);
}


void J_abl_ar(double *x, double *fx, long N, int *alpha, int *beta, int *ell, 
int *isP13type __attribute__((unused)), int Nterms, fastpt_config *config, double **Fy) 
{
  // x: k array, fx: Pin array
  const long N_original = N;
  const long N_pad = config->N_pad;
  const long N_extrap_low = config->N_extrap_low;
  const long N_extrap_high = config->N_extrap_high;
  N += (2*N_pad + N_extrap_low+N_extrap_high);

  if(N % 2) 
  {
    log_fatal("cfastpt.c: J_abl_ar: Please use even number of x !");
    exit(0);
  }
  const long halfN = N/2;

  const double x0 = x[0];
  const double dlnx = log(x[1]/x0);

  // Only calculate the m>=0 part
  double eta_m[halfN+1];
  for(long i=0; i<=halfN; i++) 
  {
    eta_m[i] = 2*M_PI / dlnx / N * i;
  }

  // biased input func
  double* fb = malloc(N* sizeof(double));
  for(long i=0; i<N_pad; i++) 
  {
    fb[i] = 0.;
    fb[N-1-i] = 0.;
  }

  if(N_extrap_low) 
  {
    int sign;
    if(fx[0] == 0) 
    {
      log_fatal("J_abl_ar: Can't log-extrapolate zero on the low side!");
      exit(1);
    }
    else if (fx[0]>0) 
    {
      sign = 1;
    }
    else 
    {
      sign=-1;
    }
    
    if(fx[1]/fx[0] <= 0) 
    {
      log_fatal("J_abl_ar: Log-extrapolation on the low side fails due to sign change!");
      exit(1);
    }
    
    double dlnf_low = log(fx[1]/fx[0]);
    
    #pragma omp parallel for
    for(long i=N_pad; i<N_pad+N_extrap_low; i++) 
    {
      const double xi = exp(log(x0) + (i-N_pad - N_extrap_low)*dlnx);
      fb[i] = sign*exp(log(fx[0]*sign) + (i- N_pad - N_extrap_low)*dlnf_low) / pow(xi, config->nu);
    }
  }
  
  #pragma omp parallel for
  for(long i=N_pad+N_extrap_low; i<N_pad+N_extrap_low+N_original; i++) 
  {
    fb[i] = fx[i-N_pad-N_extrap_low] / pow(x[i-N_pad-N_extrap_low], config->nu) ;
  }
  
  if(N_extrap_high) 
  {
    int sign;
    
    if(fx[N_original-1] == 0) 
    {
      log_fatal("J_abl_ar: Can't log-extrapolate zero on the high side!");
      exit(1);
    }
    else if(fx[N_original-1] > 0) 
    {
      sign = 1;
    }
    else 
    {
      sign=-1;
    }
    
    if(fx[N_original-1]/fx[N_original-2] <= 0) 
    {
      log_fatal("J_abl_ar: Log-extrapolation on the high side fails due to sign change!");
      exit(1);
    }
    
    const double dlnf_high = log(fx[N_original-1]/fx[N_original-2]);
    
    #pragma omp parallel for
    for(long i=N-N_pad-N_extrap_high; i<N-N_pad; i++) 
    {
      const double xi = exp(log(x[N_original-1]) + (i-N_pad - N_extrap_low- N_original)*dlnx);
      fb[i] = sign * exp(log(fx[N_original-1]*sign) + 
        (i- N_pad - N_extrap_low- N_original)*dlnf_high) / pow(xi, config->nu);
    }
  }

  fftw_complex *out = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * (halfN+1) );
  fftw_plan plan_forward;
  plan_forward = fftw_plan_dft_r2c_1d(N, fb, out, FFTW_ESTIMATE);
  
  fftw_execute(plan_forward);
  
  c_window(out, config->c_window_width, halfN);

  double **out_ifft = malloc(sizeof(double*) * Nterms);
  fftw_complex **out_vary = malloc(sizeof(fftw_complex*) * Nterms);
  fftw_plan* plan_backward = malloc(sizeof(fftw_plan) * Nterms);

  for(int i_term=0;i_term<Nterms;i_term++) 
  {
    out_ifft[i_term] = malloc(sizeof(double) * (2*N) );
    out_vary[i_term] = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * (N+1) );
    plan_backward[i_term] = 
      fftw_plan_dft_c2r_1d(2*N, out_vary[i_term], out_ifft[i_term], FFTW_ESTIMATE);
  }

  double tau_l[N+1];
  for(long i=0;i<=N;i++)
  {
    tau_l[i] = 2.*M_PI / dlnx / N * i;
  }

  // initialize FFT plans for Convolution
  fftw_complex **a = malloc(sizeof(fftw_complex*) * Nterms);
  fftw_complex **b = malloc(sizeof(fftw_complex*) * Nterms);
  fftw_complex **a1 = malloc(sizeof(fftw_complex*) * Nterms);
  fftw_complex **b1 = malloc(sizeof(fftw_complex*) * Nterms);
  fftw_complex **c = malloc(sizeof(fftw_complex*) * Nterms);
  fftw_plan* pa = malloc(sizeof(fftw_plan) * Nterms);
  fftw_plan* pb = malloc(sizeof(fftw_plan) * Nterms);
  fftw_plan* pc = malloc(sizeof(fftw_plan) * Nterms);

  long Ntotal_convolve;
  if(N%2==0) 
  { // N+1 is odd
    Ntotal_convolve = 2*N + 1;
  }
  else 
  {
    log_fatal("J_abl_ar: This fftconvolve doesn't support even size input arrays"
      " (of out_pad1, outpad2)");
    exit(1);
  }

  for(int i_term=0; i_term<Nterms; i_term++) 
  {
    a[i_term] = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * Ntotal_convolve );
    b[i_term] = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * Ntotal_convolve );
    a1[i_term] = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * Ntotal_convolve );
    b1[i_term] = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * Ntotal_convolve );
    c[i_term] = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * Ntotal_convolve );

    pa[i_term] = 
      fftw_plan_dft_1d(Ntotal_convolve, a[i_term], a1[i_term], FFTW_FORWARD, FFTW_ESTIMATE);
    pb[i_term] = 
      fftw_plan_dft_1d(Ntotal_convolve, b[i_term], b1[i_term], FFTW_FORWARD, FFTW_ESTIMATE);
    pc[i_term] = 
      fftw_plan_dft_1d(Ntotal_convolve, a1[i_term], c[i_term], FFTW_BACKWARD, FFTW_ESTIMATE);
  }

  fftw_complex** out_pad1 = (fftw_complex**) fftw_malloc(sizeof(fftw_complex*) * Nterms);
  fftw_complex** out_pad2 = (fftw_complex**) fftw_malloc(sizeof(fftw_complex*) * Nterms);
  fftw_complex** pads_convolve = (fftw_complex**) fftw_malloc(sizeof(fftw_complex*) * Nterms);

  for(int i_term=0; i_term<Nterms; i_term++) 
  {
    out_pad1[i_term] = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * (N + 1));
    out_pad2[i_term] = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * (N + 1));
    pads_convolve[i_term] =  (fftw_complex*) fftw_malloc(sizeof(fftw_complex)*(2*N + 1));
  }

  #pragma omp parallel for
  for(int i_term=0; i_term<Nterms; i_term++) 
  {
    double complex gl[halfN+1];
    g_m_vals(ell[i_term]+0.5, 1.5 + config->nu + alpha[i_term], eta_m, gl, halfN + 1);

    // Do convolutions
    for(long i=0; i<=halfN; i++) 
    {
      out_pad1[i_term][i+halfN] = out[i] / (double) N * gl[i] ;
    }
    for(long i=0; i<halfN; i++) 
    {
      out_pad1[i_term][i] = conj(out_pad1[i_term][N-i]) ;
    }

    if(alpha[i_term] != beta[i_term])
    {
      g_m_vals(ell[i_term]+0.5, 1.5 + config->nu + beta[i_term], eta_m, gl, halfN+1);

      for(long i=0; i<=halfN; i++) 
      {
        out_pad2[i_term][i+halfN] = out[i] / (double)N * gl[i] ;
      }
      for(long i=0; i<halfN; i++) 
      {
        out_pad2[i_term][i] = conj(out_pad2[i_term][N-i]);
      }
      
      fftconvolve_optimize(out_pad1[i_term], out_pad2[i_term], N+1, pads_convolve[i_term], a[i_term], 
        b[i_term], a1[i_term], b1[i_term], c[i_term], pa[i_term], pb[i_term], pc[i_term]);
    }
    else
    {
      fftconvolve_optimize(out_pad1[i_term], out_pad1[i_term], N+1, pads_convolve[i_term], a[i_term], 
        b[i_term], a1[i_term], b1[i_term], c[i_term], pa[i_term], pb[i_term], pc[i_term]);
    }

    // convolution finished
    pads_convolve[i_term][N] = creal(pads_convolve[i_term][N]);
    double complex h_part[N+1];
    for(long i=0;i<=N;i++)
    {
      h_part[i] = pads_convolve[i_term][i+N]; // C_h term in Eq.(2.21) in McEwen et al (2016)
                                              // but only take h = 0,1,2,...,N.
    }

    const int p = -5.-2.*config->nu - alpha[i_term]-beta[i_term];
    double complex fz[N+1];
    f_z(p+1, tau_l, fz, N+1);

    for(long i=0; i<=N; i++)
    {
      out_vary[i_term][i] = h_part[i] * conj(fz[i]) * cpow(2., I*tau_l[i]);
    }
    
    fftw_execute(plan_backward[i_term]);

    const int sign_ell = (ell[i_term]%2? -1:1);
    for(long i=0; i<N_original; i++)
    {
      Fy[i_term][i] = out_ifft[i_term][2*(i+N_pad+N_extrap_low)] * sign_ell / (M_PI*M_PI) * 
        pow(2., 2.+2*config->nu+alpha[i_term]+beta[i_term]) * pow(x[i],-p-2.);
    }
  }

  for(int i_term=0; i_term<Nterms; i_term++)
  {
    fftw_destroy_plan(plan_backward[i_term]);
    fftw_free(out_vary[i_term]);
    fftw_free(out_pad1[i_term]);
    fftw_free(out_pad2[i_term]);
    fftw_free(pads_convolve[i_term]);

    free(out_ifft[i_term]);

    fftw_free(a[i_term]);
    fftw_free(b[i_term]);
    fftw_free(a1[i_term]);
    fftw_free(b1[i_term]);
    fftw_free(c[i_term]);

    fftw_destroy_plan(pa[i_term]);
    fftw_destroy_plan(pb[i_term]);
    fftw_destroy_plan(pc[i_term]);
  }

  free(plan_forward);
  free(plan_backward);
  free(out_ifft);
  fftw_free(out);
  free(fb);
  free(out_vary);

  free(out_pad1);
  free(out_pad2);
  free(pads_convolve);
  free(a);
  free(b);
  free(a1);
  free(b1);
  free(c);
  free(pa);
  free(pb);
  free(pc);
}

void Pd1d2(double *k, double *Pin, long Nk, double *Pout)
{
  int alpha_ar[] = {0,0,1};
  int beta_ar[]  = {0,0,-1};
  int ell_ar[]   = {0,2,1};
  int isP13type_ar[] = {0,0,0};
  double coeff_A_ar[] = {2.*(17./21), 2.*(4./21), 2.};
  int Nterms = 3;

  fastpt_scalar(alpha_ar, beta_ar, ell_ar, isP13type_ar, coeff_A_ar, Nterms, Pout, k, Pin, Nk);
}

void Pd2d2(double *k, double *Pin, long Nk, double *Pout)
{
  int alpha_ar[] = {0};
  int beta_ar[]  = {0};
  int ell_ar[]   = {0};
  int isP13type_ar[] = {0};
  double coeff_A_ar[] = {2.};
  int Nterms = 1;

  fastpt_scalar(alpha_ar, beta_ar, ell_ar, isP13type_ar, coeff_A_ar, Nterms, Pout, k, Pin, Nk);
}

void Pd1s2(double *k, double *Pin, long Nk, double *Pout)
{
  int alpha_ar[] = {0,0,0,1,1};
  int beta_ar[]  = {0,0,0,-1,-1};
  int ell_ar[]   = {0,2,4,1,3};
  int isP13type_ar[] = {0,0,0,0,0};
  double coeff_A_ar[] = {2*(8./315.),2*(254./441.),2*(16./245.),2*(4./15.),2*(2./5.)};
  int Nterms = 5;

  fastpt_scalar(alpha_ar, beta_ar, ell_ar, isP13type_ar, coeff_A_ar, Nterms, Pout, k, Pin, Nk);
}

void Pd2s2(double *k, double *Pin, long Nk, double *Pout)
{
  int alpha_ar[] = {0};
  int beta_ar[]  = {0};
  int ell_ar[]   = {2};
  int isP13type_ar[] = {0};
  double coeff_A_ar[] = {2.*2./3.};
  int Nterms = 1;

  fastpt_scalar(alpha_ar, beta_ar, ell_ar, isP13type_ar, coeff_A_ar, Nterms, Pout, k, Pin, Nk);
}

void Ps2s2(double *k, double *Pin, long Nk, double *Pout)
{
  int alpha_ar[] = {0,0,0};
  int beta_ar[]  = {0,0,0};
  int ell_ar[]   = {0,2,4};
  int isP13type_ar[] = {0,0,0};
  double coeff_A_ar[] = {2.*(4./45.), 2*(8./63.), 2*(8./35.)};
  int Nterms = 3;

  fastpt_scalar(alpha_ar, beta_ar, ell_ar, isP13type_ar, coeff_A_ar, Nterms, Pout, k, Pin, Nk);
}

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// -- NEW IMPLEMENTATION -------------------------------------------------------
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// next_fft_size: Round up n to the next "FFT-friendly" number whose only
// prime factors are 2, 3, 5, or 7.
//
// The FFT algorithm works by recursively splitting a size-N transform into
// smaller sub-transforms based on N's prime factorization. 
// FFTW has highly optimized, SIMD-vectorized "codelets" for small prime factors 
// (2, 3, 5, 7), making these splits very fast.
//
// When N has a large prime factor p, FFTW cannot split it efficiently and
// must fall back to generic algorithms, which are slower and cannot be vectorized. 

// For example:
//   N = 10240 = 2^11 × 5  → 11 radix-2 stages + 1 radix-5 stage, all fast
//   N = 10201 = 101 × 101 → two levels of prime-101 sub-transforms, slow
//
// Padding to a slightly larger FFT-friendly size does not affect the convolution
// result: the extra elements are zeros, and we read the same output indices 
// regardless of the padded size.
// ---------------------------------------------------------------------------
static long next_fft_size(long n) {
  while (1) {
    long m = n;
    while (m % 2 == 0) m /= 2;
    while (m % 3 == 0) m /= 3;
    while (m % 5 == 0) m /= 5;
    while (m % 7 == 0) m /= 7;
    if (m == 1) return n;
    n++;
  }
  return n;
}

// ---------------------------------------------------------------------------
// J_abJ1J2Jk: Core FAST-PT computation engine
// ---------------------------------------------------------------------------
// Name breakdown
//   J        The function computes J-type integrals involving angular
//              momentum coupling via Wigner 3-j and 6-j symbols
//   a, b     The two biasing exponents alpha and beta, which control the 
//              power-law debiasing: fb1 = f / x^(-2-alpha), fb2 = f / x^(-2-beta)
//   J1,J2,Jk The angular momentum quantum numbers in the recoupling scheme:
//              J1 couples (l, l2), J2 couples (l1, l), Jk couples (l1, l2).
//              These index the Gamma-ratio kernels g_m(J) that encode the
//              spherical Bessel function content
//
// Convolution integrals using the FFT-based algorithm. Each term is specified by
// (alpha, beta, J1, J2, Jk). Result for term i is stored in Fy[i][0..N_original-1].
//
// This function is called once per cosmology evaluation, always with the 
// same k grid and config but different fx (power spectrum). 
// The implementation exploits this by caching everything that doesn't depend on fx.
//
// ALGORITHM PER TERM
//   1. Multiply precomputed windowed FFT(fb) by g_m(J) -> a, b
//   2. FFT(a) -> a1,  FFT(b) -> b1           (2 FFTs)
//   3. Pointwise multiply: a1 *= b1
//   4. IFFT(a1) -> cv                         (1 FFT)
//   5. cv * conj(g_m(Jk)) / Ntotal -> out_vary
//   6. IFFT(out_vary) -> out_ifft             (1 FFT)
//   7. Extract and normalize: Fy = out_ifft * pi^(3/2)/8 / x
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------

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
  ) {
 
  typedef fftw_complex fftwc;
  // ---------------------------------------------------------------------------
  // STATIC CACHE
  // These variables persist across calls. Everything is rebuilt only when
  // N changes (which in practice means only on the very first call).
  // ---------------------------------------------------------------------------
  #define GM_J_MAX 16
 
  // Cached padded size and derived constants
  static long s_N = 0;
  static long s_Ntotal;       // next_fft_size(2*N + 1) (convolution output size)
  static long s_c2r_size;     // 2N
  static long s_Ncut;         // number of c_window tapering points
  static long s_eta_len;      // halfN + 1 (positive-frequency count)
  static long s_tau_len;      // N + 1
  // Frequency grids (depend only on N and dlnx)
  static double *s_eta_m = NULL; // eta_m[i] = 2*pi*i / (dlnx*N), i=0..halfN
  static double *s_tau_l = NULL; // tau_l[i] = 2*pi*i / (dlnx*N), i=0..N
 
  // Precomputed c_window weights: tapering function applied to the
  // high-frequency end of the FFT output to suppress ringing.
  // Precomputed to avoid recomputing sin() on every call.
  static double *s_c_win = NULL;
 
  // Cached padded k grid: x_full is deterministic from k_min/k_max/N,
  // so we compute it once and reuse. Only f_unbias changes per call.
  static double *s_x_full = NULL;
 
  // g_m_vals cache: indexed directly by J value (0..GM_J_MAX-1).
  // g_m_vals(J+0.5, -0.5, eta_m) involves expensive lngamma evaluations
  // over arrays of size halfN+1 or N+1. Since J values are small integers
  // that repeat across terms, we compute each unique J once and cache
  // forever (until N changes).
  static double complex *s_gm_eta[GM_J_MAX]; // g_m for eta frequencies
  static double complex *s_gm_tau[GM_J_MAX]; // g_m for tau frequencies
  static int s_gm_valid[GM_J_MAX];           // 1 if cached, 0 if not
 
  // Contiguous memory blocks (allocated for exact Nterms on first call):
  //   s_dbl_block:  out_ifft arrays, 2*N doubles per term (c2r FFT output)
  //   s_fft_block2: out_vary arrays, (N+1) complex per term (c2r FFT input)
  //   s_fft_block3: a/b/a1/b1/cv, 5*Ntotal complex per term
  static double *s_dbl_block = NULL;
  static fftwc *s_fft_block2 = NULL, *s_fft_block3 = NULL;
 
  // Template FFTW plans: 5 templates instead of hundreds of per-term plans.
  // fftw_execute_dft / fftw_execute_dft_c2r applies them to different arrays.
  //   s_plan_pa:   FFT forward,  size Ntotal (a -> a1)
  //   s_plan_pb:   FFT forward,  size Ntotal (b -> b1)
  //   s_plan_pc:   FFT backward, size Ntotal (a1 -> cv)
  //   s_plan_back: c2r backward, size 2*N    (out_vary -> out_ifft)
  //   s_plan_r2c:  r2c forward,  size N      (fb_tmp -> fft_tmp, alpha/beta)
  static fftw_plan s_plan_pa = NULL, s_plan_pb = NULL, s_plan_pc = NULL;
  static fftw_plan s_plan_back = NULL, s_plan_r2c = NULL;
 
  // ---------------------------------------------------------------------------
  // PARAMETER SETUP
  // ---------------------------------------------------------------------------
  const long N_original = N;
  const long N_pad = config->N_pad;
  const long N_extrap_low = config->N_extrap_low;
  const long N_extrap_high = config->N_extrap_high;
 
  // Pad N: zero-padding + extrapolation regions on both sides
  N += (2*N_pad + N_extrap_low + N_extrap_high);
 
  if (N % 2) {
    log_fatal("J_abJ1J2Jk_ar: Please use even number of x !");
    exit(1);
  }
  const long halfN = N / 2;
  const double x0 = x[0];
  const double dlnx = log(x[1] / x0); // log-spacing of input k array
 
  // Defensive check: if the k grid changed (different x0 or spacing) but
  // N stayed the same, the cached x_full/eta_m/tau_l are stale. Force a
  // full cache rebuild by resetting s_N, which triggers the one-time setup.
  {
    static double s_x0 = 0, s_dlnx = 0;
    if (s_N == N && (fabs(s_x0 - x0) > 1e-14 * fabs(x0) || 
        fabs(s_dlnx - dlnx) > 1e-14 * fabs(dlnx))) {
      s_N = 0; // force rebuild
    }
    s_x0 = x0;
    s_dlnx = dlnx;
  }

  // ---------------------------------------------------------------------------
  // ONE-TIME SETUP (first call, or if N changes)
  //
  // Allocates memory, creates plans, precomputes grids/weights/x_full.
  // Everything here is reused on all subsequent calls since only fx
  // (the power spectrum) changes between cosmology evaluations.
  // ---------------------------------------------------------------------------
  if (s_N != N) 
  {
    if (NULL != s_eta_m) { 
      free(s_eta_m); 
      free(s_tau_l); 
      free(s_c_win); 
      free(s_x_full); 
    }
    for (int j = 0; j < GM_J_MAX; j++) {
      if (s_gm_valid[j]) { 
        free(s_gm_eta[j]); 
        free(s_gm_tau[j]); 
      }
      s_gm_valid[j] = 0;
    }
    if (NULL != s_dbl_block) {
      free(s_dbl_block); 
      fftw_free(s_fft_block2); 
      fftw_free(s_fft_block3);
    }
    if (NULL != s_plan_pa) {
      fftw_destroy_plan(s_plan_pa); 
      fftw_destroy_plan(s_plan_pb);
      fftw_destroy_plan(s_plan_pc); 
      fftw_destroy_plan(s_plan_back);
    }
    if (NULL != s_plan_r2c) {
      fftw_destroy_plan(s_plan_r2c);
    }
 
    // -------------------------------------------------------------------------
    // s_Ntotal = 2*N + 1 is the size of the complex-to-complex FFT used
    // in the convolution step. The convolution inputs a and b each have
    // N+1 complex elements (indices 0..N, representing the positive and
    // negative frequency modes plus the zero-padding). The linear (non-
    // circular) convolution of two arrays of length N+1 produces an output
    // of length (N+1) + (N+1) - 1 = 2*N + 1. This is the standard
    // zero-padded FFT convolution size that avoids wrap-around aliasing.
    // -------------------------------------------------------------------------
    s_N = N;
    
    // Pad c2c convolution size to the next FFT-friendly number (only small
    // prime factors). E.g., 10201 = 101^2 → 10240 = 2^11 × 5, giving 3-5x
    // faster FFTs. Safe because the extra entries are time-domain zeros that
    // don't affect the linear convolution result.
    s_Ntotal = next_fft_size(2*N + 1);
    
    // c2r size is NOT padded — zero-padding in the frequency domain would
    // add spurious high-frequency components and change the reconstruction.
    // 2*N = 10200 = 2^3 × 3 × 5^2 × 17, which FFTW handles reasonably well.
    s_c2r_size = 2*N; 
    // -------------------------------------------------------------------------
 
    // -------------------------------------------------------------------------
    // Frequency grids for the log-Fourier transform.
    //
    // FAST-PT works in log-space: the input k array is log-spaced with spacing
    // dlnx = ln(k[1]/k[0]). The FFT of a function on this log-grid produces
    // Fourier modes at discrete frequencies eta_m and tau_l.
    //
    // eta_m[m] = 2*pi*m / (dlnx * N) for m = 0, 1, ..., halfN
    //   The positive Fourier frequencies of the biased input arrays (fb1, fb2)
    //   They are used as the imaginary argument to g_m_vals when computing the
    //   Gamma-ratio kernels for J1/J2. Only m >= 0 modes are stored since the
    //   input is real-valued (negative freq. follow from conjugate symmetry).
    //
    // tau_l[l] = 2*pi*l / (dlnx * N) for l = 0, 1, ..., N
    //   The Fourier freq. of the convolution output. They are used as the imaginary
    //   argument to g_m_vals when computing the Gamma-ratio kernel for Jk (the 
    //   third angular momentum in the recoupling scheme). The output has N+1 modes
    //   b/c the convolution of 2 length-N sequences produces a longer result.
    // -------------------------------------------------------------------------
    s_eta_len = halfN + 1;
    s_tau_len = N + 1;
    s_eta_m = malloc(sizeof(double) * s_eta_len);
    s_tau_l = malloc(sizeof(double) * s_tau_len);
    for (long i = 0; i < s_eta_len; i++) {
      s_eta_m[i] = 2*M_PI / dlnx / N * i;
    }
    for (long i = 0; i < s_tau_len; i++) {
      s_tau_l[i] = 2.*M_PI / dlnx / N * i;
    }
    
    // -------------------------------------------------------------------------
    // c_window weights: smooth tapering applied to the high-frequency end
    // of the FFT output before convolution.
    //
    // Without tapering, the sharp truncation at the Nyquist frequency causes ringing 
    // artifacts in the convolution result. The c_window smoothly rolls off the
    // FFT coefficients near the Nyquist freq. to suppress this.
    //
    // The window function is: W(i) = i/Ncut - sin(2*pi*i/Ncut) / (2*pi)
    // which satisfies W(0) = 0 (fully suppressed at Nyquist) and W(Ncut) = 1
    // (no tapering below the cutoff). The sin term makes the transition smooth
    // (continuous first derivative at both ends).
    //
    // Ncut = halfN * c_window_width determines how many high-frequency modes are
    // tapered. A typical value c_window_width = 0.65 means the top 65% of freq.
    // are untouched, and the remaining 35% near Nyquist are smoothly rolled off
    //
    // The weights are applied in reverse order to the FFT output:
    //   out[halfN - i] *= c_win[i]
    // so c_win[0] multiplies the Nyquist frequency (strongest suppression)
    // and c_win[Ncut] multiplies the cutoff frequency (no suppression).
    // -------------------------------------------------------------------------
    s_Ncut = (long)(halfN * config->c_window_width);
    s_c_win = malloc(sizeof(double) * (s_Ncut + 1));
    for (long i = 0; i <= s_Ncut; i++) {
      s_c_win[i] = (double)i / s_Ncut - 1./(2.*M_PI) * sin(2.*i*M_PI / s_Ncut);
    }
 
    // -------------------------------------------------------------------------
    // Cached x_full (padded k grid)
    // The k grid is deterministic from k_min/k_max/N, so x_full never changes.
    // The per-call work only rebuilds f_unbias (which changing power spectrum fx).
    // -------------------------------------------------------------------------
    s_x_full = malloc(sizeof(double) * N);

    for (long i = 0; i < N_pad; i++) {
      s_x_full[i]     = exp(log(x0) + (i - N_pad - N_extrap_low) * dlnx);
      s_x_full[N-1-i] = exp(log(x0) + (N-1-i - N_pad - N_extrap_low) * dlnx);
    }

    if (N_extrap_low) {
      for (long i = N_pad; i < N_pad + N_extrap_low; i++)
        s_x_full[i] = exp(log(x0) + (i - N_pad - N_extrap_low) * dlnx);
    }

    for (long i = N_pad + N_extrap_low; i < N_pad + N_extrap_low + N_original; i++) {
      s_x_full[i] = x[i - N_pad - N_extrap_low];
    }

    if (N_extrap_high) {
      for (long i = N - N_pad - N_extrap_high; i < N - N_pad; i++)
        s_x_full[i] = exp(log(x[N_original-1])
                      + (i - N_pad - N_extrap_low - N_original) * dlnx);
    }
 
    // -------------------------------------------------------------------------
    // Template r2c plan for alpha/beta forward FFTs (Created with tmp arrays)
    // fftw_execute_dft_r2c applies it to the actual fb_tmp/fft_tmp arrays later
    // -------------------------------------------------------------------------
    {
      double *tmp_in = fftw_malloc(sizeof(double) * N);
      fftwc *tmp_out = fftw_malloc(sizeof(fftwc) * (halfN + 1));
      s_plan_r2c = fftw_plan_dft_r2c_1d(N, tmp_in, tmp_out, FFTW_ESTIMATE);
      fftw_free(tmp_in);
      fftw_free(tmp_out);
    }

    // -------------------------------------------------------------------------
    // Contiguous memory blocks for Nterms ---
    //   s_dbl_block:  Nterms * 2*N doubles    (out_ifft, c2r output)
    //   s_fft_block2: Nterms * (N+1) complex  (out_vary, c2r input)
    //   s_fft_block3: Nterms * 5*Ntotal complex (a, b, a1, b1, cv)
    // -------------------------------------------------------------------------
    s_dbl_block  = malloc(sizeof(double) * Nterms * s_c2r_size);
    s_fft_block2 = fftw_malloc(sizeof(fftwc) * Nterms * (s_c2r_size/2 + 1));
    s_fft_block3 = fftw_malloc(sizeof(fftwc) * Nterms * 5 * s_Ntotal);
 
    // -------------------------------------------------------------------------
    // 4 template plans using first term's arrays as prototypes
    // fftw_execute_dft applies the same algorithm to any term's arrays.
    // -------------------------------------------------------------------------
    fftwc *base0 = s_fft_block3;
    
    s_plan_pa = fftw_plan_dft_1d(s_Ntotal, base0,
                    base0 + 2*s_Ntotal, FFTW_FORWARD, FFTW_ESTIMATE);
    
    s_plan_pb = fftw_plan_dft_1d(s_Ntotal, base0 + s_Ntotal,
                    base0 + 3*s_Ntotal, FFTW_FORWARD, FFTW_ESTIMATE);
    
    s_plan_pc = fftw_plan_dft_1d(s_Ntotal, base0 + 2*s_Ntotal,
                    base0 + 4*s_Ntotal, FFTW_BACKWARD, FFTW_ESTIMATE);
    
    s_plan_back = fftw_plan_dft_c2r_1d(s_c2r_size, s_fft_block2, s_dbl_block, FFTW_ESTIMATE);
 
    // -------------------------------------------------------------------------
    // Pre-zero-pad a and b tails for all terms
    // The convolution requires zero-padding from index N+1 to Ntotal-1.
    // Done once here; the main loop only writes indices 0..N.
    // -------------------------------------------------------------------------
    for (int i = 0; i < Nterms; i++) {
      fftwc *base = s_fft_block3 + i * 5 * s_Ntotal;
      memset(base + (N+1), 0, sizeof(fftwc) * (s_Ntotal - (N+1)));          // a tail
      memset(base + s_Ntotal + (N+1), 0, sizeof(fftwc)*(s_Ntotal - (N+1))); // b tail
    }
  }
 
  // ---------------------------------------------------------------------------
  // CACHE g_m_vals FOR UNIQUE J VALUES
  //
  // g_m_vals computes Gamma-ratio arrays involving lngamma evaluations
  // over halfN+1 or N+1 points. J values are small integers (0-8 typically)
  // that repeat across terms. Each unique J is computed once and cached
  // forever (until N changes).
  // ---------------------------------------------------------------------------
  for (int i = 0; i < Nterms; i++) {
    int js[] = { J1[i], J2[i], Jk[i] };
    for (int k = 0; k < 3; k++) {
      int j = js[k];
      if (!s_gm_valid[j]) {
        s_gm_eta[j] = malloc(sizeof(double complex) * s_eta_len);
        s_gm_tau[j] = malloc(sizeof(double complex) * s_tau_len);
        g_m_vals(j + 0.5, -0.5, s_eta_m, s_gm_eta[j], s_eta_len);
        g_m_vals(j + 0.5, -0.5, s_tau_l, s_gm_tau[j], s_tau_len);
        s_gm_valid[j] = 1;
      }
    }
  }
 
  // ---------------------------------------------------------------------------
  // BUILD f_unbias (PER-CALL: changes each evaluation because fx changes)
  //
  // x_full is cached (deterministic from k grid), so only f_unbias needs
  // rebuilding. The padded array has:
  //   - Zero-padding at both ends (N_pad on each side)
  //   - Log-extrapolation regions beyond the original data
  //   - Original fx data in the middle
  // ---------------------------------------------------------------------------
  double f_unbias[N];
 
  for (long i = 0; i < N_pad; i++) { // Zero-padding at both ends
    f_unbias[i]     = 0.;
    f_unbias[N-1-i] = 0.;
  }
 
  if (N_extrap_low) { // Low-side log-extrapolation
    int sign = (fx[0] > 0) ? 1 : -1;
    
    const double dlnf_low = log(fx[1] / fx[0]);
    
    for (long i = N_pad; i < N_pad + N_extrap_low; i++) {
      f_unbias[i] = sign * exp(log(fx[0]*sign)
                    + (i - N_pad - N_extrap_low) * dlnf_low);
    }
  }
 
  // Copy original data into the middle
  for (long i = N_pad + N_extrap_low; i < N_pad + N_extrap_low + N_original; i++) {
    f_unbias[i] = fx[i - N_pad - N_extrap_low];
  }
  
  if (N_extrap_high) { // High-side log-extrapolation
    int sign = (fx[N_original-1] > 0) ? 1 : -1;
    
    const double dlnf_high = log(fx[N_original-1] / fx[N_original-2]);
    
    for (long i = N - N_pad - N_extrap_high; i < N - N_pad; i++) {
      f_unbias[i] = sign * exp(log(fx[N_original-1]*sign)
                    + (i - N_pad - N_extrap_low - N_original) * dlnf_high);
    }
  }
 
  // ---------------------------------------------------------------------------
  // PRECOMPUTE WINDOWED FFTs FOR UNIQUE ALPHA/BETA VALUES
  //
  // In the original code, each term i computed two biased arrays:
  //   fb1[i] = f_unbias / x^nu1   where nu1 = -2 - alpha[i]
  //   fb2[i] = f_unbias / x^nu2   where nu2 = -2 - beta[i]
  // then forward-FFT'd each one and applied c_window. This was done
  // inside the per-term loop, resulting in 2*Nterms forward FFTs.
  //
  // Key insight: fb1 depends only on alpha (not on J1, J2, Jk, or beta),
  // and fb2 depends only on beta (not on J1, J2, Jk, or alpha). The J
  // values only enter later when fb is multiplied by the g_m(J) Gamma
  // ratios. So if multiple terms share the same alpha, their fb1 arrays
  // are identical — we can compute the FFT once and reuse it.
  //
  // Procedure:
  //   1. Find unique alpha values → compute FFT(fb) + c_window once each
  //   2. Find unique beta values  → same
  //   3. Each term looks up its precomputed result by index
  // ---------------------------------------------------------------------------
  int unique_alpha[Nterms];
  int n_unique_alpha = 0;
  for (int i = 0; i < Nterms; i++) {
      int found = 0;
    
      for (int j = 0; j < n_unique_alpha; j++) {
        if (unique_alpha[j] == alpha[i]) { 
          found = 1; 
          break; 
        }
      }
      if (!found) {
        unique_alpha[n_unique_alpha++] = alpha[i];
      }
  }

  int unique_beta[Nterms];
  int n_unique_beta = 0;
  for (int i = 0; i < Nterms; i++) {
    int found = 0;
    
    for (int j = 0; j < n_unique_beta; j++) {
      if (unique_beta[j] == beta[i]) { 
        found = 1; 
        break; 
      }
    }
    if (!found) { 
      unique_beta[n_unique_beta++] = beta[i];
    }
  }
 
  // Allocate storage for precomputed windowed FFT results
  const long fft_small = halfN + 1;
  fftwc *fft_alpha = fftw_malloc(sizeof(fftwc)*n_unique_alpha*fft_small);
  fftwc *fft_beta  = fftw_malloc(sizeof(fftwc)*n_unique_beta*fft_small);
 
  // Temporary arrays for forward FFT (stack-allocated).
  // Uses cached s_plan_r2c template plan via fftw_execute_dft_r2c.
  double fb_tmp[N];
  fftwc fft_tmp[fft_small];
 
  // Compute windowed FFT for each unique alpha value:
  //   1. Bias: fb_tmp[i] = f_unbias[i] / x_full[i]^nu,  where nu = -2 - alpha
  //   2. Forward r2c FFT (using cached template plan, no plan creation)
  //   3. Apply c_window tapering to suppress high-frequency ringing
  //   4. Store result for lookup by all terms sharing this alpha
  for (int u = 0; u < n_unique_alpha; u++) {
    double nu = -2. - unique_alpha[u];
    
    for (long i = 0; i < N; i++) {
      // Bias: fb_tmp[i] = f_unbias[i] / x_full[i]^nu,  where nu = -2 - alpha
      fb_tmp[i] = f_unbias[i] / pow(s_x_full[i], nu);
    }
    
    fftw_execute_dft_r2c(s_plan_r2c, fb_tmp, fft_tmp);  // 2. Forward r2c FFT 
    
    // Apply c_window tapering to suppress high-frequency ringing
    for (long i = 0; i <= s_Ncut; i++) {
      fft_tmp[halfN-i] *= s_c_win[i];
    }
    
    // Copy the windowed FFT result from the temporary buffer into the
    // fft_alpha storage at the slot for unique alpha index u.
    // memcpy(dest, src, nbytes) copies fft_small complex values contiguously.
    memcpy(fft_alpha + u*fft_small, fft_tmp, sizeof(fftwc) * fft_small);
  }
  
  // Same for each unique beta value (fb2 depends only on beta, not alpha or J)
  for (int u = 0; u < n_unique_beta; u++) {
    double nu = -2. - unique_beta[u];
    for (long i = 0; i < N; i++) {
      fb_tmp[i] = f_unbias[i] / pow(s_x_full[i], nu);
    }
    
    fftw_execute_dft_r2c(s_plan_r2c, fb_tmp, fft_tmp);
    
    for (long i = 0; i <= s_Ncut; i++) {
      fft_tmp[halfN-i] *= s_c_win[i];
    }
    
    // Copy the windowed FFT result from the temporary buffer into the
    // fft_beta storage at the slot for unique beta index u.
    // memcpy(dest, src, nbytes) copies fft_small complex values contiguously.
    memcpy(fft_beta + u*fft_small, fft_tmp, sizeof(fftwc) * fft_small);
  }
 
  // Build lookup: for each term, which unique alpha/beta index to use
  int idx_alpha[Nterms];
  for (int i = 0; i < Nterms; i++) {
    for (int u = 0; u < n_unique_alpha; u++) {
      if (unique_alpha[u] == alpha[i]) idx_alpha[i] = u;
    }
  }
  int idx_beta[Nterms];
  for (int i = 0; i < Nterms; i++) {
    for (int u = 0; u < n_unique_beta; u++) {
      if (unique_beta[u] == beta[i]) idx_beta[i] = u;
    }
  }

  // =========================================================================
  // DERIVE POINTER ARRAYS from cached contiguous blocks.
  //
  // The static blocks (s_dbl_block, s_fft_block2, s_fft_block3) are large
  // contiguous allocations that hold all per-term working arrays. Instead
  // of malloc'ing each array individually, we carve them out of the 
  // pre-allocated blocks using pointer arithmetic.
  //
  // The pointer arrays themselves (a[], b[], out_ifft[], etc.) are cheap
  // stack-allocated VLAs: just Nterms pointers each, not data. They are
  // recomputed on every call since Nterms may vary, but the underlying
  // memory they point into (the static blocks) persists across calls.
  //
  // s_dbl_block layout (doubles, stride = 2*N per term):
  //   out_ifft[i] = s_dbl_block + i * 2*N
  //   (real-valued output of the final c2r IFFT, length 2*N)
  //
  // s_fft_block2 layout (fftw_complex, stride = N+1 per term):
  //   out_vary[i] = s_fft_block2 + i * (N+1)
  //   (complex input to the final c2r IFFT, length N+1)
  //
  // s_fft_block3 layout (fftw_complex, stride = 5*Ntotal per term):
  //   a[i]  = base + 0*Ntotal   convolution input 1 (from alpha/gl1)
  //   b[i]  = base + 1*Ntotal   convolution input 2 (from beta/gl2)
  //   a1[i] = base + 2*Ntotal   FFT(a) output, then overwritten by a1 *= b1
  //   b1[i] = base + 3*Ntotal   FFT(b) output
  //   cv[i] = base + 4*Ntotal   IFFT(a1*b1) convolution result
  // =========================================================================
  double *out_ifft[Nterms];
  fftwc *out_vary[Nterms];
  fftwc *a[Nterms], *b[Nterms], *a1[Nterms], *b1[Nterms], *cv[Nterms];
  for (int i = 0; i < Nterms; i++) {
    out_ifft[i] = s_dbl_block + i * s_c2r_size;
    out_vary[i] = s_fft_block2 + i * (s_c2r_size/2 + 1);
    fftwc *base = s_fft_block3 + i * 5 * s_Ntotal;
    a[i]  = base;
    b[i]  = base + s_Ntotal;
    a1[i] = base + 2*s_Ntotal;
    b1[i] = base + 3*s_Ntotal;
    cv[i] = base + 4*s_Ntotal;
  }
 
  const double pi_factor = pow(M_PI, 1.5) / 8.;

  #pragma omp parallel for
  for (int i_term = 0; i_term < Nterms; i_term++) {  // MAIN COMPUTATION LOOP
    // --- Step 1: Build convolution inputs a and b ---
    //
    // Each term's convolution inputs are constructed by combining two pieces:
    //   a[m] = FFT(fb1)[m] / N * g_m(J1)[m]
    //   b[m] = FFT(fb2)[m] / N * g_m(J2)[m]
    //
    // where:
    //   FFT(fb1) = precomputed windowed FFT of the alpha-biased input
    //              (looked up from fft_alpha by this term's alpha index)
    //   FFT(fb2) = precomputed windowed FFT of the beta-biased input
    //              (looked up from fft_beta by this term's beta index)
    //   g_m(J1)  = cached Gamma-ratio kernel for angular momentum J1
    //   g_m(J2)  = cached Gamma-ratio kernel for angular momentum J2
    //   1/N      = normalization from the forward FFT
    //
    // The precomputed FFT results (out_a, out_b) contain only the
    // non-negative frequencies (indices 0..halfN), as output by r2c.
    // We write these into indices halfN..N of the a/b arrays, then
    // fill indices 0..halfN-1 using conjugate symmetry:
    //   a[i] = conj(a[N-i])
    // This reconstructs the full complex spectrum needed by the
    // complex-to-complex FFT in the convolution step.
    //
    // Indices N+1..Ntotal-1 are zero-padding for the linear (non-circular)
    // convolution. These were pre-zeroed at allocation time and are not
    // touched here.

    // Look up precomputed windowed FFT for this term's alpha and beta
    const fftwc *out_a = fft_alpha + idx_alpha[i_term] * fft_small;
    const fftwc *out_b = fft_beta  + idx_beta[i_term]  * fft_small;
    
    // Look up cached Gamma-ratio kernels for J1 and J2
    const double complex *gl1 = s_gm_eta[J1[i_term]];
    const double complex *gl2 = s_gm_eta[J2[i_term]];
 
    // Positive frequencies (indices halfN..N)
    for (long i = 0; i <= halfN; i++) {
      a[i_term][i+halfN] = out_a[i] * (1.0 / (double)N) * gl1[i];
      b[i_term][i+halfN] = out_b[i] * (1.0 / (double)N) * gl2[i];
    }

    // Negative frequencies (indices 0..halfN-1): conjugate symmetry
    // For real-valued inputs, FFT[-m] = conj(FFT[m]), so a[i] = conj(a[N-i])
    for (long i = 0; i < halfN; i++) {
      a[i_term][i] = conj(a[i_term][N-i]);
      b[i_term][i] = conj(b[i_term][N-i]);
    }

    // Indices N+1..Ntotal-1 were pre-zero-padded at allocation time.
 
    // ------------------------------------------------------------------------
    // Steps 2-4: Convolution via FFT
    //
    // This computes the convolution of the two biased, windowed inputs:
    //   conv[h] = sum_m  a[m] * b[h-m]
    // which in Fourier space becomes a pointwise product:
    //   FFT(conv) = FFT(a) * FFT(b)
    //
    // a contains the windowed FFT of fb1 (biased by alpha) multiplied
    // by g_m(J1) — the first leg of the FAST-PT two-point integral.
    // b contains the windowed FFT of fb2 (biased by beta) multiplied
    // by g_m(J2) — the second leg.
    //
    // The convolution result cv is normalized here by 1/Ntotal 
    // FFTW's backward transform is unnormalized. 
    // This makes cv the true convolution values, keeping Step 5 cleaner.
    // ------------------------------------------------------------------------
    fftw_execute_dft(s_plan_pa, a[i_term], a1[i_term]);  // FFT(a) -> a1
    fftw_execute_dft(s_plan_pb, b[i_term], b1[i_term]);  // FFT(b) -> b1
    for (long i = 0; i < s_Ntotal; i++) {
      a1[i_term][i] *= b1[i_term][i];                    // a1 *= b1
    }
    fftw_execute_dft(s_plan_pc, a1[i_term], cv[i_term]); // IFFT(a1) -> cv
    for (long i = 0; i < s_Ntotal; i++) {
      cv[i_term][i] *= (1.0 / (double) s_Ntotal);        // normalize IFFT
    }

    // ------------------------------------------------------------------------
    // Step 5: Multiply by conj(g_m(Jk)) and zero-pad for c2r FFT
    //
    // After the convolution (Steps 2-4), cv holds the normalized result
    // C_h for h = 0, 1, ..., 2*N (normalization by 1/Ntotal was already
    // applied at the end of Step 4). We need only the h = 0..N portion,
    // stored at cv[N..2N] due to FFT index ordering.
    //
    // The FAST-PT algorithm (McEwen et al 2016) requires multiplying
    // C_h by the conjugate of the third Gamma-ratio kernel g_m(Jk),
    // which accounts for the Jk angular coupling in the three-point
    // recoupling scheme (J1, J2, Jk).
    //
    // cv[N] is forced real because it's the Hermitian midpoint of the
    // complex-valued convolution — any imaginary part is numerical noise.
    //
    // The result out_vary is the input for the final c2r IFFT (Step 6).
    // Its length is s_c2r_size/2 + 1, which may be larger than N+1 due
    // to FFT-friendly padding (e.g., 2*N=10200 padded to s_c2r_size=10240).
    // The physical data fills indices 0..N; the remaining indices
    // N+1..s_c2r_size/2 are zero-padded to match the padded transform size.
    // This padding does not affect the result — we read the same output
    // indices in Step 7 regardless of the padded size.
    // ------------------------------------------------------------------------
    cv[i_term][N] = creal(cv[i_term][N]); // Hermitian midpoint must be real
    const double complex *fz = s_gm_tau[Jk[i_term]];
    for (long i = 0; i <= N; i++) {
      out_vary[i_term][i] = cv[i_term][i+N] * conj(fz[i]); // C_h * conj(g_m(Jk))
    }
    // Zero-pad for FFT-friendly c2r transform size
    for (long i = N+1; i <= s_c2r_size/2; i++) {
      out_vary[i_term][i] = 0.0;
    } 

    // ------------------------------------------------------------------------
    // Step 6: Final backward c2r FFT
    //
    // Transform out_vary (complex, length N+1) back to real space
    // (length 2*N). This is the last FFT in the FAST-PT pipeline —
    // the result is the real-space contribution of this term to the
    // output power spectrum, still in the padded/extrapolated grid.
    // ------------------------------------------------------------------------
    fftw_execute_dft_c2r(s_plan_back, out_vary[i_term], out_ifft[i_term]);
 
    // ------------------------------------------------------------------------
    // Step 7: Extract result and normalize
    //
    // The c2r output out_ifft has length 2*N, but only every other element
    // corresponds to a point on the original k grid (the interleaving
    // comes from the zero-padded convolution doubling the array size).
    //
    // We extract elements at indices 2*(i + N_pad + N_extrap_low), which
    // skips the zero-padding and extrapolation regions at both ends,
    // recovering only the N_original points that correspond to the
    // original input k values.
    //
    // The normalization factor pi^(3/2) / 8 comes from the angular integration
    // in the FAST-PT formalism. Division by x[i] = k[i] converts from
    // the log-spaced convolution variable back to the physical power spectrum.
    // ------------------------------------------------------------------------
    for (long i = 0; i < N_original; i++) {
      Fy[i_term][i] = out_ifft[i_term][2*(i+N_pad+N_extrap_low)]*pi_factor/x[i];
    }
  }
 
  // CLEANUP: only the per-call alpha/beta FFT storage.
  // Everything else persists in static variables for the next call.
  fftw_free(fft_alpha);
  fftw_free(fft_beta);
}
