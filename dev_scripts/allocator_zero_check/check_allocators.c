// Check empty allocations, logical row access and positive-size layouts.
// Link this file with the actual basics.c, structs.c and log.c objects.
// No allocation implementation is copied into this test.
#include "cosmolike/basics.h"
#include <assert.h>
#include <complex.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void check_owner(
  const void* owner // one block returned by a core allocator
)
{
  assert(NULL != owner);
  assert(0 == (uintptr_t)owner % 64);
}

// Character access is valid for every object representation. Row patterns
// detect overlaps between logical rows without reading padding or treating
// a multidimensional allocation as one flat data block.
static void write_row(
  void* row, // start of a logical data row
  const size_t bytes, // logical bytes, excluding alignment padding
  const int pattern // distinct byte value for this row
)
{
  memset(row, pattern, bytes);
}

static void check_row(
  const void* row, // row whose pattern was written before other rows
  const size_t bytes, // logical bytes, excluding padding
  const int pattern // expected byte value
)
{
  const unsigned char* values = row;
  for (size_t i=0; i<bytes; i++) {
    assert((unsigned char)pattern == values[i]);
  }
}

// Report offsets, never absolute addresses. Comparing this output between
// builds checks allocation layout independently of process memory placement.
static void report_row(
  const char* name, // allocator and dimension label
  const void* owner, // start of the allocation
  const void* row // an existing row or an empty row's end pointer
)
{
  printf("%s %td\n", name,
    (const unsigned char*)row - (const unsigned char*)owner);
}

static void check_one_dimension(
  const int n // zero or a positive logical count
)
{
  int* integers = malloc1d_int(n);
  double* values = malloc1d(n);
  double* zeros = calloc1d(n);
  check_owner(integers);
  check_owner(values);
  check_owner(zeros);
  write_row(integers, sizeof(int)*(size_t)n, 17);
  write_row(values, sizeof(double)*(size_t)n, 31);
  check_row(integers, sizeof(int)*(size_t)n, 17);
  check_row(values, sizeof(double)*(size_t)n, 31);
  check_row(zeros, sizeof(double)*(size_t)n, 0);
  printf("one_dimension %d\n", n);
  free(integers);
  free(values);
  free(zeros);
}

static void check_two_dimensions(
  const int nx, // row count, including zero
  const int ny // logical entries per row, including zero
)
{
  double** values = (double**)malloc2d(nx, ny);
  int** integers = (int**)malloc2d_int(nx, ny);
  fftw_complex** fourier = (fftw_complex**)malloc2d_fftwc(nx, ny);
  fftw_plan** plans = (fftw_plan**)malloc2d_fftwp(nx, ny);
  double*** pointers = (double***)malloc2d_ptr(nx, ny);
  check_owner(values);
  check_owner(integers);
  check_owner(fourier);
  check_owner(plans);
  check_owner(pointers);
  printf("two_dimensions %d %d\n", nx, ny);
  for (int i=0; i<nx; i++) {
    assert(0 == (uintptr_t)values[i] % 64);
    assert(0 == (uintptr_t)integers[i] % 64);
    assert(0 == (uintptr_t)fourier[i] % 64);
    write_row(values[i], sizeof(double)*(size_t)ny, 1+i);
    write_row(integers[i], sizeof(int)*(size_t)ny, 1+i);
    write_row(fourier[i], sizeof(fftw_complex)*(size_t)ny, 1+i);
    for (int j=0; j<ny; j++) {
      plans[i][j] = NULL;
      pointers[i][j] = &values[i][j];
    }
    report_row("double", values, values[i]);
    report_row("int", integers, integers[i]);
    report_row("fftw_complex", fourier, fourier[i]);
    report_row("fftw_plan", plans, plans[i]);
    report_row("double_pointer", pointers, pointers[i]);
  }
  for (int i=0; i<nx; i++) {
    check_row(values[i], sizeof(double)*(size_t)ny, 1+i);
    check_row(integers[i], sizeof(int)*(size_t)ny, 1+i);
    check_row(fourier[i], sizeof(fftw_complex)*(size_t)ny, 1+i);
    for (int j=0; j<ny; j++) {
      assert(NULL == plans[i][j]);
      assert(&values[i][j] == pointers[i][j]);
    }
  }
  free(values);
  free(integers);
  free(fourier);
  free(plans);
  free(pointers);
}

static void check_three_dimensions(
  const int nx, // first extent, including zero
  const int ny, // second extent, including zero
  const int nz // final extent, including zero
)
{
  double*** values = (double***)malloc3d(nx, ny, nz);
  double complex*** complex_values =
    (double complex***)malloc3d_complex(nx, ny, nz);
  double**** pointers = (double****)malloc3d_ptr(nx, ny, nz);
  check_owner(values);
  check_owner(complex_values);
  check_owner(pointers);
  printf("three_dimensions %d %d %d\n", nx, ny, nz);
  for (int i=0; i<nx; i++) {
    report_row("double_level2", values, values[i]);
    report_row("complex_level2", complex_values, complex_values[i]);
    report_row("pointer_level2", pointers, pointers[i]);
    for (int j=0; j<ny; j++) {
      assert(0 == (uintptr_t)values[i][j] % 64);
      assert(0 == (uintptr_t)complex_values[i][j] % 64);
      const int pattern = 1+i*ny+j;
      write_row(values[i][j], sizeof(double)*(size_t)nz, pattern);
      write_row(complex_values[i][j],
        sizeof(double complex)*(size_t)nz, pattern);
      for (int k=0; k<nz; k++) {
        pointers[i][j][k] = &values[i][j][k];
      }
      report_row("double", values, values[i][j]);
      report_row("complex", complex_values, complex_values[i][j]);
      report_row("pointer", pointers, pointers[i][j]);
    }
  }
  for (int i=0; i<nx; i++) {
    for (int j=0; j<ny; j++) {
      const int pattern = 1+i*ny+j;
      check_row(values[i][j], sizeof(double)*(size_t)nz, pattern);
      check_row(complex_values[i][j],
        sizeof(double complex)*(size_t)nz, pattern);
      for (int k=0; k<nz; k++) {
        assert(&values[i][j][k] == pointers[i][j][k]);
      }
    }
  }
  free(values);
  free(complex_values);
  free(pointers);
}

static void check_four_dimensions(
  const int nx, // first extent, including zero
  const int ny, // second extent, including zero
  const int nz, // third extent, including zero
  const int nw // final extent, including zero
)
{
  double**** values = (double****)malloc4d(nx, ny, nz, nw);
  check_owner(values);
  printf("four_dimensions %d %d %d %d\n", nx, ny, nz, nw);
  for (int i=0; i<nx; i++) {
    report_row("level3", values, values[i]);
    for (int j=0; j<ny; j++) {
      report_row("level2", values, values[i][j]);
      for (int k=0; k<nz; k++) {
        assert(0 == (uintptr_t)values[i][j][k] % 64);
        const int pattern = 1+(i*ny+j)*nz+k;
        write_row(values[i][j][k], sizeof(double)*(size_t)nw, pattern);
        report_row("double", values, values[i][j][k]);
      }
    }
  }
  for (int i=0; i<nx; i++) {
    for (int j=0; j<ny; j++) {
      for (int k=0; k<nz; k++) {
        const int pattern = 1+(i*ny+j)*nz+k;
        check_row(values[i][j][k], sizeof(double)*(size_t)nw, pattern);
      }
    }
  }
  free(values);
}

int main(
  const int argc, // optional --positive excludes empty allocations
  char** argv // command-line arguments
)
{
  assert(1 == argc || 2 == argc);
  const int positive = 2 == argc;
  if (positive) assert(0 == strcmp(argv[1], "--positive"));
  const int dimensions[3] = {0, 1, 9};
  const int first = positive ? 1 : 0;
  for (int a=first; a<3; a++) {
    check_one_dimension(dimensions[a]);
    for (int b=first; b<3; b++) {
      check_two_dimensions(dimensions[a], dimensions[b]);
      for (int c=first; c<3; c++) {
        check_three_dimensions(dimensions[a], dimensions[b], dimensions[c]);
        // Keep the upper three dimensions small; the last extent of nine
        // still crosses a cache-line padding boundary for doubles.
        for (int d=first; d<3; d++) {
          check_four_dimensions(a, b, c, dimensions[d]);
        }
      }
    }
  }
  puts("allocator checks passed");
  return 0;
}
