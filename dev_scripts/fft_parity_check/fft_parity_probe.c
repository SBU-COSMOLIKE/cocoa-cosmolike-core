// fft_parity_probe.c: command-line probe of the FFT-length selection.
//
// check_fft_parity.py (same folder) compiles this file together with
// the repository's own cosmolike/basics.c, cosmolike/structs.c and
// log.c/src/log.c, so every run exercises the next_fft_size that is
// in the working tree right now, never an extracted copy. The probe
// only prints plain numbers; the python driver owns every pass/fail
// decision and checks the numbers against an independent enumeration
// of the admissible lengths (products of powers of 2, 3, 5 and 7).
//
// Modes (each selected by argv[1]; every length prints in base 10):
//   range <lo> <hi>  one line "n direct caller retries" for every
//                    request n from lo to hi: direct is
//                    next_fft_size(n), caller is the even-length
//                    selection below, retries counts its extra scans
//   cases <n> ...    the same line for each listed request
//   one <n>          a single next_fft_size(n) call, result printed.
//                    The driver runs this mode in a child process for
//                    the requests that must terminate the program
//                    through log_fatal: a request below 1, and a
//                    request above every representable length.
//
// Exit codes: 0 after printing every requested line, 2 on a usage or
// parse error. Mode "one" never returns for a refused request, since
// next_fft_size calls exit(1) through log_fatal.

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "cosmolike/basics.h"

// ---------------------------------------------------------------------------
// The even-length selection exactly as the call sites perform it:
// J_abl in cfastpt/cfastpt.c and the FFTLog spectra of
// cosmolike/cosmo2D.c and cosmolike/cosmo2D_cluster.c. The raw
// request is rounded up to even, handed to next_fft_size, and the
// scan is retried from the next integer until the length lands even.
// A single retry is not enough, because consecutive admissible
// lengths can share parity: the even request 242 meets 243 = 3^5 and
// then 245 = 5 * 7^2, both odd, before the even 250. The loop is the
// repair this probe guards; the retry count lets the driver confirm
// the loop really runs more than once where the enumeration says it
// must.
//
// Parameters:
//   request = raw transform-length request, at least 1
//   retries = out: how many extra next_fft_size scans the loop needed
// Returns:
//   the smallest even admissible length at or above the even-rounded
//   request (the python driver verifies exactly this property)
// ---------------------------------------------------------------------------
static long caller_even_selection(const long request, long* retries)
{
  long size = request;
  if (size % 2) size++;
  size = next_fft_size(size);
  *retries = 0;
  while (size % 2) {
    size = next_fft_size(size + 1);
    *retries += 1;
  }
  return size;
}

// ---------------------------------------------------------------------------
// Parse one base-10 long from text, refusing partial or out-of-range
// input, so a typo in the driver cannot silently probe the wrong
// request.
//
// Parameters:
//   text  = the command-line argument to parse
//   value = out: the parsed number
// Returns:
//   0 on success, 2 when text is not exactly one in-range number
// ---------------------------------------------------------------------------
static int parse_long(const char* text, long* value)
{
  char* end = NULL;
  errno = 0;
  *value = strtol(text, &end, 10);
  if (end == text || *end != '\0' || errno == ERANGE) {
    fprintf(stderr, "fft_parity_probe: not a long integer: %s\n", text);
    return 2;
  }
  return 0;
}

// ---------------------------------------------------------------------------
// Print the probe line for one request: the request, the direct
// next_fft_size result, the caller-level even selection and its
// retry count, separated by single spaces.
//
// Parameters:
//   request = the transform-length request to probe
// Returns:
//   nothing; one line goes to stdout
// ---------------------------------------------------------------------------
static void print_probe_line(const long request)
{
  const long direct = next_fft_size(request);
  long retries = 0;
  const long caller = caller_even_selection(request, &retries);
  printf("%ld %ld %ld %ld\n", request, direct, caller, retries);
}

int main(int argc, char* argv[])
{
  if (argc >= 4 && strcmp(argv[1], "range") == 0) {
    long lo = 0;
    long hi = 0;
    if (parse_long(argv[2], &lo) || parse_long(argv[3], &hi)) {
      return 2;
    }
    if (lo < 1 || hi < lo) {
      fprintf(stderr, "fft_parity_probe: need 1 <= lo <= hi\n");
      return 2;
    }
    for (long request = lo; request <= hi; request++) {
      print_probe_line(request);
    }
    return 0;
  }
  if (argc >= 3 && strcmp(argv[1], "cases") == 0) {
    for (int i = 2; i < argc; i++) {
      long request = 0;
      if (parse_long(argv[i], &request)) {
        return 2;
      }
      print_probe_line(request);
    }
    return 0;
  }
  if (argc == 3 && strcmp(argv[1], "one") == 0) {
    long request = 0;
    if (parse_long(argv[2], &request)) {
      return 2;
    }
    // For a request below 1 or beyond every representable length this
    // call does not return: next_fft_size reports through log_fatal
    // and exits with status 1, which is what the driver's refusal
    // checks assert on.
    printf("%ld\n", next_fft_size(request));
    return 0;
  }
  fprintf(stderr,
          "usage: fft_parity_probe range <lo> <hi> | cases <n> ... |"
          " one <n>\n");
  return 2;
}
