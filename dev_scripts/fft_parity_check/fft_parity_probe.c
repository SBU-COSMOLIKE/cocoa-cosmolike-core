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
//   range <lo> <hi>  one line "n direct even" for every request n
//                    from lo to hi: direct is next_fft_size(n) and
//                    even is next_even_fft_size_radix7(n), the
//                    selector every even-required production call
//                    site uses
//   cases <n> ...    the same line for each listed request
//   one <n>          a single next_fft_size(n) call, result printed.
//                    The driver runs this mode in a child process for
//                    the requests that must terminate the program
//                    through log_fatal: a request below 1, and a
//                    request above every representable length.
//   evenone <n>      the same child-process mode for
//                    next_even_fft_size_radix7(n)
//
// Exit codes: 0 after printing every requested line, 2 on a usage or
// parse error. Modes "one" and "evenone" never return for a refused
// request, since both selectors call exit(1) through log_fatal.

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "cosmolike/basics.h"

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
// next_fft_size result and the next_even_fft_size_radix7 result -
// the selector J_abl (cfastpt/cfastpt.c) and the FFTLog spectra of
// cosmolike/cosmo2D.c and cosmolike/cosmo2D_cluster.c call for their
// even transform lengths - separated by single spaces. Nothing is
// replicated here: both numbers come from the live basics.c.
//
// Parameters:
//   request = the transform-length request to probe
// Returns:
//   nothing; one line goes to stdout
// ---------------------------------------------------------------------------
static void print_probe_line(const long request)
{
  const long direct = next_fft_size(request);
  const long even = next_even_fft_size_radix7(request);
  printf("%ld %ld %ld\n", request, direct, even);
}

int main(int argc, char* argv[])
{
  if (argc >= 4 && strcmp(argv[1], "range") == 0) {
    long lo = 0;
    long hi = 0;
    if (parse_long(argv[2], &lo) || parse_long(argv[3], &hi)) {
      return 2;
    }
    // The even selector refuses requests below 2 (its domain floor),
    // so the shared probe line starts there; the refusal modes cover
    // the floor itself.
    if (lo < 2 || hi < lo) {
      fprintf(stderr, "fft_parity_probe: need 2 <= lo <= hi\n");
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
  if (argc == 3 && strcmp(argv[1], "evenone") == 0) {
    long request = 0;
    if (parse_long(argv[2], &request)) {
      return 2;
    }
    // The even selector's own refusals: a request below 2, and a
    // request whose even-stepping scan finds no admissible length
    // before the end of the long range.
    printf("%ld\n", next_even_fft_size_radix7(request));
    return 0;
  }
  fprintf(stderr,
          "usage: fft_parity_probe range <lo> <hi> | cases <n> ... |"
          " one <n> | evenone <n>\n");
  return 2;
}
