// ---------------------------------------------------------------------------
// The shared NFW profile table lives entirely in halo_profile.h: the
// readers are inlined into their callers, and the builders are inline
// as well, so every target that compiles a consumer links without a
// dedicated object - the standalone three-point covariance target
// among them. This translation unit stays as the conventional home
// that project build files name; it defines nothing of its own.
// ---------------------------------------------------------------------------
#include "halo_profile.h"
