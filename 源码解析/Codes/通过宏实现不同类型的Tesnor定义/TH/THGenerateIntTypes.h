#ifndef TH_GENERIC_FILE
#error "You must define TH_GENERIC_FILE before including THGenerateFloatType.h"
#endif

#define Real Int
#define real int
#line 1 TH_GENERIC_FILE
#include TH_GENERIC_FILE
#undef Real
#undef real
