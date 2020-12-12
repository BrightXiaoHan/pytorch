#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/Tensor.h"
#else

struct THPTensor{
    THPTensor(real value): cdata(value) {};
    real cdata;
};

void THPTensor_(add)(THPTensor *a, THPTensor *b, THPTensor *c){
    THPTensor_CData(a) = THPTensor_CData(b) + THPTensor_CData(c);
}
#endif