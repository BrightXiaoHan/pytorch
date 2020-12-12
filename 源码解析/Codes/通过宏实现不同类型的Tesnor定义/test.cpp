#include <stdio.h>
#include "Tensor.h"


int main(int argc, char const *argv[])
{
#define Real Int
#define real int
    THPTensor it1(2);
    THPTensor it2(3);
    THPTensor it3(4);
    THPTensor_(add)(&it1, &it2, &it3);
    printf("%d + %d = %d\n", THPTensor_CData(&it1), THPTensor_CData(&it2), THPTensor_CData(&it3));
#undef Real
#undef real

#define Real Float
#define real float
    THPTensor ft1(2.0);
    THPTensor ft2(3.0);
    THPTensor ft3(4.0);
    THPTensor_(add)(&ft1, &ft2, &ft3);
    printf("%f + %f = %f\n", THPTensor_CData(&ft1), THPTensor_CData(&ft2), THPTensor_CData(&ft3));
#undef Real
#undef real
    return 0;
}
