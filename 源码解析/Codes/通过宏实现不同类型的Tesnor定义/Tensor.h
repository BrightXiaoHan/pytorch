// 定义声明拼接的一些宏
#define TH_CONCAT_3(x,y,z) TH_CONCAT_3_EXPAND(x,y,z)
#define TH_CONCAT_3_EXPAND(x,y,z) x ## y ## z
#define TH_CONCAT_4_EXPAND(x,y,z,w) x ## y ## z ## w
#define TH_CONCAT_4(x,y,z,w) TH_CONCAT_4_EXPAND(x,y,z,w)

// 定义python绑定函数声明的宏，如 Real为Float时，THPTensor_(add) -> THPFloatTensor_add
#define THPTensor_(NAME) TH_CONCAT_4(THP, Real, Tensor_, NAME)

// 定义python绑定对象类型，如Real为Float时，THPTensor -> THPFloatTensor
#define THPTensor TH_CONCAT_3(THP,Real,Tensor)

// 定义THPTensor的指针访问
#define THPTensor_CData(obj)  (obj)->cdata

#include "generic/Tensor.h"
#include "TH/THGenerateAllTypes.h"