#! https://zhuanlan.zhihu.com/p/336416611
# Pytorch源码解析-通过宏实现不同类型的Tensor定义
pytorch中有多种Tensor类型，如FloatTensor, IntTensor, LongTensor等，这些数据类型以及其对应的方法在c++中都对应特定类型的实现。但是对于每一种数据类型都实现一套类似的代码，只是底层数据类型不同显然很麻烦，一方面不利于代码的维护，也不利于类型的扩展。比较容易想到的方式是使用C++中的模板，而由于Pytorch底层是基于 torch 的底层代码，底层是用c语言实现的，而C语言中没有模板的概念，那么是如何实现模板的类似功能的呢，本文就来解答这个问题。

## 使用`#define`,`undefine`实现多种Tensor类型的定义
比如我们要实现THPFloatTensor, THPIntTensor两种数据类型，每个类型对应一个add方法实现加法操作。

第一步：首先创建一个通用的模板代码
```c
// generic/Tensor.h
struct THPTensor{
    THPTensor(real value): cdata(value) {};
    real cdata;
};

void THPTensor_(add)(THPTensor *a, THPTensor *b, THPTensor *c){
    THPTensor_CData(a) = THPTensor_CData(b) + THPTensor_CData(c);
}
```
第二步：通过宏定义生成不同的代码
```c++
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

#define TH_GENERIC_FILE "generic/Tensor.h"

#define Real Float
#define real float
#line 1 TH_GENERIC_FILE
#include TH_GENERRIC_FILE
#undef Real
#undef real


#define Real Int
#define real int
#line 1 TH_GENERIC_FILE
#include TH_GENERIC_FILE
#undef Real
#undef real
```
简单而言，上述代码通过宏替换实现了如下功能，首先`THPTensor`->`THPRealTensor`->(`THPFloatTensor`, `ThPIntTensor`)，`THPTensor_(add)`->`THPFloatTensor_add`->`THPIntTensor_add`。对此我们可以编写一个简单的程序验证一下
```c++
// Test.cpp
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
```
编译运行程序
```
g++ -I . test.cpp -o test
./test
```
程序输出
```
7 + 3 = 4
7.000000 + 3.000000 = 4.000000
```
当我们编译运行程序，并看到这样的输出时，便成功使用宏生成了多种数据类型的简单Tensor。特别是当有数种Tensor类型时以及上百个与特定Tensor类型绑定的方法时（add, matmul, log等等），这种方法避免了针对每一种Tensor编写重复的代码。而pytorch中不止是Tensor，以及底层的Storage等与特定数据类型绑定的代码都是通过这种机制生成的。

完整的代码可以参考我的Github[Pytorch源码解析](https://github.com/BrightXiaoHan/pytorch/tree/LearnSourceCode/%E6%BA%90%E7%A0%81%E8%A7%A3%E6%9E%90)（一定要看哟，便于理解）。

## Pytorch中多种Tensor类型的定义
首先当我们创建了一个FloatTensor时都发生了什么呢
```python
import torch
a = torch.FloatTensor([1,2,3])
```
首先我们看FloatTensor Python 的定义 (pytorch 0.31 torch/__init__.py:173)
```python
class FloatTensor(_C.FloatTensorBase, _TensorBase):

    def is_signed(self):
        return True

    @classmethod
    def storage_type(cls):
        return FloatStorage
```
可以看到它继承了`_C.FloatTensorBase`，`_TensorBase`两个基类，`_TensorBase`不用说，它里面定义了一些python实现的绑定方法，最终要的是`_C.FloatTensorBase`，我们在python文件中找不到对应的实现，它其实是C++代码定义的一个扩展类型。（具体到python是如何定义C++扩展类型的，可以看我前面的文章[使用c/c++编写python扩展（三）：自定义Python内置类型](https://zhuanlan.zhihu.com/p/106773873)）。_C.FloatTensorBase类型定义是在`pytorch 0.31 torch/csrc/generic/Tensor.h, Tensor.cpp）`两个文件中。肯定有人会问，我在这两个文件中没有找到`FloatTensorBase`，其实它的定义是在`torch/csrc/generic/Tensor.cpp:1658`
```c++
// TODO: implement equality
PyTypeObject THPTensorType = {
  PyVarObject_HEAD_INIT(NULL, 0)
  "torch._C." THPTensorBaseStr,          /* tp_name */
  sizeof(THPTensor),                     /* tp_basicsize */
  0,                                     /* tp_itemsize */
  (destructor)THPTensor_(dealloc),       /* tp_dealloc */
  ...
```
这里的`"torch._C." THPTensorBaseStr`会以宏替换的形式生成`FloatTensorBase`，`IntTensorBase`等多种类型，生成机制就是我们上面讲到的宏替换机制。

在`torch/csrc/generic/Tensor.h:11`中可以看到，我们python类型的Tensor，底层其实就是包装了一个THTensor类型。
```
struct THPTensor {
  PyObject_HEAD
  // Invariant: After __new__ (not __init__), this field is always non-NULL.
  THTensor *cdata;
};
```
同样THTensor类型也通过相同的宏替换机制，生成不同的类型，并绑定到对应的THPTensor中。

## 总结
其实遍历整个源码，所有的动态类型代码生成都是采取这样的生成机制（pytorch采用这种生成代码机制的原因其实是和老的torch项目代码有关。）
- generic文件夹中定义模板代码
- 通过宏定义不同的 `GENERIC_FILE` （GENERIC_FILE通常是generic文件夹中的模板文件），通过`torch/lib/TH/THGenerateAllTypes.h`文件多次`#include GENERIC_FILE`来生成不同类型的代码
