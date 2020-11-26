#! https://zhuanlan.zhihu.com/p/295318684
# Pytorch源码解析-搭建Pytorch源码学习环境
工欲善其事，必先利其器。在正式讲解pytorch源代码之前，我们先基于pytorch源码搭建实验环境。这里我们选择基于pytorch 0.31版本进行讲解。选择这个版本的原因是在0.4版本之后，pytorch将caffe合并了进来，后续一些复杂的特性如jit、移动端支持、windows支持等特性增加了项目的复杂度，不利于我们学习pytorch的核心原理。与新版本的一个比较大的区别是0.31之后将Tensor和Variable进行了合并，但是这反而比较有利于我们进行学习，合并后可以求导的Tensor与0.31版本之前的Variable原理基本一致，我们可以由浅入深，先学习不带求导功能的Tensor，之后再来理解Variable。

本节我们就下载并编译pytorch源码，编写一个简单的程序，并对其python，c++代码进行联合调试。
## 实验环境
- 操作系统：ubuntu 16.04+ 或者 Centos 7+
- 软件：Anaconda3，gcc，gdb，cmake, nvidia-driver
- IDE：Vscode + Python, C++,  Bookmarks插件
## 编译源码
克隆pytorch源码，并切换到实验版本v0.3.1
```
git clone https://github.com/pytorch/pytorch.git
git checkout v0.3.1
```
创建conda虚拟环境并编译源码(具体可参考pytorchv0.3.1 README.md 文档)
```bash
# create conda virtual env
conda create -n pytorch_source python=3.6 anaconda
conda activate pytorch_source
export CMAKE_PREFIX_PATH="$(dirname $(which conda))/../" # [anaconda root directory]

# Install basic dependencies
conda install numpy pyyaml mkl setuptools cmake cffi

# Add LAPACK support for the GPU
conda install -c soumith magma-cuda80 # or magma-cuda75 if CUDA 7.5
```
开启debug模式编译源码
```
export DEBUG=True
python setup.py build_deps  # 编译torch c语言库
python setup.py build_ext --inplace  # 编译torch c++模块
python setup.py build_py  # 构建torch python模块
```
## python c++联合调试
调试的具体方法可以参考我前面的文章，这里只把具体步骤列出来。（注意最好在root用户下运行，因为gdb attach操作需要root权限）

安装ptvsd
```
pip install ptvsd
```
编写一个小程序 hello_torch.py
```python
import torch
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--ptvsd", action="store_true", help="是否启动ptvsd调试。")
args = parser.parse_args()

if args.ptvsd:
    import ptvsd
    ptvsd.enable_attach(address =('127.0.0.1', 10010), redirect_output=True)
    ptvsd.wait_for_attach()

a = torch.Tensor([1, 2, 3])
c = torch.add(a, a)
```
编写配置文件launch.json
```json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Python Attach",
            "type": "python",
            "request": "attach",
            "pathMappings": [
                {
                    "localRoot": "${workspaceFolder}", // You may also manually specify the directory containing your source code.
                    "remoteRoot": "${workspaceFolder}", // Linux example; adjust as necessary for your OS and situation.
                }
            ],
            "port": 10010,
            "host": "localhost"
        },
        {
            "name": "GDB Attach",
            "type": "cppdbg",
            "request": "attach",
            "program": "/home/hanbing/anaconda3/envs/pytorch0.3.1/bin/python",  // 替换成你虚拟环境python可执行程序的路径
            "processId": "${command:pickProcess}",
            "MIMode": "gdb"
        }
    ]
}
```
分别在
- `torch/csrc/generic/TensorMethods.cpp`第`20972`行
- `hello_torch.py`第14行
打上断点。

运行`hello_torch.py`
```
python hello_torch.py --ptvsd
```
在vscode中分别先后运行`Python Attach`, `GDB Attach`两个调试配置。如果观察到程序在`torch/csrc/generic/TensorMethods.cpp`第`20972`行处停下，则表示运行成功。
