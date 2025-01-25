# 1.15
Test the connection of server, try the hello sample.

puckpone:naive$ clang++ -lcudart -L$COREX_HOME/lib he.cu -o he

puckpone:naive$ ./he

hello world from GPU by thread:0
hello world from GPU by thread:1
hello world from GPU by thread:2
hello world from GPU by thread:3

# 1.17
read matmul code

# 1.18
add alias _nvcc='clang++ -lcudart -L$COREX_HOME/lib' to ~/.bashrc

read softmax code

make TinyInfiniTensor

# 1.21
read gather in ONNX Doc

accept three arguments: data, indices, axis. 

return a ouput martrix

```
data = [
    [1.0, 1.2],
    [2.3, 3.4],
    [4.5, 5.7],
]
indices = [
    [0, 1],
    [1, 2],
]
output = [
    [
        [1.0, 1.2],
        [2.3, 3.4],
    ],
    [
        [2.3, 3.4],
        [4.5, 5.7],
    ],
]
```

```
data = [
    [1.0, 1.2, 1.9],
    [2.3, 3.4, 3.9],
    [4.5, 5.7, 5.9],
]
indices = [
    [0, 2],
]
axis = 1,
output = [
        [[1.0, 1.9]],
        [[2.3, 3.9]],
        [[4.5, 5.9]],
]
```

---
~~make, output ERROR:~~

~~CMake Error in CMakeLists.txt:~~
  ~~CMAKE_CUDA_ARCHITECTURES must be set to ivcore10 or ivcore11.~~

~~tried add set(CMAKE_CUDA_ARCHITECTURES ivcore10) in CMakeLists.txt, but it didnt work.~~

use iluvatar branch

---



---
run.sh
默认编译CPU端代码，运行仓库命令是：

bash run.sh

编译结束以后，可以直接做python端测试，测试softmax算子的CPU端代码命令为：

python test/test_softmax.py --device cpu

如果需要编译测试其他平台代码，比如说GPU端测试，那么修改run.sh里面的cmake ../ -DUSE_CPU=ON为 cmake ../ -DUSE_CUDA=ON，对应的测试python脚本--device cpu也修改为--device cuda
---

# 1.25

read sample code , try to complete gather.

