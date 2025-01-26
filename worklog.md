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


~~make, output ERROR:~~

~~CMake Error in CMakeLists.txt:~~
  ~~CMAKE_CUDA_ARCHITECTURES must be set to ivcore10 or ivcore11.~~

~~tried add set(CMAKE_CUDA_ARCHITECTURES ivcore10) in CMakeLists.txt, but it didnt work.~~

use iluvatar branch


# 1.25

read sample code , try to complete gather.

# 1.26
I caused the server to crash yesterday while testing the CUDA code
Reason: I thought ctrl-z was to end the task, but in reality it is suspended and running in the background. Ending the task is ctrl-c

Quickly initialize the server:

```
echo "alias _nvcc='clang++ -lcudart -L\$COREX_HOME/lib'" >> ~/.bashrc
echo "export COREX_HOME=/usr/local/corex" >> ~/.bashrc
echo "export PATH=\$PATH:\$COREX_HOME/bin" >> ~/.bashrc
echo "export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:\$COREX_HOME/lib" >> ~/.bashrc
echo "export PYTHONPATH=/usr/local/corex/lib64/python3/dist-packages" >> ~/.bashrc
echo 'export PS1="\u \$(basename \$(dirname \$PWD))/\$(basename \$PWD) \$ "' >> ~/.bashrc
source ~/.bashrc
```



