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
add alias nvcc='clang++ -lcudart -L$COREX_HOME/lib' to ~/.bashrc
read softmax code