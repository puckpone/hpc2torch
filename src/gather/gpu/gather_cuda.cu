#include <cstdio>

static __global__ void gather(int unit, void *output, void const *indices, void const *input) {
}

extern "C" void gather_nv(int unit, int y, int x, int yi, int xi, int axis, void *output, void const *indices, void const *input) {
    printf("gather_nv called (%d, %d) -> (%d, %d), unit = %d, axis = %d\n", yi, xi, y, x, unit, axis);
    gather<<<1, 1>>>(unit, output, indices, input);
}
