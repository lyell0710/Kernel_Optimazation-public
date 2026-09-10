#include "reduce_common.h"
float cpu_reduce(const float* data, int n)
{
    float sum = 0.0f;
    for (int i = 0; i < n; i++)
    {
        sum += data[i];
    }
    return sum;
}