#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <immintrin.h>
#include <xmmintrin.h>
#include <time.h>


// AVX print functions
void _mm128i_print_int32(__m128i * src) {
    int * ptr = (int*) src;
    for (int i = 0; i < 4; i++) {
        printf("%d ", ptr[i]);
    }
    printf("\n");
}
void _mm256i_print_int8(__m256i * src) {
    char * ptr = (char*) src;
    for (int i = 0; i < 32; i++) {
        printf("%d ", ptr[i]);
    }
    printf("\n");
}
void _mm256i_print_int32(__m256i * src) {
    int * ptr = (int*) src;
    for (int i = 0; i < 8; i++) {
        printf("%d ", ptr[i]);
    }
    printf("\n");
}
void _mm256_print_fp32(__m256 * src) {
    float* ptr = (float*) src;
    for (int i = 0; i < 8; i++) {
        printf("%f ", ptr[i]);
    }
    printf("\n");
}
void _mm512i_print_int16(__m512i * src) {
    short * ptr = (short*) src;
    for (int i = 0; i < 32; i++) {
        printf("%d ", ptr[i]);
    }
    printf("\n");
}
void _mm512i_print_int32(__m512i * src) {
    int * ptr = (int*) src;
    for (int i = 0; i < 16; i++) {
        printf("%d ", ptr[i]);
    }
    printf("\n");
}

// AVX Set functions
void _mm256_arange_epi32(__m256i * src) {
    __int32_t* ptr = (__int32_t*) src;
    for (int i = 0; i < 8; i++) {
        ptr[i] = (__int32_t) i;
    }
}
void _mm256_randomize_epi8(__m256i* x, int minimum, int maximum) {
    int range = maximum - minimum;
    signed char * ptr = (signed char*) x;
    for (int i = 0; i < 32; i++) {
        ptr[i] = (rand() % range) + minimum;
    }
}
void _mm512_randomize_epi32(__m512i* x, int minimum, int maximum) {
    int range = maximum - minimum;
    int * ptr = (int*) x;
    for (int i = 0; i < 16; i++) {
        ptr[i] = (rand() % range) + minimum;
    }
}

// AVX Compare functions
int _mm256_equal_ps(__m256* x, __m256* y) {
    float* xi = (float*) x;
    float* yi = (float*) y;
    for (int i = 0; i < 8; i++) {
        if (*xi != *yi) {
            return 0;
        }
        xi += 1;
        yi += 1;
    }
    return 1;
}
int _mm256_close_ps(__m256* x, __m256* y, float tol) {
    float* xi = (float*) x;
    float* yi = (float*) y;
    float error;
    for (int i = 0; i < 8; i++) {
        error = (*xi - *yi) / *yi;
        if (fabsf(error) > tol) {
            return 0;
        }
        xi += 1;
        yi += 1;
    }
    return 1;
}

// Misc. functions
float random_fp32(float minimum, float maximum) {
    float unit = (float) rand() / (float) RAND_MAX;
    return (unit * (maximum - minimum)) + minimum;
}