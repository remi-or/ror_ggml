// gcc -mavx512f -march=native unit_test_added_fn.c && ./a.out 

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <immintrin.h>
#include <xmmintrin.h>

static inline __m256 mul_sqr_sum_int8_fp32(const __m256i x, const __m256i y) {
    /*
    Given two 32*int8 vectors x = [x0, ..., x31] and y = [y0, ..., y31] computes 
    [sum_{i=0}^3(xi² * yi²), sum_{i=4}^7(xi² * yi²), ...] and returns it as a 8*fp32 vector.
    */

    // Get absolute values of x vectors
    const __m256i ax = _mm256_sign_epi8(x, x);
    // Sign the values of the y vectors
    const __m256i sy = _mm256_sign_epi8(y, x);
    
    // V_plus = [x0y0 + x1y1, ....] (16*int16)
    const __m256i V_plus = _mm256_maddubs_epi16(ax, sy);
    // V_minus = [x0y0 - x1y1, ....] (16*int16)
    const __m256i sign_mask =  _mm256_set1_epi16 ((short) 384);
    const __m256i ssy =  _mm256_sign_epi8(sy, sign_mask);
    const __m256i V_minus = _mm256_maddubs_epi16(ax, ssy);
    // _mm256_dpwssds_epi32 with themselves , which squares their elements and adds them 2 by 2
    __m256i acc = _mm256_setzero_si256();
    acc = _mm256_dpwssds_epi32(V_plus, V_plus, acc);
    acc = _mm256_dpwssds_epi32(V_minus, V_minus, acc);

    // Right here, acc = 2*[(x0y0)² + (x1y1)² + (x2y2)² + (x3y3)², ....] (8*int32) so we divide it by 2
    acc = _mm256_srai_epi32(acc, 1);
    
    // Return the converted vector [(x0y0)² + (x1y1)² + (x2y2)² + (x3y3)², ....] (8*fp32)
    return _mm256_cvtepi32_ps(acc);
}

void mul_sqr_sum_int8_fp32_scalar(signed char* x, signed char* y, float* result) {
    /*Performs the same computation as [mul_sqr_sum_int8_fp32] on two
    int8 vectors x and y of length*/
}

void _mm256_randomize_epi8(__m256i* x, int minimum, int maximum) {
    int range = maximum - minimum;
    signed char * ptr = (signed char*) x;
    for (int i = 0; i < 32; i++) {
        ptr[i] = (rand() % range) + minimum;
    }
}


int main() {

    __m256i x, y;
    _mm256_randomize_epi8(&x, 0, 16);
    _mm256_randomize_epi8(&y, -128, 128);

    __m256 result = mul_sqr_sum_int8_fp32(x, y);

}