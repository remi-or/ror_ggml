#include "test_functions.c"

__m256 mul_sqr_sum_int8_fp32(__m256i x, __m256i y) {
    /*
    Given two 32*int8 vectors x = [x0, ..., x31] and y = [y0, ..., y31] computes 
    [sum_{i=0}^3(xi² * yi²), sum_{i=4}^7(xi² * yi²), ...] and returns it as a 8*fp32 vector.
    Assumes y is in [-127, 127].
    */
    
    // V_plus = [x0y0 + x1y1, ....] (16*int16)
    const __m256i V_plus = _mm256_maddubs_epi16(x, y);
    // V_minus = [x0y0 - x1y1, ....] (16*int16)
    const __m256i sign_mask =  _mm256_set1_epi16 ((short) 384);
    const __m256i alt_y =  _mm256_sign_epi8(y, sign_mask);
    const __m256i V_minus = _mm256_maddubs_epi16(x, alt_y);
    // _mm256_dpwssds_epi32 with themselves , which squares their elements and adds them 2 by 2
    __m256i acc = _mm256_setzero_si256();
    acc = _mm256_madd_epi16 (V_plus, V_plus);
    acc = _mm256_add_epi32(acc, _mm256_madd_epi16(V_minus, V_minus));

    // Right here, acc = 2*[(x0y0)² + (x1y1)² + (x2y2)² + (x3y3)², ....] (8*int32) so we divide it by 2
    return _mm256_cvtepi32_ps(acc); // Stop point, no div by 2 <- TODO
    acc = _mm256_srli_epi32(acc, (int) 1);
    
    // Return the converted vector [(x0y0)² + (x1y1)² + (x2y2)² + (x3y3)², ....] (8*fp32)
    return _mm256_cvtepi32_ps(acc);
}

void mul_sqr_sum_int8_fp32_scalar(char* x, char* y, float* r_acc) {
    /*
    Given two 32*int8 vectors x = [x0, ..., x31] and y = [y0, ..., y31] computes 
    [sum_{i=0}^3(xi² * yi²), sum_{i=4}^7(xi² * yi²), ...] and returns it as a 8*fp32 vector.
    */

    int acc; 
    for (int i = 0; i < 8; i++) {
        acc = 0;
        for (int j = 0; j < 4; j++) {
            acc += ((int) *x) * ((int) *x) * ((int) *y) * ((int) *y);
            x += 1;
            y += 1;
        }
        r_acc[i] = (float) 2*acc;
    }
}

int main() {
    srand (time(NULL));

    __m256i x = _mm256_setzero_si256();
    __m256i y = _mm256_setzero_si256();
    __m256 result, ref;

    for (int i = 0; i < 1000; i++) {
        _mm256_randomize_epi8(&x, 0, 16);
        _mm256_randomize_epi8(&y, -127, 128);
        result = mul_sqr_sum_int8_fp32(x, y);
        mul_sqr_sum_int8_fp32_scalar((char*) &x, (char*) &y, (float*) &ref);
        if (1 - _mm256_equal_ps(&result, &ref)) {
            printf("Test failed.\n");
            break;
        }
    }

    _mm256i_print_int8(&x);
    _mm256i_print_int8(&y);
    _mm256_print_fp32(&result);
    _mm256_print_fp32(&ref);
}