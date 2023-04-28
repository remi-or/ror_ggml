#include "test_functions.c"

__m256i _mm256_xy2_epi8_epi32(const __m128i x, const __m128i y) {
    /*
    Given two 16*int8 vectors x = [x0, ..., x15] and y = [y0, ..., y15] computes 
    [sum_{i=0}^1(2*xi*yi²), sum_{i=2}^3(2*xi*yi²), ...] and returns it as a 8*int32 vector.
    Assumes xs is in the [0, 15] range.
    */

    // Convert the vectors from 32*int8 to 32*int16
    const __m256i wide_x = _mm256_cvtepi8_epi16(x); 
    const __m256i wide_y = _mm256_cvtepi8_epi16(y); 

    // Multiply x by 2; Square y
    const __m256i wide_2x = _mm256_slli_epi16(wide_x, 1);
    const __m256i wide_y2 = _mm256_mullo_epi16(wide_y, wide_y); 

    // Since (0 <= x <= 2^5) and (0 <= y² <= 2^14) then (0 <= xy² <= 2^19) we don't need saturation to int32
    return _mm256_madd_epi16(wide_2x, wide_y2); 
}

__m256 _mm256_xy2_epi8_fp32(const __m256i x, const __m256i y) {
    /*
    Given two 32*int8 vectors x = [x0, ..., x31] and y = [y0, ..., y31] computes 
    [sum_{i=0}^3(2*xi*yi²), sum_{i=4}^7(2*xi*yi²), ...] and returns it as a 8*fp32 vector.
    Assumes xs is in the [0, 15] range.
    */

    // Compute result as low and high part
    __m128i* ptr_x = (__m128i*) &x;
    __m128i* ptr_y = (__m128i*) &y;
    __m256i result_low = _mm256_xy2_epi8_epi32(ptr_x[0], ptr_y[0]); // [r0, ..., r15]
    __m256i result_high = _mm256_xy2_epi8_epi32(ptr_x[1], ptr_y[1]); // [r16, ..., r31]

    // Prepare result
    __m256i result = _mm256_setzero_si256();
    __m128i* ptr_r = (__m128i*) &result;
    // Accumulate low
    __m128i* ptr_low = (__m128i*) &result_low;
    __m128i summed_result_low = _mm_add_epi32(ptr_low[0], ptr_low[1]);
    ptr_r[0] = _mm_add_epi32(ptr_low[0], ptr_low[1]);
    // Accumulate low
    __m128i* ptr_high = (__m128i*) &result_high;
    __m128i summed_result_high = _mm_add_epi32(ptr_high[0], ptr_high[1]);
    ptr_r[1] = _mm_add_epi32(ptr_high[0], ptr_high[1]);
    return _mm256_setzero_ps();

    return _mm256_cvtepi32_ps(result);
}

void xy2_epi8_fp32_scalar(char* x, char* y, float* r_acc) {
    /*
    Given two 32*int8 vectors x = [x0, ..., x31] and y = [y0, ..., y31] computes 
    [sum_{i=0}^3(2*xi*yi²), sum_{i=4}^7(2*xi*yi²), ...] and returns it as a 8*fp32 vector.
    */

    int acc; 
    for (int i = 0; i < 8; i++) {
        acc = 0;
        for (int j = 0; j < 4; j++) {
            acc += 2 * ((int) *x) * ((int) *y) * ((int) *y);
            x += 1;
            y += 1;
        }
        r_acc[i] = (float) acc;
    }
}

int main() {
    srand (time(NULL));

    __m256i x = _mm256_setzero_si256();
    __m256i y = _mm256_setzero_si256();
    __m256 result, ref;

    for (int i = 0; i < 1; i++) {
        _mm256_randomize_epi8(&x, 0, 16);
        _mm256_randomize_epi8(&y, -127, 128);
        result = _mm256_xy2_epi8_fp32(x, y);
        xy2_epi8_fp32_scalar((char*) &x, (char*) &y, (float*) &ref);
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

