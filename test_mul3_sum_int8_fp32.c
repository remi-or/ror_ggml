#include "test_functions.c"

// New function: 2nd term aux function
static inline __m256i _mm256_xy2_epi8_epi32(const __m128i x, const __m128i y) {
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

// New function: 2nd term main function
static inline __m256 _mul3_sum_int8_fp32(const __m256i x, const __m256i y) {
    /*
    Given two 32*int8 vectors x = [x0, ..., x31] and y = [y0, ..., y31] computes 
    [sum_{i=0}^3(2*xi*yi²), sum_{i=4}^7(2*xi*yi²), ...] and returns it as a 8*fp32 vector.
    Assumes xs is in the [0, 15] range.
    */

    // Compute result as low and high part
    const __m128i* ptr_x = (__m128i*) &x;
    const __m128i* ptr_y = (__m128i*) &y;
    const __m256i result_low = _mm256_xy2_epi8_epi32(ptr_x[0], ptr_y[0]); // [2x0y0² + 2x1y1², ..., 2x14y14² + 2x15y15²]
    const __m256i result_high = _mm256_xy2_epi8_epi32(ptr_x[1], ptr_y[1]); // [2x16y16² + 2x17y17², ..., 2x30y30² + 2x31y31²]

    // Compute shuffled result
    __m256i shuffled = _mm256_hadd_epi32(result_low, result_high); // [r0, r1, r4, r5, r2, r3, r6, r7]
    // Unshuffle it
    const __m256i permutation = _mm256_set_epi32(7, 6, 3, 2, 5, 4, 1, 0);
    shuffled = _mm256_permutevar8x32_epi32(shuffled, permutation);

    return _mm256_cvtepi32_ps(shuffled);
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

    // Shuffle test
    __m256i aranged;
    _mm256_arange_epi32(&aranged);
    _mm256i_print_int32(&aranged);
    const __m256i permutation = _mm256_set_epi32(7, 6, 3, 2, 5, 4, 1, 0);
    __m256i shuffled = _mm256_permutevar8x32_epi32(aranged, permutation);
    _mm256i_print_int32(&shuffled);

    srand (time(NULL));

    __m256i x = _mm256_setzero_si256();
    __m256i y = _mm256_setzero_si256();
    __m256 result, ref;

    for (int i = 0; i < 1; i++) {
        _mm256_randomize_epi8(&x, 0, 16);
        _mm256_randomize_epi8(&y, -127, 128);
        result = _mul3_sum_int8_fp32(x, y);
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

