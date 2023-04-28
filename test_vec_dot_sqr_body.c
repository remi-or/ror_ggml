// gcc -Wall -march=native -lm -mavx -march=native test_vec_dot_sqr_body.c && ./a.out 

#include "test_functions.c"

#define N_TEST 2048
#define MIN_FP32 -1000.0f
#define MAX_FP32 +1000.0f


static inline __m256 sum_i16_pairs_float(const __m256i x) {
    const __m256i ones = _mm256_set1_epi16(1);
    const __m256i summed_pairs = _mm256_madd_epi16(ones, x);
    /// _mm256_madd_epi16 : [int16_a, int16_b, ...] , [int16_A, int16_B] -> [int32_(aA+bB), ...]
    /// _mm256_madd_epi16(1, ...) : [int16_a, int16_b, ...] -> [int32_(a+b), ...]
    return _mm256_cvtepi32_ps(summed_pairs);
}

static inline __m256 mul_sum_i8_pairs_float(const __m256i x, const __m256i y) {
    // Get absolute values of x vectors
    const __m256i ax = _mm256_sign_epi8(x, x);
    // Sign the values of the y vectors
    const __m256i sy = _mm256_sign_epi8(y, x);
#if __AVXVNNI__
    const __m256i zero = _mm256_setzero_si256();
    const __m256i summed_pairs = _mm256_dpbusd_epi32(zero, ax, sy);
    return _mm256_cvtepi32_ps(summed_pairs);
#else
    // Perform multiplication and create 16-bit values
    /// maddubs: [uint8_a, uint8_b, ...] , [int8_A, int8_B, ...] -> [int16_(aA+bB), ...]
    const __m256i dot = _mm256_maddubs_epi16(ax, sy);
    return sum_i16_pairs_float(dot);
#endif
}

static inline __m256 mul_sqr_sum_int8_fp32(__m256i x, __m256i y) {
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
    acc = _mm256_srli_epi32(acc, (int) 1);
    
    // Return the converted vector [(x0y0)² + (x1y1)² + (x2y2)² + (x3y3)², ....] (8*fp32)
    return _mm256_cvtepi32_ps(acc);
}

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
 // TODO Scan from here
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


static inline __m256 vec_dot_sqr_body(float scalar_dx0, float scalar_dx1, float scalar_dy,
                                      float scalar_m0, float scalar_m1,
                                      __m256i bx, __m256i by)
{
    __m256 acc = _mm256_setzero_ps();

    // Compute dx = [dx1, dx1, dx1, dx1, dx0, dx0, dx0, dx0] (8*fp32)
    const __m128 d0 = _mm_set1_ps(scalar_dx0);
    const __m128 d1 = _mm_set1_ps(scalar_dx1);
    const __m256 dx = _mm256_set_m128(d1, d0);

    // Compute m = [m1, m1, m1, m1, m0, m0, m0, m0] (8*fp32)
    const __m128 m0 = _mm_set1_ps(scalar_m0);
    const __m128 m1 = _mm_set1_ps(scalar_m1);
    const __m256 ms = _mm256_set_m128(m1, m0);

    // Load dy = [dy, ..., dy] (8*fp32)
    const __m256 dy = _mm256_set1_ps(scalar_dy);

    // Compute delta = dx*dy and ms_dy = ms*dy
    const __m256 delta = _mm256_mul_ps(dx, dy);
    const __m256 ms_dy = _mm256_mul_ps(ms, dy);

    // Accumulate 1st term
    const __m256 first_term = mul_sqr_sum_int8_fp32(bx, by);
    acc = _mm256_fmadd_ps(first_term, _mm256_mul_ps(delta, delta), acc);
    // Accumulate 2nd term
    const __m256 second_term = _mul3_sum_int8_fp32(bx, by);
    acc = _mm256_fmadd_ps(second_term, _mm256_mul_ps(delta, ms_dy), acc);
    // Accumulate 3rd term
    const __m256 third_term = mul_sum_i8_pairs_float(by, by);
    acc = _mm256_fmadd_ps(third_term, _mm256_mul_ps(ms_dy, ms_dy), acc);

    return acc;
}

static inline __m256 vec_dot_sqr_body_scalar(float dx0, float dx1, float dy,
                                             float m0, float m1,
                                             __m256i bx, __m256i by)
{
    __m256 acc = _mm256_setzero_ps();
    float* acc_ptr = (float*) &acc;

    char* bx_ptr = (char*) &bx;
    char* by_ptr = (char*) &by;
    long x, y;
    float local_acc, m, dx, delta;

    for (int i = 0; i < 8; i++) {

        local_acc = 0.0f;
        m = (i < 4) ? m0 : m1;
        dx = (i < 4) ? dx0 : dx1;
        delta = dx * dy;

        for (int j = 0; j < 4; j++) {
            x = (long) *bx_ptr;
            y = (long) *by_ptr;
            local_acc += (delta*delta)  *  (float) (x*x*y*y);
            local_acc += (2*delta*m*dy) *  (float) (x*y*y);
            local_acc += (m*m*dy*dy)    *  (float) (y*y);

            bx_ptr++;
            by_ptr++;
        }
        acc_ptr[i] = local_acc;
    }
    return acc;
}

int main()
{
    srand(time(NULL));

    float dx0, dx1, dy, m0, m1;
    __m256i bx = _mm256_setzero_si256();
    __m256i by = _mm256_setzero_si256();
    __m256 result, ref;

    // Binary tests
    dx0 = 1;
    dx1 = 1;
    dy = 1;
    m0 = 0;
    m1 = 0;
    bx = _mm256_set1_epi8(1);
    by = _mm256_set1_epi8(1);
    result = vec_dot_sqr_body(dx0, dx1, dy, m0, m1, bx, by);
    ref = vec_dot_sqr_body_scalar(dx0, dx1, dy, m0, m1, bx, by);
    if (1 - _mm256_equal_ps(&result, &ref)) {
            printf("Binary test failed.\n");
            printf("Scalar: ");
            _mm256_print_fp32(&ref);
            printf("Result: ");
            _mm256_print_fp32(&result);
            exit(1);
    }

    // Simple tests
    dx0 = 2;
    dx1 = 3;
    dy = 5;
    m0 = 1;
    m1 = 2;
    bx = _mm256_set1_epi8(1);
    by = _mm256_set1_epi8(1);
    result = vec_dot_sqr_body(dx0, dx1, dy, m0, m1, bx, by);
    ref = vec_dot_sqr_body_scalar(dx0, dx1, dy, m0, m1, bx, by);
    if (1 - _mm256_equal_ps(&result, &ref)) {
            printf("Simple test failed.\n");
            printf("Scalar: ");
            _mm256_print_fp32(&ref);
            printf("Result: ");
            _mm256_print_fp32(&result);
            exit(1);
    }

    // Unit tests
    for (int i = 0; i < N_TEST; i++)
    {
        dx0 = random_fp32(MIN_FP32, MAX_FP32);
        dx1 = random_fp32(MIN_FP32, MAX_FP32);
        dy = random_fp32(MIN_FP32, MAX_FP32);
        m0 = random_fp32(MIN_FP32, MAX_FP32);
        m1 = random_fp32(MIN_FP32, MAX_FP32);
        _mm256_randomize_epi8(&bx, 0, 16);
        _mm256_randomize_epi8(&by, -127, 128);

        result = vec_dot_sqr_body(dx0, dx1, dy, m0, m1, bx, by);
        ref = vec_dot_sqr_body_scalar(dx0, dx1, dy, m0, m1, bx, by);
        if (1 - _mm256_close_ps(&result, &ref, 5e-4f))
        {
            printf("Unit test %d test failed.\n\nInputs:\n", i);
            printf("dx0: %f\ndx1: %f\ndy: %f\nm0: %f\nm1: %f\nbx: ", dx0, dx1, dy, m0, m1);
            _mm256i_print_int8(&bx);
            printf("by: ");
            _mm256i_print_int8(&by);
            printf("\nScalar: ");
            _mm256_print_fp32(&ref);
            printf("Result: ");
            _mm256_print_fp32(&result);
            exit(1);
        }
    }

    printf("All tests were passed.\n");

    dx0 = 2.5f;
    dx1 = -3.0f;
    dy = 4.0f;
    m0 = 1.5f;
    m1 = -3.0f;
    _mm256_randomize_epi8(&bx, 0, 16);
    _mm256_randomize_epi8(&by, -127, 128);
    result = vec_dot_sqr_body(dx0, dx1, dy, m0, m1, bx, by);
    ref = vec_dot_sqr_body_scalar(dx0, dx1, dy, m0, m1, bx, by);
    printf("dx0: %f\ndx1: %f\ndy: %f\nm0: %f\nm1: %f\nbx: ", dx0, dx1, dy, m0, m1);
    _mm256i_print_int8(&bx);
    printf("by: ");
    _mm256i_print_int8(&by);
    printf("\nScalar: ");
    _mm256_print_fp32(&ref);
    printf("Result: ");
    _mm256_print_fp32(&result);
}
