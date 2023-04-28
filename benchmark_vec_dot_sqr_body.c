// gcc -Wall -march=native -lm -mavx -march=native test_vec_dot_sqr_body.c && ./a.out 

#include "test_functions.c"

#define N_TEST 1000
#define ITER_PER_TEST 1000
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

static inline __m128i bytes_from_nibbles_16(const __int8_t * rsi)
{
    // Load 8 bytes from memory 
    /// _mm_loadl_epi64 : Loads 64-bit integer from memory into first element of returned vector
    __m128i tmp = _mm_loadl_epi64( ( const __m128i* )rsi );

    // Expand bytes into uint16_t values
    __m128i bytes = _mm_cvtepu8_epi16( tmp );

    // Unpack values into individual bytes
    const __m128i lowMask = _mm_set1_epi8( 0xF );
    __m128i high = _mm_andnot_si128( lowMask, bytes );
    __m128i low = _mm_and_si128( lowMask, bytes );
    high = _mm_slli_epi16( high, 4 );
    bytes = _mm_or_si128( low, high );
    return bytes;
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

static inline __m256 vec_dot_body(float scalar_dx0, float scalar_dx1, const float* scalar_dy,
                                      float scalar_m0, float scalar_m1,
                                      __m256i bx, __m256i by)
{
    float summs = 0.0f;
    __m256 acc = _mm256_setzero_ps();

    const __m128 d0 = _mm_set1_ps(scalar_dx0);
    const __m128 d1 = _mm_set1_ps(scalar_dx1);
    const __m256 dx = _mm256_set_m128(d1, d0);

    summs +=  (scalar_m0) * *scalar_dy
            + (scalar_m1) * *scalar_dy;

    const __m256 dy = _mm256_broadcast_ss(scalar_dy); /// le fp32 delta_y broadcast 8 fois dans un vecteur de 8*32=256 bit

    /// ca melange mais osef parce qu'on multiplie par le produit des deux deltas
    const __m256 q = mul_sum_i8_pairs_float(bx, by); /// chaque float32 contient 4 produit de nibble sommés
    /// ce qui donne au final 256/32 = 8 sommes de 4 produit de nibble

    return acc;
}

int main()
{
    srand(time(NULL));

    float dx0, dx1, dy, m0, m1;
    __m256i bx = _mm256_setzero_si256();
    __m256i by = _mm256_setzero_si256();
    int i, j;

    // Test loop
    for (i = 0; i < N_TEST; i++)
    {   
        // Compute test variables
        dx0 = random_fp32(MIN_FP32, MAX_FP32);
        dx1 = random_fp32(MIN_FP32, MAX_FP32);
        dy = random_fp32(MIN_FP32, MAX_FP32);
        m0 = random_fp32(MIN_FP32, MAX_FP32);
        m1 = random_fp32(MIN_FP32, MAX_FP32);
        _mm256_randomize_epi8(&bx, 0, 16);
        _mm256_randomize_epi8(&by, -127, 128);

        // Iter loop
        for (j = 0; j < ITER_PER_TEST; j++) {
            // vec_dot_sqr_body(dx0, dx1, dy, m0, m1, bx, by);
            vec_dot_body(dx0, dx1, &dy, m0, m1, bx, by);
        }
    }

    printf("Run finished with %d iterations.\n", N_TEST * ITER_PER_TEST);
    return 0;
}
