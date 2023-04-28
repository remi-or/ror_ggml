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