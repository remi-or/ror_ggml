// g++ -mavx512f -mnative test_file.c && ./a.out 

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <immintrin.h>
#include <xmmintrin.h>

void _mm256i_print_int8(__m256i * src) {
    char * ptr = (char*) src;
    for (int i = 0; i < 32; i++) {
        printf("%d ", ptr[i]);
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


void _mm256i_arange_int8(__m256i * src) {
    int sign;
    char * ptr = (char*) src;
    for (int i = 0; i < 32; i++) {
        sign = (i%2) ? 1 : -1;
        ptr[i] = sign * i;
    }
}

void _mm256i_rand_int8(__m256i * src) {
    int sign;
    char * ptr = (char*) src;
    for (int i = 0; i < 32; i++) {
        ptr[i] = (rand() % 256) - 128;
    }
}


int main() {

    printf("Starting testing...\n");
    __m256i x;
    __m512i wide_x;


    // TEST : Check the behavior of 16*int8 -> 8*int16 + squarring
    // First with arange numbers
    printf("Square test: arange\n");
    _mm256i_arange_int8(&x);
    wide_x = _mm512_cvtepi8_epi16(x); 
    wide_x = _mm512_mullo_epi16(wide_x, wide_x); // we only keep low bits but that's ok since y is an int8 (check it works though)
    _mm512i_print_int16(&wide_x);
    // Now with random values
    printf("\nSquare test: random\n");
    _mm256i_rand_int8(&x);
    _mm256i_print_int8(&x);
    wide_x = _mm512_cvtepi8_epi16(x); 
    wide_x = _mm512_mullo_epi16(wide_x, wide_x); // we only keep low bits but that's ok since y is an int8 (check it works though)
    _mm512i_print_int16(&wide_x);


    // TEST : multiply _m512i storing epi16 by shifting bits works
    printf("\nDouble test: arange\n");
    _mm256i_arange_int8(&x);
    wide_x = _mm512_cvtepi8_epi16(x); 
    wide_x = _mm512_slli_epi16(wide_x, 1); // we only keep low bits but that's ok since y is an int8 (check it works though)
    _mm512i_print_int16(&wide_x);
    // Now with random values
    printf("\nDouble test: random\n");
    _mm256i_rand_int8(&x);
    _mm256i_print_int8(&x);
    wide_x = _mm512_cvtepi8_epi16(x); 
    wide_x = _mm512_slli_epi16(wide_x, 1); // we only keep low bits but that's ok since y is an int8 (check it works though)
    _mm512i_print_int16(&wide_x);

    // TEST : 

    printf("\nDone with testing.\n");
}