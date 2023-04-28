// gcc -Wall -march=native -lm -mavx -march=native test_file.c && ./a.out 

#include "test_functions.c"


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
    __m256i x256i;
    __m512i x512i;


    // TEST : Check the behavior of 16*int8 -> 8*int16 + squarring
    // First with arange numbers
    printf("Square test: arange\n");
    _mm256i_arange_int8(&x256i);
    x512i = _mm512_cvtepi8_epi16(x256i); 
    x512i = _mm512_mullo_epi16(x512i, x512i); // we only keep low bits but that's ok since y is an int8 (check it works though)
    _mm512i_print_int16(&x512i);
    // Now with random values
    printf("\nSquare test: random\n");
    _mm256i_rand_int8(&x256i);
    _mm256i_print_int8(&x256i);
    x512i = _mm512_cvtepi8_epi16(x256i); 
    x512i = _mm512_mullo_epi16(x512i, x512i); // we only keep low bits but that's ok since y is an int8 (check it works though)
    _mm512i_print_int16(&x512i);


    // TEST : multiply _m512i storing epi16 by shifting bits works
    printf("\nDouble test: arange\n");
    _mm256i_arange_int8(&x256i);
    x512i = _mm512_cvtepi8_epi16(x256i); 
    x512i = _mm512_slli_epi16(x512i, 1); // we only keep low bits but that's ok since y is an int8 (check it works though)
    _mm512i_print_int16(&x512i);
    // Now with random values
    printf("\nDouble test: random\n");
    _mm256i_rand_int8(&x256i);
    _mm256i_print_int8(&x256i);
    x512i = _mm512_cvtepi8_epi16(x256i); 
    x512i = _mm512_slli_epi16(x512i, 1); // we only keep low bits but that's ok since y is an int8 (check it works though)
    _mm512i_print_int16(&x512i);

    // TEST : broadcast _m512i into two _m256i
    printf("\nBroadcast test 512 -> 256 | 256\n");
    _mm512_randomize_epi32(&x512i, -1000, 1000);
    _mm512i_print_int32(&x512i);
    x256i = _mm512_castsi512_si256(x512i);
    _mm256i_print_int32(&x256i);

    printf("\nDone with testing.\n");
}