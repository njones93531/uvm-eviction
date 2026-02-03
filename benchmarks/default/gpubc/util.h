#ifndef UTIL_H__
#define UTIL_H__

#include <xmmintrin.h>
#include <immintrin.h>
#include <emmintrin.h>

#ifdef __MIC__
#define BitCount32(x) _mm_countbits_32(x)
#else
//#ifdef USE_SSE_42
//#define BitCount32(x) _mm_popcnt_u32(x)
//#else
inline int BitCount32(int x)
{
    int i;
    int res = 0;
    for(i = 0; i < 32; i++) {
        int mask = 1 << i;
        if (x & mask) {
            res ++;
        }
    }
    return res;
}
//#endif
#endif

/*
inline int bitCount_256(__m256 x)
{

    int result[8];
    _mm256_store_ps ((float *)result, x);

    int sum = 0;
    for (int i = 0; i < 8; ++i) {
        sum += BitCount32(result[i]);
    }

    return sum;
}
*/

inline int bitCount_128(__m128i x)
{
    int result[4];
    _mm_store_si128 ((__m128i *)result, x);

    int sum = 0;
    for (int i = 0; i < 4; ++i) {
        sum += BitCount32(result[i]);
    }
    
    return sum;
}



#endif

