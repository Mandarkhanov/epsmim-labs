#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <smmintrin.h>

#define N_x 12
#define N_y 3
#define ANY 0
#define MASK_ABBB 0b1110
#define MASK_AAAB 0b1000
#define VEC_SIZE 4

int main() {
    float *phi;
    if (posix_memalign((void**)&phi, 16, N_x*N_y*sizeof(float)) != 0) exit(1);
    
    // // Выводим все адреса элементов массива
    // for (int i = 0; i < N_x * N_y; i++) {
    //     void *address = (void*)&rho[i];
    //     printf("Address of rho[%d] = %p, divisible by 16: %s\n", 
    //            i, 
    //            address, 
    //            (((uintptr_t)address & 0xF) == 0) ? "Yes" : "No");
    // }

    for (int i = 0; i < N_y; i++) {
        for (int j = 0; j < N_x; j++) {
            phi[i*N_x + j] = i*N_x+j;
            printf("%f", phi[i*N_x + j]);
            printf(" | ");
        }
        printf("\n");
    }
    printf("\n");

    int index, vec_index;
    vec_index = 16;

    __m128 av_phi_left      = _mm_load_ps(&phi[vec_index - 4]);
    __m128 av_phi_left_bot  = _mm_load_ps(&phi[vec_index - 4 - N_x]);
    __m128 av_phi_left_top  = _mm_load_ps(&phi[vec_index - 4 + N_x]);

    __m128 av_phi_right     = _mm_load_ps(&phi[vec_index + 4]);
    __m128 av_phi_right_bot = _mm_load_ps(&phi[vec_index + 4 - N_x]);
    __m128 av_phi_right_top = _mm_load_ps(&phi[vec_index + 4 + N_x]);

    __m128 av_phi_cnt       = _mm_load_ps(&phi[vec_index]);
    __m128 av_phi_bot       = _mm_load_ps(&phi[vec_index - N_x]);
    __m128 av_phi_top       = _mm_load_ps(&phi[vec_index + N_x]);

    __m128 v_phi_left_top  = _mm_blend_ps(_mm_shuffle_ps(av_phi_left_top , av_phi_left_top , _MM_SHUFFLE(ANY, ANY, ANY, 3  )),
                                            _mm_shuffle_ps(av_phi_top      , av_phi_top      , _MM_SHUFFLE(2  , 1  , 0  , ANY)),
                                            MASK_ABBB);
    __m128 v_phi_left      = _mm_blend_ps(_mm_shuffle_ps(av_phi_left     , av_phi_left     , _MM_SHUFFLE(ANY, ANY, ANY, 3  )),
                                            _mm_shuffle_ps(av_phi_cnt      , av_phi_cnt      , _MM_SHUFFLE(2  , 1  , 0  , ANY)),
                                            MASK_ABBB);
    __m128 v_phi_left_bot  = _mm_blend_ps(_mm_shuffle_ps(av_phi_left_bot , av_phi_left_bot , _MM_SHUFFLE(ANY, ANY, ANY, 3  )),
                                            _mm_shuffle_ps(av_phi_bot      , av_phi_bot      , _MM_SHUFFLE(2  , 1  , 0  , ANY)),
                                            MASK_ABBB);
    
    __m128 v_phi_right_top = _mm_blend_ps(_mm_shuffle_ps(av_phi_top, av_phi_top, _MM_SHUFFLE(ANY, 3  , 2  , 1  )),
                                            _mm_shuffle_ps(av_phi_right_top      , av_phi_right_top      , _MM_SHUFFLE(0  , ANY, ANY, ANY)),
                                            MASK_AAAB);
    __m128 v_phi_right     = _mm_blend_ps(_mm_shuffle_ps(av_phi_cnt      , av_phi_cnt      , _MM_SHUFFLE(ANY, 3  , 2  , 1  )),
                                            _mm_shuffle_ps(av_phi_right    , av_phi_right    , _MM_SHUFFLE(0  , ANY, ANY, ANY)),
                                            MASK_AAAB);
    __m128 v_phi_right_bot = _mm_blend_ps(_mm_shuffle_ps(av_phi_bot, av_phi_bot, _MM_SHUFFLE(ANY, 3  , 2  , 1  )),
                                            _mm_shuffle_ps(av_phi_right_bot     ,  av_phi_right_bot     , _MM_SHUFFLE(0  , ANY, ANY, ANY)),
                                            MASK_AAAB);

    
    _mm_store_ps(&phi[vec_index - 4]      , v_phi_left);
    _mm_store_ps(&phi[vec_index - 4 - N_x], v_phi_left_bot);
    _mm_store_ps(&phi[vec_index - 4 + N_x], v_phi_left_top);
    _mm_store_ps(&phi[vec_index + 4]      , v_phi_right);
    _mm_store_ps(&phi[vec_index + 4 - N_x], v_phi_right_bot);
    _mm_store_ps(&phi[vec_index + 4 + N_x], v_phi_right_top);


    for (int i = 0; i < N_y; i++) {
        for (int j = 0; j < N_x; j++) {
            printf("%f", phi[i*N_x + j]);
            printf(" | ");
        }
        printf("\n");
    }

    free(phi);
}