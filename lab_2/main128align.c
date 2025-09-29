#include <stdio.h>
#include <math.h>
#include <malloc.h>
#include <time.h>
#include <stdlib.h>
#include <smmintrin.h>

#define VEC_SIZE 4
#define ANY 0
#define MASK_ABBB 0b1110
#define MASK_AAAB 0b1000

float X_a = 0.0;
float X_b = 4.0;
float Y_a = 0.0;
float Y_b = 4.0;

int N_x = 8000;
int N_y = 8000;
int N_t = 128;

void swapFloatPointers(float** a, float** b) {
    float* tmp = *a;
    *a = *b;
    *b = tmp;
}

void fillFile(float* matrix, char* filename) {
    FILE *fp;
    fp = fopen(filename, "wb");

    if (fp == NULL) {
        printf("Error opening file!\n");
        return;
    }

    int index;
    for (int i = 0; i < N_y; i++) {
        index = i * N_x;
        for (int j = 0; j < N_x; j++) {
            fprintf(fp, "%f ", matrix[index]);
            index++;
        }
        fprintf(fp, "\n");
    }
    fclose(fp);
}

void initRho(float* rho) {
    float h_x = (X_b - X_a) / (N_x - 1);
    float h_y = (Y_b - Y_a) / (N_y - 1);
    
    float X_s1 = X_a + (X_b - X_a) / 3.0;
    float Y_s1 = Y_a + (Y_b - Y_a) * 2.0 / 3.0;
    float X_s2 = X_a + (X_b - X_a) * 2.0 / 3.0;
    float Y_s2 = Y_a + (Y_b - Y_a) / 3.0;

    float R = 0.1 * fmin(X_b - X_a, Y_b - Y_a);

    int index;

    for (float i = 0; i < N_y; i++) {
        index = i * N_x;
        for (float j = 0; j < N_x; j++) {
    
            if ((X_a + j * h_x - X_s1) * (X_a + j * h_x - X_s1) + (Y_a + i * h_y - Y_s1) * (Y_a + i * h_y - Y_s1) < R * R) {
                rho[index] = 1.0;
            }
            else if ((X_a + j * h_x - X_s2) * (X_a + j * h_x - X_s2)  + (Y_a + i * h_y - Y_s2) * (Y_a + i * h_y - Y_s2) < R * R) {
                rho[index] = -1.0;
            }
            else {
                rho[index] = 0.0;
            }

            index ++;
        }
    }
}

void runJacobyMethod (float* rho) {
    int countVecInLine = (N_x - 2) / VEC_SIZE;
    int remainsInLine = (N_x - 2) % VEC_SIZE;
    int N_x_align = VEC_SIZE + (countVecInLine * VEC_SIZE) + VEC_SIZE; // left VEC_SIZE for border
                                                                       // right VEC_SIZE for remains and border

    float* phi;
    if (posix_memalign((void**)&phi, 16, 2*N_x_align*N_y*sizeof(float)) != 0) exit(1);
    float* phi_new = phi + N_x * N_y;

    for (int i = 0; i < N_y * 2; i++) {
        for (int j = 0; j < N_x_align; j++) {
            phi[i*N_y + j] = 0.0f;
        }
    }

    float h_x = (X_b - X_a) / (N_x - 1);
    float h_y = (Y_b - Y_a) / (N_y - 1);

    float mainKoef = 0.2 / (1.0 / (h_x * h_x) + 1.0 / (h_y * h_y));     __m128 mainKoef_m128  = _mm_set1_ps(mainKoef);
    float firstKoef = 2.5 / (h_x * h_x) - 0.5 / (h_y * h_y);            __m128 firstKoef_m128 = _mm_set1_ps(firstKoef);
    float secondKoef = 2.5 / (h_y * h_y) - 0.5 / (h_x * h_x);           __m128 secondKoef_m128= _mm_set1_ps(secondKoef);
    float thirdKoef = 0.25 / (h_x * h_x) + 0.25 / (h_y * h_y);          __m128 thirdKoef_m128 = _mm_set1_ps(thirdKoef);

    int index, vec_index;
    float d, stepDelta;
    float globalDelta = 1.0;
    int iterNumber = 0;

        long long t1, t2;
        double tDiff;
        struct timespec curTime;
        clock_gettime(CLOCK_BOOTTIME, &curTime);
        t1 = curTime.tv_sec * 1000000000 + curTime.tv_nsec;

    while (iterNumber <= N_t) {
        stepDelta = -1.0;

        for (int i = 1; i < N_y - 1; i++) {
            index = i * N_x + 4;

            for (int j = 0; j < countVecInLine * VEC_SIZE + 1; j += VEC_SIZE) {
                vec_index = index + j;
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
                
                __m128 v_phi_right_top = _mm_blend_ps(_mm_shuffle_ps(av_phi_top      , av_phi_top      , _MM_SHUFFLE(ANY, 3  , 2  , 1  )),
                                                      _mm_shuffle_ps(av_phi_right_top, av_phi_right_top, _MM_SHUFFLE(0  , ANY, ANY, ANY)),
                                                      MASK_AAAB);
                __m128 v_phi_right     = _mm_blend_ps(_mm_shuffle_ps(av_phi_cnt      , av_phi_cnt      , _MM_SHUFFLE(ANY, 3  , 2  , 1  )),
                                                      _mm_shuffle_ps(av_phi_right    , av_phi_right    , _MM_SHUFFLE(0  , ANY, ANY, ANY)),
                                                      MASK_AAAB);
                __m128 v_phi_right_bot = _mm_blend_ps(_mm_shuffle_ps(av_phi_bot      , av_phi_bot      , _MM_SHUFFLE(ANY, 3  , 2  , 1  )),
                                                      _mm_shuffle_ps(av_phi_right_bot, av_phi_right_bot, _MM_SHUFFLE(0  , ANY, ANY, ANY)),
                                                      MASK_AAAB);

                __m128 av_rho_cnt      = _mm_load_ps(&rho[vec_index]);
                __m128 av_rho_bot      = _mm_load_ps(&rho[vec_index - N_x]);
                __m128 av_rho_top      = _mm_load_ps(&rho[vec_index + N_x]);
                __m128 av_rho_left     = _mm_load_ps(&rho[vec_index - 4]);
                __m128 av_rho_right    = _mm_load_ps(&rho[vec_index + 4]);
                
                __m128 v_rho_left      = _mm_blend_ps(_mm_shuffle_ps(av_rho_left     , av_rho_left     , _MM_SHUFFLE(ANY, ANY, ANY, 3  )),
                                                      _mm_shuffle_ps(av_rho_cnt      , av_rho_cnt      , _MM_SHUFFLE(2  , 1  , 0  , ANY)),
                                                      MASK_ABBB);
                __m128 v_rho_right     = _mm_blend_ps(_mm_shuffle_ps(av_rho_cnt      , av_rho_cnt      , _MM_SHUFFLE(ANY, 3  , 2  , 1  )),
                                                      _mm_shuffle_ps(av_rho_right    , av_rho_right    , _MM_SHUFFLE(0  , ANY, ANY, ANY)),
                                                      MASK_AAAB);

                __m128 first_line   = _mm_add_ps(_mm_mul_ps(firstKoef_m128, _mm_add_ps(v_phi_left, v_phi_right)),
                                                 _mm_mul_ps(secondKoef_m128, _mm_add_ps(av_phi_bot, av_phi_top)));
                __m128 second_line  = _mm_mul_ps(thirdKoef_m128,
                                                 _mm_add_ps(_mm_add_ps(v_phi_left_bot, v_phi_left_top),
                                                            _mm_add_ps(v_phi_right_bot, v_phi_right_top)));
                __m128 third_line   = _mm_add_ps(_mm_mul_ps(_mm_set1_ps(0.25f),
                                                            _mm_add_ps(_mm_add_ps(av_rho_bot, av_rho_top),
                                                                       _mm_add_ps(v_rho_left, v_rho_right))),
                                                 _mm_mul_ps(_mm_set1_ps(2.0f), av_rho_cnt));
                __m128 result       = _mm_mul_ps(mainKoef_m128,
                                                  _mm_add_ps(first_line, _mm_add_ps(second_line, third_line)));

                _mm_store_ps(&phi_new[vec_index], result);
                
                for (int di = 0; di < VEC_SIZE; di++) {
                    d = fabs(phi[vec_index + di] - phi_new[vec_index + di]);
                    if (d > stepDelta) stepDelta = d;
                }
            }
        }

        int N_x_remains_index =  VEC_SIZE + countVecInLine * VEC_SIZE + 1;
        for (int i = 1; i < N_y - 1; i++) {
            index = i * N_x + N_x_remains_index;
            for (int j = N_x_remains_index; j < N_x - 1; j++) {
                phi_new[index] = mainKoef * (firstKoef * (phi[index - 1] + phi[index + 1]) +
                                            secondKoef * (phi[index - N_x] + phi[index + N_x]) +
                                            thirdKoef * (phi[index - N_x - 1] + phi[index - N_x + 1] + phi[index + N_x - 1] + phi[index + N_x + 1]) +
                                            2.0f * rho[index] +
                                            (rho[index - N_x] + rho[index + N_x] + rho[index - 1] + rho[index + 1]) * 0.25f);
                
                d = fabs(phi[index] - phi_new[index]);
                if (d > stepDelta) stepDelta = d;
                index++;
            }
        }


        if ((stepDelta - globalDelta) < 0.0000001) {
            globalDelta = stepDelta;
            swapFloatPointers(&phi, &phi_new);
            iterNumber++;
        }
        else {
            printf("Delta is growwing!\nJacoby method stopped\n");
            break;
        }
    }

        clock_gettime(CLOCK_BOOTTIME, &curTime);
        t2 = curTime.tv_sec * 1000000000 + curTime.tv_nsec;
        tDiff = (double) (t2 - t1) / 1000000000.0;
        printf("Time = %g s\n", tDiff);

    // fillFile(phi, "phi_align.dat");

    if(iterNumber % 2 == 1) swapFloatPointers(&phi, &phi_new);
    free(phi);
}

int main() {
    int countVecInLine = (N_x - 2) / VEC_SIZE;
    int remainsInLine = (N_x - 2) % VEC_SIZE;
    int N_x_align = VEC_SIZE + (countVecInLine * VEC_SIZE) + VEC_SIZE;

    float* rho;
    if (posix_memalign((void**)&rho, 16, N_x_align*N_y*sizeof(float)) != 0) exit(1);
    initRho(rho);
    // fillFile(rho, "rho_align.dat");

    runJacobyMethod(rho);
    
    free(rho);
    return 0;
}