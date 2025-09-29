#include <stdio.h>
#include <math.h>
#include <malloc.h>
#include <time.h>
#include <stdlib.h>
#include <xmmintrin.h>



float X_a = 0.0;
float X_b = 4.0;
float Y_a = 0.0;
float Y_b = 4.0;

int N_x = 8000;
int N_y = 8000;
int N_t = 100;

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

void runJacobyMethod (float *rho) {
    float *phi;
    phi = (float*)malloc(2 * N_x * N_y * sizeof(float));
    float *phi_new = phi + N_x * N_y;

    for (int i = 0; i < N_y * 2; i++) {
        for (int j = 0; j < N_x; j++) {
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

    int step1vector = 4;
    int countVecInLine = (N_x - 2) / 4;
    int remainsInLine = (N_x - 2) % 4;

        long long t1, t2;
        double tDiff;
        struct timespec curTime;
        clock_gettime(CLOCK_BOOTTIME, &curTime);
        t1 = curTime.tv_sec * 1000000000 + curTime.tv_nsec;

    while (iterNumber <= N_t) {
        stepDelta = -1.0;
        
        for (int i = 1; i < N_y - 1; i++) {
            index = i * N_x;

            for (int j = 1; j < countVecInLine * step1vector + 1; j += step1vector) {
                vec_index = index  + j;
                
                __m128 v_phi_left     = _mm_loadu_ps(&phi[vec_index - 1]);
                __m128 v_phi_right    = _mm_loadu_ps(&phi[vec_index + 1]);
                __m128 v_phi_bottom   = _mm_loadu_ps(&phi[vec_index - N_x]);
                __m128 v_phi_top      = _mm_loadu_ps(&phi[vec_index + N_x]);
                __m128 v_phi_bot_left = _mm_loadu_ps(&phi[vec_index - N_x - 1]);
                __m128 v_phi_bot_right= _mm_loadu_ps(&phi[vec_index - N_x + 1]);
                __m128 v_phi_top_left = _mm_loadu_ps(&phi[vec_index + N_x - 1]);
                __m128 v_phi_top_right= _mm_loadu_ps(&phi[vec_index + N_x + 1]);
                
                __m128 v_rho_center   = _mm_loadu_ps(&rho[vec_index]);
                __m128 v_rho_bottom   = _mm_loadu_ps(&rho[vec_index - N_x]);
                __m128 v_rho_top      = _mm_loadu_ps(&rho[vec_index + N_x]);
                __m128 v_rho_left     = _mm_loadu_ps(&rho[vec_index - 1]);
                __m128 v_rho_right    = _mm_loadu_ps(&rho[vec_index + 1]);

                __m128 first_line    = _mm_add_ps(_mm_mul_ps(firstKoef_m128, _mm_add_ps(v_phi_left, v_phi_right)),
                                                  _mm_mul_ps(secondKoef_m128, _mm_add_ps(v_phi_bottom, v_phi_top)));
                __m128 second_line   = _mm_mul_ps(thirdKoef_m128,
                                                  _mm_add_ps(_mm_add_ps(v_phi_bot_left, v_phi_top_left),
                                                             _mm_add_ps(v_phi_bot_right, v_phi_top_right)));
                __m128 third_line    = _mm_add_ps(_mm_mul_ps(_mm_set1_ps(0.25f),
                                                             _mm_add_ps(_mm_add_ps(v_rho_bottom, v_rho_top),
                                                                        _mm_add_ps(v_rho_left, v_rho_right))),
                                                  _mm_mul_ps(_mm_set1_ps(2.0f), v_rho_center));
                __m128 result        = _mm_mul_ps(mainKoef_m128,
                                                  _mm_add_ps(first_line, _mm_add_ps(second_line, third_line)));

                _mm_storeu_ps(&phi_new[vec_index], result);
                
                for (int di = 0; di < step1vector; di++) {
                    d = fabs(phi[vec_index + di] - phi_new[vec_index + di]);
                    if (d > stepDelta) stepDelta = d;
                }
            }
        }

        int N_x_remains_index = countVecInLine * step1vector + 1;
        for (int i = 1; i < N_y - 1; i++) {
            index = i * N_x;
            for (int j = N_x_remains_index; j < N_x - 1; j++) {
                index++;

                phi_new[index] = mainKoef * (firstKoef * (phi[index - 1] + phi[index + 1]) +
                                            secondKoef * (phi[index - N_x] + phi[index + N_x]) +
                                            thirdKoef * (phi[index - N_x - 1] + phi[index - N_x + 1] + phi[index + N_x - 1] + phi[index + N_x + 1]) +
                                            2.0f * rho[index] +
                                            (rho[index - N_x] + rho[index + N_x] + rho[index - 1] + rho[index + 1]) * 0.25f);
                
                d = fabs(phi[index] - phi_new[index]);
                if (d > stepDelta) stepDelta = d;
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

    // fillFile(phi, "phi_unalign.dat");

    if(iterNumber % 2 == 1) swapFloatPointers(&phi, &phi_new);
    free(phi);
}

int main() {
    float *rho = (float*)malloc(N_x * N_y * sizeof(float));
    initRho(rho);
    // fillFile(rho, "rho_unalign.dat");

    runJacobyMethod(rho);
    
    free(rho);
    return 0;
}