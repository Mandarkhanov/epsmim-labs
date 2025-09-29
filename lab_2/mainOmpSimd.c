#include<stdio.h>
#include<math.h>
#include<malloc.h>
#include<time.h>


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
    float *phi = (float*)malloc(2 * N_x * N_y * sizeof(float));
    float *phi_new = phi + N_x * N_y;

    for (int i = 0; i < N_y * 2; i++) {
      for (int j = 0; j < N_x; j++) {
          phi[i*N_y + j] = 0.0f;
      }
  }

    float h_x = (X_b - X_a) / (N_x - 1);
    float h_y = (Y_b - Y_a) / (N_y - 1);

    float mainKoef = 0.2 / (1.0 / (h_x * h_x) + 1.0 / (h_y * h_y));
    float firstKoef = 2.5 / (h_x * h_x) - 0.5 / (h_y * h_y);
    float secondKoef = 2.5 / (h_y * h_y) - 0.5 / (h_x * h_x);
    float thirdKoef = 0.25 / (h_x * h_x) + 0.25 / (h_y * h_y);

    int index;
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
            index = i * N_x;
            
            #pragma omp simd reduction(max: stepDelta)
            for (int j = 1; j < N_x - 1; j++) {
                index++;

                phi_new[index] = mainKoef * (firstKoef * (phi[index - 1] + phi[index + 1]) +
                                            secondKoef * (phi[index - N_x] + phi[index + N_x]) +
                                            thirdKoef * (phi[index - N_x - 1] + phi[index - N_x + 1] + phi[index + N_x - 1] + phi[index + N_x + 1]) +
                                            2.0 * rho[index] +
                                            (rho[index - N_x] + rho[index + N_x] + rho[index - 1] + rho[index + 1]) * 0.25);
                
                d = fabs(phi[index] - phi_new[index]);
                if (d > stepDelta) stepDelta = d;
            }
        }

        if ((stepDelta - globalDelta) < 0.0000001) {
            globalDelta = stepDelta;
            // printf("[%d]globalDelta = %f\n", iterNumber, globalDelta);
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
    
    fillFile(phi, "phi_omp_simd.dat");

    if(iterNumber % 2 == 1) swapFloatPointers(&phi, &phi_new);
    free(phi);
}

int main() {
    float *rho = (float*)malloc(N_x * N_y * sizeof(float));
    initRho(rho);
    fillFile(rho, "rho_omp_simd.dat");

    runJacobyMethod(rho);

    free(rho);
    return 0;
}