#include <stdio.h>
#include <stdlib.h>
#include <malloc.h>
#include <stdint.h>
#include <time.h>
#include <math.h>
#include <sys/time.h>
#include <omp.h>

#include "mpi.h"

#define IMAGESIZE 54
// Macro for checking errors in CUDA API calls
#define cudaErrorCheck(call)                                                                     \
    do                                                                                           \
    {                                                                                            \
        cudaError_t cuErr = call;                                                                \
        if (cudaSuccess != cuErr)                                                                \
        {                                                                                        \
            printf("CUDA Error - %s:%d: '%s'\n", __FILE__, __LINE__, cudaGetErrorString(cuErr)); \
            exit(0);                                                                             \
        }                                                                                        \
    } while (0)

#pragma pack(push, 2)
typedef struct
{
    char sign;
    int size;
    int notused;
    int data;
    int headwidth;
    int width;
    int height;
    short numofplanes;
    short bitpix;
    int method;
    int arraywidth;
    int horizresol;
    int vertresol;
    int colnum;
    int basecolnum;
} img;
#pragma pop

unsigned char *openImg(int inputFileNumber, img *bmp);
void generateImg(unsigned char *imgdata, img *bmp);
int setBoundary(int i, int min, int max);

__global__ void perform(int nStart, int nStop, int width, int height, int radius, unsigned char *red, unsigned char *green, unsigned char *blue)
{
    printf("HERE");
    int t = threadIdx.x;
    for (int i = nStart; i < nStop; i++)
    {
        for (int j = 0; j < width; j++)
        {
            double row;
            double col;
            double redSum = 0;
            double greenSum = 0;
            double blueSum = 0;
            double weightSum = 0;
            for (row = i - radius; row <= i + radius; row++)
            {
                for (col = j - radius; col <= j + radius; col++)
                {
                    int x = col;
                    if (x < 0)
                        x = 0;
                    else if (x > width - 1)
                        x = width - 1;
                    int y = row;
                    if (y < 0)
                        y = 0;
                    else if (y > height - 1)
                        y = height - 1;
                    int tempPos = y * width + x;
                    double square = (col - j) * (col - j) + (row - i) * (row - i);
                    double sigma = radius * radius;
                    double weight = exp(-square / (2 * sigma)) / (3.14 * 2 * sigma);
                    redSum += red[tempPos] * weight;
                    greenSum += green[tempPos] * weight;
                    blueSum += blue[tempPos] * weight;
                    weightSum += weight;
                }
            }
            red[i * width + j] = round(redSum / weightSum);
            green[i * width + j] = round(greenSum / weightSum);
            blue[i * width + j] = round(blueSum / weightSum);
            redSum = 0;
            greenSum = 0;
            blueSum = 0;
            weightSum = 0;
        }
    }
}

int main(int argc, char *argv[])
{
    unsigned char *imgdata;
    img *bmp = (img *)malloc(IMAGESIZE);
    int radius = atoi(argv[1]);
    int inputFileNumber = atoi(argv[2]);
    imgdata = openImg(inputFileNumber, bmp);

    int width = bmp->width;
    int height = bmp->height;
    int SIZE = width * height * sizeof(unsigned char);

    int i, j;
    int rgb_width = width * 3;
    if ((width * 3 % 4) != 0)
    {
        rgb_width += (4 - (width * 3 % 4));
    }

    unsigned char *red;
    unsigned char *green;
    unsigned char *blue;
    cudaErrorCheck(cudaMallocManaged(&red, width * height * sizeof(unsigned char)));
    cudaErrorCheck(cudaMallocManaged(&green, width * height * sizeof(unsigned char)));
    cudaErrorCheck(cudaMallocManaged(&blue, width * height * sizeof(unsigned char)));

    int pos = 0;
    for (i = 0; i < height; i++)
    {
        for (j = 0; j < width * 3; j += 3, pos++)
        {
            red[pos] = imgdata[i * rgb_width + j];
            green[pos] = imgdata[i * rgb_width + j + 1];
            blue[pos] = imgdata[i * rgb_width + j + 2];
        }
    }

    struct timeval start_time, stop_time, elapsed_time;
    gettimeofday(&start_time, NULL);

    int my_PE_num;
    int threadNumber;
    int nStart, nStop;

    MPI_Status status;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_PE_num);
    MPI_Comm_size(MPI_COMM_WORLD, &threadNumber);

    // Map MPI ranks to GPUs
    // cudaErrorCheck(cudaSetDevice(my_PE_num));
    // int deviceCount = 0;
    // cudaGetDeviceCount(&deviceCount);
    // printf("%d", deviceCount);

    int subSize = height / threadNumber;

    nStart = my_PE_num * subSize;
    nStop = (my_PE_num + 1) * subSize;

    unsigned char *redBuffer;
    unsigned char *greenBuffer;
    unsigned char *blueBuffer;
    cudaErrorCheck(cudaMallocManaged(&redBuffer, width * height * sizeof(unsigned char)));
    cudaErrorCheck(cudaMallocManaged(&greenBuffer, width * height * sizeof(unsigned char)));
    cudaErrorCheck(cudaMallocManaged(&blueBuffer, width * height * sizeof(unsigned char)));

    if (my_PE_num == 0)
    {
        int k, n_proc;
        MPI_Comm_size(MPI_COMM_WORLD, &n_proc);
        for (k = 1; k < n_proc; k++)
        {
            MPI_Recv(redBuffer, SIZE, MPI_UNSIGNED_CHAR, k, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
            MPI_Recv(greenBuffer, SIZE, MPI_UNSIGNED_CHAR, k, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
            MPI_Recv(blueBuffer, SIZE, MPI_UNSIGNED_CHAR, k, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

            int otherThreadStart = k * subSize;
            int otherThreadStop = (k + 1) * subSize;

            for (i = otherThreadStart; i < otherThreadStop; i++)
            {
                for (j = 0; j < width; j++)
                {
                    red[i * width + j] = redBuffer[i * width + j];
                    green[i * width + j] = greenBuffer[i * width + j];
                    blue[i * width + j] = blueBuffer[i * width + j];
                }
            }
        }
        perform<<<1, 10>>>(nStart, nStop, width, height, radius, red, green, blue);
        cudaErrorCheck(cudaPeekAtLastError());
        cudaErrorCheck(cudaDeviceSynchronize());
    }

    else
    {
        perform<<<1, 10>>>(nStart, nStop, width, height, radius, red, green, blue);
        cudaErrorCheck(cudaPeekAtLastError());
        cudaErrorCheck(cudaDeviceSynchronize());
        for (i = nStart; i < nStop; i++)
        {
            for (j = 0; j < width; j++)
            {
                redBuffer[i * width + j] = red[i * width + j];
                greenBuffer[i * width + j] = green[i * width + j];
                blueBuffer[i * width + j] = blue[i * width + j];
            }
        }

        MPI_Send(redBuffer, SIZE, MPI_UNSIGNED_CHAR, 0, 0, MPI_COMM_WORLD);
        MPI_Send(greenBuffer, SIZE, MPI_UNSIGNED_CHAR, 0, 0, MPI_COMM_WORLD);
        MPI_Send(blueBuffer, SIZE, MPI_UNSIGNED_CHAR, 0, 0, MPI_COMM_WORLD);
    }

    if (my_PE_num == 0)
    {
        // print elapsed time
        gettimeofday(&stop_time, NULL);
        timersub(&stop_time, &start_time, &elapsed_time);
        printf("Took %f seconds \n", elapsed_time.tv_sec + elapsed_time.tv_usec / 1000000.0);

        pos = 0;
        for (i = 0; i < height; i++)
        {
            for (j = 0; j < width * 3; j += 3, pos++)
            {
                imgdata[i * rgb_width + j] = red[pos];
                imgdata[i * rgb_width + j + 1] = green[pos];
                imgdata[i * rgb_width + j + 2] = blue[pos];
            }
        }
        generateImg(imgdata, bmp);
    }
    cudaErrorCheck(cudaFree(redBuffer));
    cudaErrorCheck(cudaFree(greenBuffer));
    cudaErrorCheck(cudaFree(blueBuffer));

    MPI_Finalize();
    cudaFree(red);
    cudaFree(green);
    cudaFree(blue);
    free(bmp);
    return 0;
}

unsigned char *openImg(int inputFileNumber, img *in)
{
    char inPutFileNameBuffer[32];
    sprintf(inPutFileNameBuffer, "%d.bmp", inputFileNumber);

    FILE *file;
    if (!(file = fopen(inPutFileNameBuffer, "rb")))
    {
        printf("File not found!");
        free(in);
        exit(1);
    }
    fread(in, 54, 1, file);

    unsigned char *data = (unsigned char *)malloc(in->arraywidth);
    fseek(file, in->data, SEEK_SET);
    fread(data, in->arraywidth, 1, file);
    fclose(file);
    return data;
}

void generateImg(unsigned char *imgdata, img *out)
{
    FILE *file;
    time_t now;
    time(&now);
    char fileNameBuffer[32];
    sprintf(fileNameBuffer, "%s.bmp", ctime(&now));
    file = fopen(fileNameBuffer, "wb");
    fwrite(out, IMAGESIZE, 1, file);
    fseek(file, out->data, SEEK_SET);
    fwrite(imgdata, out->arraywidth, 1, file);
    fclose(file);
}