/************************
 * author: SharEDITor
 * date:   2016-08-02
 * brief:  read MNIST data
 ************************/
#include <stdio.h>
#include <stdint.h>
#include <assert.h>
#include <stdlib.h>

unsigned char *lables = NULL;

/**
 * All the integers in the files are stored in the MSB first (high endian) format
 */
void copy_int(uint32_t *target, unsigned char *src)
{
    *(((unsigned char*)target)+0) = src[3];
    *(((unsigned char*)target)+1) = src[2];
    *(((unsigned char*)target)+2) = src[1];
    *(((unsigned char*)target)+3) = src[0];
}

int read_lables()
{
    FILE *fp = fopen("./train-labels-idx1-ubyte", "r");
    if (NULL == fp)
    {
        return -1;
    }
    unsigned char head[8];
    fread(head, sizeof(unsigned char), 8, fp);
    uint32_t magic_number = 0;
    uint32_t item_num = 0;
    copy_int(&magic_number, &head[0]);
    // magic number check
    assert(magic_number == 2049);
    copy_int(&item_num, &head[4]);

    uint64_t values_size = sizeof(unsigned char) * item_num;
    lables = (unsigned char*)malloc(values_size);
    fread(lables, sizeof(unsigned char), values_size, fp);

    fclose(fp);
    return 0;
}

int read_images()
{
    FILE *fp = fopen("./train-images-idx3-ubyte", "r");
    if (NULL == fp)
    {
        return -1;
    }
    unsigned char head[16];
    fread(head, sizeof(unsigned char), 16, fp);
    uint32_t magic_number = 0;
    uint32_t images_num = 0;
    uint32_t rows = 0;
    uint32_t cols = 0;
    copy_int(&magic_number, &head[0]);
    // magic number check
    assert(magic_number == 2051);
    copy_int(&images_num, &head[4]);
    copy_int(&rows, &head[8]);
    copy_int(&cols, &head[12]);

    printf("rows=%d cols=%d\n", rows, cols);

    uint64_t image_size = rows * cols;
    uint64_t values_size = sizeof(unsigned char) * images_num * rows * cols;
    unsigned char *values = (unsigned char*)malloc(values_size);
    fread(values, sizeof(unsigned char), values_size, fp);

    for (int image_index = 0; image_index < images_num; image_index++)
    {
        // print the label
        printf("=========================================  %d  ======================================\n", lables[image_index]);
        for (int row_index = 0; row_index < rows; row_index++)
        {
            for (int col_index = 0; col_index < cols; col_index++)
            {
                // print the pixels of image
                printf("%3d", values[image_index*image_size+row_index*cols+col_index]);
            }
            printf("\n");
        }
        printf("\n");
    }

    free(values);
    fclose(fp);
    return 0;
}

int main(int argc, char *argv[])
{
    if (-1 == read_lables())
    {
        return -1;
    }
    if (-1 == read_images())
    {
        return -1;
    }
    return 0;
}
