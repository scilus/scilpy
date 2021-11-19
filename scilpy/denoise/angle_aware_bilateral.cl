/*
OpenCL kernel code for computing angle-aware bilateral filtering.
*/

// Compiler definitions with placeholder values
#define IN_N_COEFFS 0
#define OUT_N_COEFFS 0
#define N_DIRS 0
#define SIGMA_RANGE 0
#define IM_X_DIM 0
#define IM_Y_DIM 0
#define IM_Z_DIM 0
#define H_X_DIM 0
#define H_Y_DIM 0
#define H_Z_DIM 0

int get_flat_index(const int x, const int y,
                   const int z, const int w,
                   const int xLen,
                   const int yLen,
                   const int zLen)
{
    return x +
           y * xLen +
           z * xLen * yLen +
           w * xLen * yLen * zLen;
}

float sf_for_direction(const int idx, const int idy, const int idz,
                       const int dir_id, global const float* sh_buffer,
                       global const float* sh_to_sf_mat)
{
    float sf_coeff = 0.0f;
    for(int i = 0; i < IN_N_COEFFS; ++i)
    {
        const int im_index = get_flat_index(idx, idy, idz, i,
                                            IM_X_DIM + H_X_DIM - 1,
                                            IM_Y_DIM + H_Y_DIM - 1,
                                            IM_Z_DIM + H_Z_DIM - 1);
        const float ylmu = sh_to_sf_mat[get_flat_index(i, dir_id, 0, 0,
                                                        IN_N_COEFFS,
                                                        N_DIRS, 0)];
        sf_coeff += sh_buffer[im_index] * ylmu;
    }
    return sf_coeff;
}

void sf_to_sh(const float* sf_coeffs, global const float* sf_to_sh_mat,
              float* sh_coeffs)
{
    // although vector operations are supported
    // in OpenCL, maximum n value is of 16.
    for(int i = 0; i < OUT_N_COEFFS; ++i)
    {
        sh_coeffs[i] = 0.0f;
        for(int u = 0; u < N_DIRS; ++u)
        {
            const float ylmu = sf_to_sh_mat[get_flat_index(u, i, 0, 0,
                                                           N_DIRS,
                                                           OUT_N_COEFFS,
                                                           0)];
            sh_coeffs[i] += sf_coeffs[u] * ylmu;
        }
    }
}

float range_distribution(const float iVal, const float jVal)
{
    const float x = fabs(iVal - jVal);
    return exp(-pow(x, 2)/2.0f/pown((float)SIGMA_RANGE, 2))
        / SIGMA_RANGE / sqrt(2.0f * M_PI);
}

__kernel void correlate(__global const float *sh_buffer,
                        __global const float *h_weights,
                        __global const float *sh_to_sf_mat,
                        __global const float *sf_to_sh_mat,
                        __global float *out_sh_buffer)
{
    const int idx = get_global_id(0);
    const int idy = get_global_id(1);
    const int idz = get_global_id(2);

    float sf_coeffs[N_DIRS];
    float norm_w;
    for(int u = 0; u < N_DIRS; ++u)
    {
        sf_coeffs[u] = 0.0f;
        const float sf_center = sf_for_direction(idx + H_X_DIM / 2,
                                                 idy + H_Y_DIM / 2,
                                                 idz + H_Z_DIM / 2,
                                                 u, sh_buffer,
                                                 sh_to_sf_mat);
        norm_w = 0.0f;
        for(int hx = 0; hx < H_X_DIM; ++hx)
        {
            for(int hy = 0; hy < H_Y_DIM; ++hy)
            {
                for(int hz = 0; hz < H_Z_DIM; ++hz)
                {
                    const float sf_u =
                        sf_for_direction(idx + hx, idy + hy,
                                         idz + hz, u, sh_buffer,
                                         sh_to_sf_mat);

                    const float range_w =
                        range_distribution(sf_center, sf_u);

                    const float weight =
                        h_weights[get_flat_index(hx, hy, hz, u,
                                                 H_X_DIM, H_Y_DIM,
                                                 H_Z_DIM)] * range_w;

                    sf_coeffs[u] += sf_u * weight;
                    norm_w += weight;
                }
            }
        }
        sf_coeffs[u] /= norm_w;
    }

    float sh_coeffs[OUT_N_COEFFS];
    sf_to_sh(sf_coeffs, sf_to_sh_mat, sh_coeffs);
    for(int i = 0; i < OUT_N_COEFFS; ++i)
    {
        const int out_index = get_flat_index(idx, idy, idz, i,
                                             IM_X_DIM, IM_Y_DIM,
                                             IM_Z_DIM);
        out_sh_buffer[out_index] = sh_coeffs[i];
    }
}
