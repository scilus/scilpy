#define WIN_WIDTH 0
#define SIGMA_RANGE 1.0f
#define N_DIRS 200
#define EXCLUDE_SELF false
#define DISABLE_ANGLE false
#define DISABLE_RANGE false

int get_flat_index(const int x, const int y, const int z,
                   const int w, const int x_len,
                   const int y_len, const int z_len)
{
    return x + y * x_len + z * x_len * y_len +
           w * x_len * y_len * z_len;
}

float range_filter(const float x)
{
    return exp(-pow(x, 2)/2.0f/pown((float)SIGMA_RANGE, 2));
}

// SF data is padded while out_sf isn't
__kernel void filter(__global const float* sf_data,
                     __global const float* nx_filter,
                     __global const float* uv_filter,
                     __global const float* uv_weights_offset,
                     __global const float* v_indices,
                     __global float* out_sf)
{
    // *output* image dimensions
    const int x_len = get_global_size(0);
    const int y_len = get_global_size(1);
    const int z_len = get_global_size(2);

    // padded dimensions
    const int x_pad_len = x_len + WIN_WIDTH - 1;
    const int y_pad_len = y_len + WIN_WIDTH - 1;
    const int z_pad_len = z_len + WIN_WIDTH - 1;

    // output voxel indice
    const int x_ind = get_global_id(0);
    const int y_ind = get_global_id(1);
    const int z_ind = get_global_id(2);

    // window half width
    const int win_hwidth = WIN_WIDTH / 2;

    // process each direction inside voxel
    for(int ui = 0; ui < N_DIRS; ++ui)
    {
        // in input volume, dimensions are padded
        const int xui_flat_ind = get_flat_index(x_ind + win_hwidth,
                                                y_ind + win_hwidth,
                                                z_ind + win_hwidth,
                                                ui, x_pad_len, y_pad_len,
                                                z_pad_len);
        const float psi_xui = sf_data[xui_flat_ind];

        // output value
        float w_norm = 0.0f;
        float tilde_psi_xui = 0.0f;
#if DISABLE_ANGLE
        const int vi_in_image = ui;
#else
        const int vi_offset = (int)uv_weights_offset[ui];
        const int n_ang_neigbrs = (int)uv_weights_offset[ui + 1] - vi_offset;
        for(int dir_i = 0; dir_i < n_ang_neigbrs; ++dir_i)
        {
            const int vi_in_filter = vi_offset + dir_i;
            const int vi_in_image = v_indices[vi_in_filter];
#endif
            for(int hi = 0; hi < WIN_WIDTH; ++hi)
            {
                for(int hj = 0; hj < WIN_WIDTH; ++hj)
                {
                    for(int hk = 0; hk < WIN_WIDTH; ++hk)
                    {
                        const int yvi_flat_ind =
                            get_flat_index(x_ind + hi, y_ind + hj, z_ind + hk,
                                           vi_in_image, x_pad_len, y_pad_len,
                                           z_pad_len);
                        const float psi_yvi = sf_data[yvi_flat_ind];
#if DISABLE_RANGE
                        const float r_weight = 1.0f;
#else
                        const float r_weight = range_filter(fabs(psi_xui - psi_yvi));
#endif
                        // contains "align" weight, so direction is ui
                        const int y_in_nx_flat_ind = get_flat_index(hi, hj, hk, ui,
                                                                    WIN_WIDTH, WIN_WIDTH,
                                                                    WIN_WIDTH);
                        const float nx_weight = nx_filter[y_in_nx_flat_ind];

#if DISABLE_ANGLE
                        const float uv_weight = 1.0f;
#else
                        const float uv_weight = uv_filter[vi_in_filter];
#endif

                        const float res_weight_yvi = nx_weight * r_weight * uv_weight;
                        tilde_psi_xui += res_weight_yvi * psi_yvi;
                        w_norm += res_weight_yvi;
                    }
                }
            }
#if !DISABLE_ANGLE
        }
#endif
        // normalize and assign
        const int output_flat_ind = get_flat_index(x_ind, y_ind, z_ind, ui,
                                                   x_len, y_len, z_len);
        out_sf[output_flat_ind] = tilde_psi_xui / w_norm;
    }
}