/*
Local tracking OpenCL implementation.

Tracking is performed in voxel space with origin corner.
*/

// Compiler definitions with placeholder values
#define IM_X_DIM 0
#define IM_Y_DIM 0
#define IM_Z_DIM 0
#define IM_N_COEFFS 0
#define N_DIRS 0

#define N_THETAS 0
#define STEP_SIZE 0
#define MAX_LENGTH 0
#define SF_THRESHOLD 0.1f
#define FORWARD_ONLY false
#define SH_INTERP_NN false

// CONSTANTS
#define FLOAT_TO_BOOL_EPSILON 0.1f
#define NULL_SF_EPS 0.0001f

int get_flat_index(const int x, const int y, const int z, const int w,
                   const int xLen, const int yLen, const int zLen)
{
    return x + y * xLen + z * xLen * yLen + w * xLen * yLen * zLen;
}

void reverse_streamline(const int num_strl_points,
                        const int max_num_strl,
                        const size_t seed_indice,
                        __global float* output_tracks,
                        float3* last_pos, float3* last_dir)
{
    // reset last direction to initial direction
    (*last_dir).x = output_tracks[get_flat_index(seed_indice, 0, 0, 0,
                                                 max_num_strl, MAX_LENGTH, 3)]
                - output_tracks[get_flat_index(seed_indice, 1, 0, 0,
                                               max_num_strl, MAX_LENGTH, 3)];
    (*last_dir).y = output_tracks[get_flat_index(seed_indice, 0, 1, 0,
                                                 max_num_strl, MAX_LENGTH, 3)]
                - output_tracks[get_flat_index(seed_indice, 1, 1, 0,
                                               max_num_strl, MAX_LENGTH, 3)];
    (*last_dir).z = output_tracks[get_flat_index(seed_indice, 0, 2, 0,
                                                 max_num_strl, MAX_LENGTH, 3)]
                - output_tracks[get_flat_index(seed_indice, 1, 2, 0,
                                               max_num_strl, MAX_LENGTH, 3)];
    last_dir[0] = normalize(last_dir[0]);

    // reset last position to initial position
    (*last_pos).x = output_tracks[get_flat_index(seed_indice, 0, 0, 0,
                                                 max_num_strl, MAX_LENGTH, 3)];
    (*last_pos).y = output_tracks[get_flat_index(seed_indice, 0, 1, 0,
                                                 max_num_strl, MAX_LENGTH, 3)];
    (*last_pos).z = output_tracks[get_flat_index(seed_indice, 0, 2, 0,
                                                 max_num_strl, MAX_LENGTH, 3)];

    // invert whole streamline
    for(int i = 0; i < (int)(num_strl_points/2); ++i)
    {
        const size_t headx = get_flat_index(seed_indice, i, 0, 0,
                                            max_num_strl, MAX_LENGTH, 3);
        const size_t heady = get_flat_index(seed_indice, i, 1, 0,
                                            max_num_strl, MAX_LENGTH, 3);
        const size_t headz = get_flat_index(seed_indice, i, 2, 0,
                                            max_num_strl, MAX_LENGTH, 3);
        const size_t tailx = get_flat_index(seed_indice, num_strl_points-i-1, 0,
                                            0, max_num_strl, MAX_LENGTH, 3);
        const size_t taily = get_flat_index(seed_indice, num_strl_points-i-1, 1,
                                            0, max_num_strl, MAX_LENGTH, 3);
        const size_t tailz = get_flat_index(seed_indice, num_strl_points-i-1, 2,
                                            0, max_num_strl, MAX_LENGTH, 3);

        // swap start and end points
        const float3 temp_pt = {output_tracks[headx],
                                output_tracks[heady],
                                output_tracks[headz]};
        output_tracks[headx] = output_tracks[tailx];
        output_tracks[heady] = output_tracks[taily];
        output_tracks[headz] = output_tracks[tailz];
        output_tracks[tailx] = temp_pt.x;
        output_tracks[taily] = temp_pt.y;
        output_tracks[tailz] = temp_pt.z;
    }
}

void sh_to_sf(const float* sh_coeffs, __global const float* sh_to_sf_mat,
              const float curr_sf_max, const bool is_first_step,
              __global const float* vertices, const float3 last_dir,
              const float max_cos_theta, float* sf_coeffs)
{
    const float sf_thres = curr_sf_max * SF_THRESHOLD;
    for(int u = 0; u < N_DIRS; ++u)
    {
        const float3 vertice = {
            vertices[get_flat_index(u, 0, 0, 0, N_DIRS, 3, 1)],
            vertices[get_flat_index(u, 1, 0, 0, N_DIRS, 3, 1)],
            vertices[get_flat_index(u, 2, 0, 0, N_DIRS, 3, 1)],
        };

        // all directions are valid for first step.
        bool is_valid = is_first_step;
        // if not in first step, we need to check that
        // we are inside the tracking cone
        if(!is_valid)
        {
            is_valid = dot(last_dir, vertice) > max_cos_theta;
        }

        sf_coeffs[u] = 0.0f;
        if(is_valid)
        {
            for(int j = 0; j < IM_N_COEFFS; ++j)
            {
                const float ylmu_inv = sh_to_sf_mat[get_flat_index(j, u, 0, 0,
                                                                   IM_N_COEFFS,
                                                                   N_DIRS, 1)];
                sf_coeffs[u] += ylmu_inv * sh_coeffs[j];
            }

            // clip values below threshold
            if(sf_coeffs[u] < sf_thres)
            {
                sf_coeffs[u] = 0.0f;
            }
        }
    }
}

void get_value_trilinear(__global const float* image, const int n_channels,
                          const float3 pos, float* values)
{
    /*
    A note on trilinear interpolation
    ---------------------------------
            Z = 0               Z = 1
    2.0 +-----+-----+   2.0 +-----+-----+
        |  C  |  D  |       |  G  |  H  |
    1.0 +-----+-----+   1.0 +-----+-----+
        |  A  |  B  |       |  E  |  F  |
        +-----+-----+       +-----+-----+
    0.0      1.0   2.0     0.0   1.0   2.0
    Because origin is corner, but the ODF at voxel v is centered, i.e.
    we are 100% A at p=(0.5, 0.5), we need to translate p by -0.5
    */
    const float3 t_pos = pos - 0.5f;
    const float xd = t_pos.x - floor(t_pos.x);
    const float yd = t_pos.y - floor(t_pos.y);
    const float zd = t_pos.z - floor(t_pos.z);

    // clip negative values and values higher than image dimensions
    const int x0 = max(min(floor(t_pos.x), (float)(IM_X_DIM - 1)), 0.0f);
    const int x1 = max(min(ceil(t_pos.x), (float)(IM_X_DIM - 1)), 0.0f);
    const int y0 = max(min(floor(t_pos.y), (float)(IM_Y_DIM - 1)), 0.0f);
    const int y1 = max(min(ceil(t_pos.y), (float)(IM_Y_DIM - 1)), 0.0f);
    const int z0 = max(min(floor(t_pos.z), (float)(IM_Z_DIM - 1)), 0.0f);
    const int z1 = max(min(ceil(t_pos.z), (float)(IM_Z_DIM - 1)), 0.0f);

    // we need to loop over all channels
    for(int w = 0; w < n_channels; ++w)
    {
        // Get values at 8 corners
        const float A = image[get_flat_index(x0, y0, z0, w, IM_X_DIM, IM_Y_DIM, IM_Z_DIM)];
        const float B = image[get_flat_index(x1, y0, z0, w, IM_X_DIM, IM_Y_DIM, IM_Z_DIM)];
        const float C = image[get_flat_index(x0, y1, z0, w, IM_X_DIM, IM_Y_DIM, IM_Z_DIM)];
        const float D = image[get_flat_index(x1, y1, z0, w, IM_X_DIM, IM_Y_DIM, IM_Z_DIM)];

        const float E = image[get_flat_index(x0, y0, z1, w, IM_X_DIM, IM_Y_DIM, IM_Z_DIM)];
        const float F = image[get_flat_index(x1, y0, z1, w, IM_X_DIM, IM_Y_DIM, IM_Z_DIM)];
        const float G = image[get_flat_index(x0, y1, z1, w, IM_X_DIM, IM_Y_DIM, IM_Z_DIM)];
        const float H = image[get_flat_index(x1, y1, z1, w, IM_X_DIM, IM_Y_DIM, IM_Z_DIM)];

        // x-interpolation, 4 values AB, CD, EF, GH
        const float AB = xd*B + (1.0f-xd)*A;
        const float CD = xd*D + (1.0f-xd)*C;
        const float EF = xd*F + (1.0f-xd)*E;
        const float GH = xd*H + (1.0f-xd)*G;

        // y-interpolation, 2 values
        const float ABCD = yd*CD + (1.0f-yd)*AB;
        const float EFGH = yd*GH + (1.0f-yd)*EF;

        // finally z-interpolation
        values[w] = EFGH*zd+(1.0f-zd)*ABCD;
    }
}

void get_value_nn(__global const float* image, const int n_channels,
                  const float3 pos, float* value)
{
    const int x = (int)pos.x;
    const int y = (int)pos.y;
    const int z = (int)pos.z;
    for(int w = 0; w < n_channels; ++w)
    {
        value[w] = image[get_flat_index(x, y, z, w, IM_X_DIM, IM_Y_DIM, IM_Z_DIM)];
    }
}

void get_value(__global const float* image, const int n_channels,
               const float3 pos, bool nn_interp, float* value)
{
    if(nn_interp)
    {
        get_value_nn(image, n_channels, pos, value);
    }
    else
    {
        get_value_trilinear(image, n_channels, pos, value);
    }
}

bool is_valid_pos(__global const float* tracking_mask, const float3 pos)
{
    const bool is_inside_volume = pos.x >= 0.0 && pos.x < IM_X_DIM &&
                                  pos.y >= 0.0 && pos.y < IM_Y_DIM &&
                                  pos.z >= 0.0 && pos.z < IM_Z_DIM;

    if(is_inside_volume)
    {
        float mask_value[1];
        get_value_nn(tracking_mask, 1, pos, mask_value);
        const bool is_inside_mask = mask_value[0] > FLOAT_TO_BOOL_EPSILON;
        return is_inside_mask;
    }
    return false;
}

int sample_sf(const float* odf_sf, const float randv)
{
    float cumsum[N_DIRS];
    cumsum[0] = odf_sf[0];
    for(int i = 1; i < N_DIRS; ++i)
    {
        cumsum[i] = odf_sf[i] + cumsum[i - 1];
    }

    if(cumsum[N_DIRS - 1] < NULL_SF_EPS)
    {
        return -1;
    }

    const float where = (randv * cumsum[N_DIRS - 1]);
    int index = 0;

    // increase index until the probability of
    // first element is non-null
    while(cumsum[index] < NULL_SF_EPS)
    {
        ++index;
    }

    // pick sampled direction
    while(cumsum[index] < where && index < N_DIRS)
    {
        ++index;
    }
    return index;
}

int propagate(float3 last_pos, float3 last_dir, int current_length,
              bool is_forward, const size_t seed_indice,
              const size_t n_seeds, const float max_cos_theta_local,
              __global const float* tracking_mask,
              __global const float* sh_coeffs,
              __global const float* sf_max,
              __global const float* rand_f,
              __global const float* vertices,
              __global const float* sh_to_sf_mat,
              __global float* out_streamlines)
{
    bool is_valid = is_valid_pos(tracking_mask, last_pos);

#if !FORWARD_ONLY
    const int max_length = is_forward ?
                           ceil((float)MAX_LENGTH / 2.0f) :
                           current_length + floor((float)MAX_LENGTH / 2.0f);
#else
    const int max_length = MAX_LENGTH;
#endif
    while(current_length < max_length && is_valid)
    {
        // Sample SF at position.
        // Get SH at position.
        float odf_sh[IM_N_COEFFS];
        get_value(sh_coeffs, IM_N_COEFFS, last_pos, SH_INTERP_NN, odf_sh);

        // Convert SH to SF.
        const float curr_sf_max = sf_max[get_flat_index(last_pos.x, last_pos.y,
                                                        last_pos.z, 0, IM_X_DIM,
                                                        IM_Y_DIM, IM_Z_DIM)];
        float odf_sf[N_DIRS];
        sh_to_sf(odf_sh, sh_to_sf_mat, curr_sf_max, current_length == 1,
                 vertices, last_dir, max_cos_theta_local, odf_sf);

        // Sample distribution.
        const float randv = rand_f[get_flat_index(seed_indice, current_length, 0, 0,
                                                  n_seeds, MAX_LENGTH, 1)];
        const int vert_indice = sample_sf(odf_sf, randv);
        if(vert_indice >= 0)
        {
            const float3 direction = {
                vertices[get_flat_index(vert_indice, 0, 0, 0, N_DIRS, 3, 1)],
                vertices[get_flat_index(vert_indice, 1, 0, 0, N_DIRS, 3, 1)],
                vertices[get_flat_index(vert_indice, 2, 0, 0, N_DIRS, 3, 1)]
            };

            // Try step.
            const float3 next_pos = last_pos + STEP_SIZE * direction;
            is_valid = is_valid_pos(tracking_mask, next_pos);
            last_dir = normalize(next_pos - last_pos);
            last_pos = next_pos;
        }
        else
        {
            is_valid = false;
        }

        if(is_valid)
        {
            // save current streamline position
            out_streamlines[get_flat_index(seed_indice, current_length, 0, 0,
                                           n_seeds, MAX_LENGTH, 3)] = last_pos.x;
            out_streamlines[get_flat_index(seed_indice, current_length, 1, 0,
                                           n_seeds, MAX_LENGTH, 3)] = last_pos.y;
            out_streamlines[get_flat_index(seed_indice, current_length, 2, 0,
                                           n_seeds, MAX_LENGTH, 3)] = last_pos.z;

            // increment track length
            ++current_length;
        }
    }
    // finally, we return the streamline length
    return current_length;
}

int track(float3 seed_pos,
          const size_t seed_indice,
          const size_t n_seeds,
          const float max_cos_theta_local,
          __global const float* tracking_mask,
          __global const float* sh_coeffs,
          __global const float* sf_max,
          __global const float* rand_f,
          __global const float* vertices,
          __global const float* sh_to_sf_mat,
          __global float* out_streamlines)
{
    float3 last_pos = seed_pos;
    int current_length = 0;

    // add seed position to track
    out_streamlines[get_flat_index(seed_indice, current_length, 0, 0,
                                   n_seeds, MAX_LENGTH, 3)] = last_pos.x;
    out_streamlines[get_flat_index(seed_indice, current_length, 1, 0,
                                   n_seeds, MAX_LENGTH, 3)] = last_pos.y;
    out_streamlines[get_flat_index(seed_indice, current_length, 2, 0,
                                   n_seeds, MAX_LENGTH, 3)] = last_pos.z;
    ++current_length;

    // forward track
    float3 last_dir;
    current_length = propagate(last_pos, last_dir, current_length, true,
                               seed_indice, n_seeds, max_cos_theta_local,
                               tracking_mask, sh_coeffs, sf_max, rand_f, vertices,
                               sh_to_sf_mat, out_streamlines);

    // reverse streamline for backward tracking
#if !FORWARD_ONLY
    if(current_length > 1 && current_length < MAX_LENGTH)
    {
        reverse_streamline(current_length, n_seeds,
                           seed_indice, out_streamlines,
                           &last_pos, &last_dir);

        // track backward
        current_length = propagate(last_pos, last_dir, current_length, false,
                                   seed_indice, n_seeds, max_cos_theta_local,
                                   tracking_mask, sh_coeffs, sf_max, rand_f, vertices,
                                   sh_to_sf_mat, out_streamlines);
    }
#endif
    return current_length;
}

__kernel void tracker(__global const float* sh_coeffs,
                      __global const float* vertices,
                      __global const float* sh_to_sf_mat,
                      __global const float* sf_max,
                      __global const float* tracking_mask,
                      __global const float* max_cos_theta,
                      __global const float* seed_positions,
                      __global const float* rand_f,
                      __global float* out_streamlines,
                      __global float* out_nb_points)
{
    // 1. Get seed position from global_id.
    const size_t seed_indice = get_global_id(0);
    const int n_seeds = get_global_size(0);
    float max_cos_theta_local = max_cos_theta[0];

    const float3 seed_pos = {
        seed_positions[get_flat_index(seed_indice, 0, 0, 0, n_seeds, 3, 1)],
        seed_positions[get_flat_index(seed_indice, 1, 0, 0, n_seeds, 3, 1)],
        seed_positions[get_flat_index(seed_indice, 2, 0, 0, n_seeds, 3, 1)]
    };

    if(N_THETAS > 1) // Varying radius of curvature.
    {
        // extract random value using fractional part of voxel position
        // for selecting the max theta for this streamline
        float itpr;
        const float rand_v = fract(seed_pos.x+seed_pos.y+seed_pos.z, &itpr);
        max_cos_theta_local = max_cos_theta[(int)(rand_v * (float)N_THETAS)];
    }

    int current_length = track(seed_pos, seed_indice, n_seeds,
                               max_cos_theta_local, tracking_mask,
                               sh_coeffs, sf_max, rand_f, vertices,
                               sh_to_sf_mat, out_streamlines);

    out_nb_points[seed_indice] = (float)current_length;
}
