/*
OpenCL kernel code for computing short-tracks tractogram from
SH volume. Tracking is performed in voxel space.
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
#define FORWARD_ONLY false

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
                        __global float* output_tracks)
{
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
              const bool is_first_step, __global const float* vertices,
              const float3 last_dir, const float max_cos_theta, float* sf_coeffs)
{
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
            // clip negative values
            if(sf_coeffs[u] < 0.0f)
            {
                sf_coeffs[u] = 0.0f;
            }
        }
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
              const size_t seed_indice, const size_t n_seeds,
              const float max_cos_theta_local,
              __global const float* tracking_mask,
              __global const float* sh_coeffs,
              __global const float* rand_f,
              __global const float* vertices,
              __global const float* sh_to_sf_mat,
              __global float* out_streamlines)
{
    bool is_valid = is_valid_pos(tracking_mask, last_pos);
    while(current_length < MAX_LENGTH && is_valid)
    {
        // Sample SF at position.
        float odf_sh[IM_N_COEFFS];
        float odf_sf[N_DIRS];
        get_value_nn(sh_coeffs, IM_N_COEFFS, last_pos, odf_sh);
        sh_to_sf(odf_sh, sh_to_sf_mat, current_length == 1, vertices,
                 last_dir, max_cos_theta_local, odf_sf);

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
    return current_length;
}

__kernel void track(__global const float* sh_coeffs,
                    __global const float* vertices,
                    __global const float* sh_to_sf_mat,
                    __global const float* tracking_mask,
                    __global const float* seed_positions,
                    __global const float* rand_f,
                    __global const float* max_cos_theta,
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

    float3 last_pos = seed_pos;
    int current_length = 0;

    // add seed position to track
    out_streamlines[get_flat_index(seed_indice, current_length, 0, 0,
                                   n_seeds, MAX_LENGTH, 3)] = last_pos.x;
    out_streamlines[get_flat_index(seed_indice, current_length, 1, 0,
                                   n_seeds, MAX_LENGTH, 3)] = last_pos.y;
    out_streamlines[get_flat_index(seed_indice, current_length, 2, 0,
                                   n_seeds, MAX_LENGTH, 3)] = last_pos.z;
    // forward track
    float3 last_dir;
    ++current_length;
    current_length = propagate(last_pos, last_dir, current_length,
                               seed_indice, n_seeds, max_cos_theta_local,
                               tracking_mask, sh_coeffs, rand_f, vertices,
                               sh_to_sf_mat,  out_streamlines);

    // reverse streamline for backward tracking
    if(current_length > 1 && current_length < MAX_LENGTH && !FORWARD_ONLY)
    {
        // reset last direction to initial direction
        last_dir.x = out_streamlines[get_flat_index(seed_indice, 0, 0, 0,
                                                    n_seeds, MAX_LENGTH, 3)]
                   - out_streamlines[get_flat_index(seed_indice, 1, 0, 0,
                                                    n_seeds, MAX_LENGTH, 3)];
        last_dir.y = out_streamlines[get_flat_index(seed_indice, 0, 1, 0,
                                                    n_seeds, MAX_LENGTH, 3)]
                   - out_streamlines[get_flat_index(seed_indice, 1, 1, 0,
                                                    n_seeds, MAX_LENGTH, 3)];
        last_dir.z = out_streamlines[get_flat_index(seed_indice, 0, 2, 0,
                                                    n_seeds, MAX_LENGTH, 3)]
                   - out_streamlines[get_flat_index(seed_indice, 1, 2, 0,
                                                    n_seeds, MAX_LENGTH, 3)];
        last_dir = normalize(last_dir);

        // reverse streamline so the output is continuous
        reverse_streamline(current_length, n_seeds,
                           seed_indice, out_streamlines);

        // track backward
        current_length = propagate(last_pos, last_dir, current_length,
                                   seed_indice, n_seeds, max_cos_theta_local,
                                   tracking_mask, sh_coeffs, rand_f, vertices,
                                   sh_to_sf_mat,  out_streamlines);
    }
    out_nb_points[seed_indice] = (float)current_length;
}
