/*
GPU particle-filtering tracking OpenCL implementation.

This kernel extends local tracking with a particle-based recovery stage
triggered when normal propagation fails.

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
#define PARTICLE_COUNT 0
#define BACK_STEPS 0
#define FRONT_STEPS 0
#define RECOVERY_STEPS 0

// CONSTANTS
#define FLOAT_TO_BOOL_EPSILON 0.1f
#define NULL_SF_EPS 0.0001f

int get_flat_index(const int x, const int y, const int z, const int w,
                   const int xLen, const int yLen, const int zLen)
{
    return x + y * xLen + z * xLen * yLen + w * xLen * yLen * zLen;
}

uint hash_u32(uint x)
{
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    x ^= x >> 16;
    return x;
}

float rand01(const size_t seed_indice, const int current_length,
             const int particle_id, const int step_id)
{
    uint state = 0x9e3779b9u;
    state ^= (uint)(seed_indice + 1u) * 0x85ebca6bu;
    state ^= (uint)(current_length + 1) * 0xc2b2ae35u;
    state ^= (uint)(particle_id + 1) * 0x27d4eb2du;
    state ^= (uint)(step_id + 1) * 0x165667b1u;
    return (float)hash_u32(state) / 4294967296.0f;
}

void reverse_streamline(const int num_strl_points,
                        const int max_num_strl,
                        const size_t seed_indice,
                        __global float* output_tracks,
                        float3* last_pos, float3* last_dir)
{
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

    (*last_pos).x = output_tracks[get_flat_index(seed_indice, 0, 0, 0,
                                                 max_num_strl, MAX_LENGTH, 3)];
    (*last_pos).y = output_tracks[get_flat_index(seed_indice, 0, 1, 0,
                                                 max_num_strl, MAX_LENGTH, 3)];
    (*last_pos).z = output_tracks[get_flat_index(seed_indice, 0, 2, 0,
                                                 max_num_strl, MAX_LENGTH, 3)];

    for(int i = 0; i < (int)(num_strl_points / 2); ++i)
    {
        const size_t headx = get_flat_index(seed_indice, i, 0, 0,
                                            max_num_strl, MAX_LENGTH, 3);
        const size_t heady = get_flat_index(seed_indice, i, 1, 0,
                                            max_num_strl, MAX_LENGTH, 3);
        const size_t headz = get_flat_index(seed_indice, i, 2, 0,
                                            max_num_strl, MAX_LENGTH, 3);
        const size_t tailx = get_flat_index(seed_indice, num_strl_points - i - 1, 0,
                                            0, max_num_strl, MAX_LENGTH, 3);
        const size_t taily = get_flat_index(seed_indice, num_strl_points - i - 1, 1,
                                            0, max_num_strl, MAX_LENGTH, 3);
        const size_t tailz = get_flat_index(seed_indice, num_strl_points - i - 1, 2,
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

void get_value_trilinear(__global const float* image, const int n_channels,
                         const float3 pos, float* values)
{
    const float3 t_pos = pos - 0.5f;
    const float xd = t_pos.x - floor(t_pos.x);
    const float yd = t_pos.y - floor(t_pos.y);
    const float zd = t_pos.z - floor(t_pos.z);

    const int x0 = max(min(floor(t_pos.x), (float)(IM_X_DIM - 1)), 0.0f);
    const int x1 = max(min(ceil(t_pos.x), (float)(IM_X_DIM - 1)), 0.0f);
    const int y0 = max(min(floor(t_pos.y), (float)(IM_Y_DIM - 1)), 0.0f);
    const int y1 = max(min(ceil(t_pos.y), (float)(IM_Y_DIM - 1)), 0.0f);
    const int z0 = max(min(floor(t_pos.z), (float)(IM_Z_DIM - 1)), 0.0f);
    const int z1 = max(min(ceil(t_pos.z), (float)(IM_Z_DIM - 1)), 0.0f);

    for(int w = 0; w < n_channels; ++w)
    {
        const float A = image[get_flat_index(x0, y0, z0, w, IM_X_DIM, IM_Y_DIM, IM_Z_DIM)];
        const float B = image[get_flat_index(x1, y0, z0, w, IM_X_DIM, IM_Y_DIM, IM_Z_DIM)];
        const float C = image[get_flat_index(x0, y1, z0, w, IM_X_DIM, IM_Y_DIM, IM_Z_DIM)];
        const float D = image[get_flat_index(x1, y1, z0, w, IM_X_DIM, IM_Y_DIM, IM_Z_DIM)];
        const float E = image[get_flat_index(x0, y0, z1, w, IM_X_DIM, IM_Y_DIM, IM_Z_DIM)];
        const float F = image[get_flat_index(x1, y0, z1, w, IM_X_DIM, IM_Y_DIM, IM_Z_DIM)];
        const float G = image[get_flat_index(x0, y1, z1, w, IM_X_DIM, IM_Y_DIM, IM_Z_DIM)];
        const float H = image[get_flat_index(x1, y1, z1, w, IM_X_DIM, IM_Y_DIM, IM_Z_DIM)];

        const float AB = xd * B + (1.0f - xd) * A;
        const float CD = xd * D + (1.0f - xd) * C;
        const float EF = xd * F + (1.0f - xd) * E;
        const float GH = xd * H + (1.0f - xd) * G;

        const float ABCD = yd * CD + (1.0f - yd) * AB;
        const float EFGH = yd * GH + (1.0f - yd) * EF;

        values[w] = EFGH * zd + (1.0f - zd) * ABCD;
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

bool is_inside_volume(const float3 pos)
{
    return pos.x >= 0.0f && pos.x < IM_X_DIM &&
           pos.y >= 0.0f && pos.y < IM_Y_DIM &&
           pos.z >= 0.0f && pos.z < IM_Z_DIM;
}

bool is_valid_pos(__global const float* tracking_mask, const float3 pos)
{
    if(!is_inside_volume(pos))
    {
        return false;
    }

    float mask_value[1];
    get_value_nn(tracking_mask, 1, pos, mask_value);
    return mask_value[0] > FLOAT_TO_BOOL_EPSILON;
}

bool is_include_pos(__global const float* include_map,
                    __global const float* exclude_map,
                    const float3 pos)
{
    if(!is_inside_volume(pos))
    {
        return false;
    }

    float include_v[1];
    float exclude_v[1];
    get_value_nn(include_map, 1, pos, include_v);
    get_value_nn(exclude_map, 1, pos, exclude_v);
    return include_v[0] > FLOAT_TO_BOOL_EPSILON &&
           exclude_v[0] <= FLOAT_TO_BOOL_EPSILON;
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

        bool is_valid = is_first_step;
        if(!is_valid)
        {
            is_valid = dot(last_dir, vertice) > max_cos_theta;
        }

        sf_coeffs[u] = 0.0f;
        if(is_valid)
        {
            for(int j = 0; j < IM_N_COEFFS; ++j)
            {
                const float ylmu_inv = sh_to_sf_mat[
                    get_flat_index(j, u, 0, 0, IM_N_COEFFS, N_DIRS, 1)];
                sf_coeffs[u] += ylmu_inv * sh_coeffs[j];
            }

            if(sf_coeffs[u] < sf_thres)
            {
                sf_coeffs[u] = 0.0f;
            }
        }
    }
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

    const float where = randv * cumsum[N_DIRS - 1];
    int index = 0;

    while(index < N_DIRS && cumsum[index] < NULL_SF_EPS)
    {
        ++index;
    }
    if(index >= N_DIRS)
    {
        return -1;
    }

    while(index < N_DIRS && cumsum[index] < where)
    {
        ++index;
    }
    if(index >= N_DIRS)
    {
        index = N_DIRS - 1;
    }
    return index;
}

bool step_once(const float3 pos, const float3 last_dir,
               const float max_cos_theta_local,
               __global const float* sh_coeffs,
               __global const float* sf_max,
               __global const float* vertices,
               __global const float* sh_to_sf_mat,
               const float randv,
               float3* next_pos,
               float3* next_dir)
{
    float odf_sh[IM_N_COEFFS];
    get_value(sh_coeffs, IM_N_COEFFS, pos, SH_INTERP_NN, odf_sh);

    const float curr_sf_max = sf_max[get_flat_index(pos.x, pos.y,
                                                    pos.z, 0,
                                                    IM_X_DIM, IM_Y_DIM, IM_Z_DIM)];

    float odf_sf[N_DIRS];
    sh_to_sf(odf_sh, sh_to_sf_mat, curr_sf_max, false,
             vertices, last_dir, max_cos_theta_local, odf_sf);

    const int vert_indice = sample_sf(odf_sf, randv);
    if(vert_indice < 0)
    {
        return false;
    }

    const float3 direction = {
        vertices[get_flat_index(vert_indice, 0, 0, 0, N_DIRS, 3, 1)],
        vertices[get_flat_index(vert_indice, 1, 0, 0, N_DIRS, 3, 1)],
        vertices[get_flat_index(vert_indice, 2, 0, 0, N_DIRS, 3, 1)]
    };

    next_pos[0] = pos + STEP_SIZE * direction;
    next_dir[0] = normalize(next_pos[0] - pos);
    return true;
}

int recover_with_particles(const size_t seed_indice,
                           const int n_seeds,
                           int current_length,
                           float3* last_pos,
                           float3* last_dir,
                           const float max_cos_theta_local,
                           __global const float* tracking_mask,
                           __global const float* include_map,
                           __global const float* exclude_map,
                           __global const float* sh_coeffs,
                           __global const float* sf_max,
                           __global const float* vertices,
                           __global const float* sh_to_sf_mat,
                           __global float* out_streamlines)
{
    if(PARTICLE_COUNT <= 0 || RECOVERY_STEPS <= 0 || current_length < 1)
    {
        return current_length;
    }

    int anchor_idx = current_length - 1 - BACK_STEPS;
    if(anchor_idx < 0)
    {
        anchor_idx = 0;
    }

    float3 anchor_pos = {
        out_streamlines[get_flat_index(seed_indice, anchor_idx, 0, 0,
                                       n_seeds, MAX_LENGTH, 3)],
        out_streamlines[get_flat_index(seed_indice, anchor_idx, 1, 0,
                                       n_seeds, MAX_LENGTH, 3)],
        out_streamlines[get_flat_index(seed_indice, anchor_idx, 2, 0,
                                       n_seeds, MAX_LENGTH, 3)]
    };

    float3 anchor_dir = last_dir[0];
    if(anchor_idx > 0)
    {
        float3 prev_pos = {
            out_streamlines[get_flat_index(seed_indice, anchor_idx - 1, 0, 0,
                                           n_seeds, MAX_LENGTH, 3)],
            out_streamlines[get_flat_index(seed_indice, anchor_idx - 1, 1, 0,
                                           n_seeds, MAX_LENGTH, 3)],
            out_streamlines[get_flat_index(seed_indice, anchor_idx - 1, 2, 0,
                                           n_seeds, MAX_LENGTH, 3)]
        };
        anchor_dir = normalize(anchor_pos - prev_pos);
    }

    float3 best_path[RECOVERY_STEPS];
    int best_steps = 0;
    bool best_included = false;

    for(int p = 0; p < PARTICLE_COUNT; ++p)
    {
        float3 path[RECOVERY_STEPS];
        float3 p_pos = anchor_pos;
        float3 p_dir = anchor_dir;
        int p_steps = 0;
        bool p_included = false;

        for(int k = 0; k < RECOVERY_STEPS; ++k)
        {
            float3 next_pos;
            float3 next_dir;
            const float rv = rand01(seed_indice, current_length, p, k);
            bool sampled = step_once(p_pos, p_dir,
                                     max_cos_theta_local,
                                     sh_coeffs, sf_max,
                                     vertices, sh_to_sf_mat,
                                     rv, &next_pos, &next_dir);
            if(!sampled || !is_valid_pos(tracking_mask, next_pos))
            {
                break;
            }

            path[p_steps] = next_pos;
            ++p_steps;
            p_pos = next_pos;
            p_dir = next_dir;

            if(is_include_pos(include_map, exclude_map, p_pos))
            {
                p_included = true;
                break;
            }
        }

        bool take_candidate = false;
        if(p_included && !best_included)
        {
            take_candidate = true;
        }
        else if(p_included == best_included && p_steps > best_steps)
        {
            take_candidate = true;
        }

        if(take_candidate)
        {
            best_steps = p_steps;
            best_included = p_included;
            for(int i = 0; i < best_steps; ++i)
            {
                best_path[i] = path[i];
            }
        }
    }

    if(best_steps <= 0)
    {
        return current_length;
    }

    current_length = anchor_idx + 1;
    for(int i = 0; i < best_steps && current_length < MAX_LENGTH; ++i)
    {
        out_streamlines[get_flat_index(seed_indice, current_length, 0, 0,
                                       n_seeds, MAX_LENGTH, 3)] = best_path[i].x;
        out_streamlines[get_flat_index(seed_indice, current_length, 1, 0,
                                       n_seeds, MAX_LENGTH, 3)] = best_path[i].y;
        out_streamlines[get_flat_index(seed_indice, current_length, 2, 0,
                                       n_seeds, MAX_LENGTH, 3)] = best_path[i].z;
        last_pos[0] = best_path[i];
        if(i > 0)
        {
            last_dir[0] = normalize(best_path[i] - best_path[i - 1]);
        }
        ++current_length;
    }

    return current_length;
}

int propagate(float3 last_pos, float3 last_dir, int current_length,
              bool is_forward, const size_t seed_indice,
              const size_t n_seeds, const float max_cos_theta_local,
              __global const float* tracking_mask,
              __global const float* include_map,
              __global const float* exclude_map,
              __global const float* sh_coeffs,
              __global const float* sf_max,
              __global const float* rand_f,
              __global const float* vertices,
              __global const float* sh_to_sf_mat,
              __global float* out_streamlines)
{
    bool can_continue = is_valid_pos(tracking_mask, last_pos);

#if !FORWARD_ONLY
    const int max_length = is_forward ?
                           ceil((float)MAX_LENGTH / 2.0f) :
                           current_length + floor((float)MAX_LENGTH / 2.0f);
#else
    const int max_length = MAX_LENGTH;
#endif

    while(current_length < max_length && can_continue)
    {
        float3 next_pos;
        float3 next_dir;

        const float randv = rand_f[get_flat_index(seed_indice, current_length, 0, 0,
                                                  n_seeds, MAX_LENGTH, 1)];
        bool sampled = step_once(last_pos, last_dir,
                                 max_cos_theta_local,
                                 sh_coeffs, sf_max,
                                 vertices, sh_to_sf_mat,
                                 randv,
                                 &next_pos, &next_dir);

        if(!sampled || !is_valid_pos(tracking_mask, next_pos))
        {
            int recovered_len = recover_with_particles(
                seed_indice,
                n_seeds,
                current_length,
                &last_pos,
                &last_dir,
                max_cos_theta_local,
                tracking_mask,
                include_map,
                exclude_map,
                sh_coeffs,
                sf_max,
                vertices,
                sh_to_sf_mat,
                out_streamlines);

            if(recovered_len == current_length)
            {
                break;
            }
            current_length = recovered_len;
            continue;
        }

        last_pos = next_pos;
        last_dir = next_dir;

        out_streamlines[get_flat_index(seed_indice, current_length, 0, 0,
                                       n_seeds, MAX_LENGTH, 3)] = last_pos.x;
        out_streamlines[get_flat_index(seed_indice, current_length, 1, 0,
                                       n_seeds, MAX_LENGTH, 3)] = last_pos.y;
        out_streamlines[get_flat_index(seed_indice, current_length, 2, 0,
                                       n_seeds, MAX_LENGTH, 3)] = last_pos.z;
        ++current_length;
    }

    return current_length;
}

int track(float3 seed_pos,
          const size_t seed_indice,
          const size_t n_seeds,
          const float max_cos_theta_local,
          __global const float* tracking_mask,
          __global const float* include_map,
          __global const float* exclude_map,
          __global const float* sh_coeffs,
          __global const float* sf_max,
          __global const float* rand_f,
          __global const float* vertices,
          __global const float* sh_to_sf_mat,
          __global float* out_streamlines)
{
    float3 last_pos = seed_pos;
    int current_length = 0;

    out_streamlines[get_flat_index(seed_indice, current_length, 0, 0,
                                   n_seeds, MAX_LENGTH, 3)] = last_pos.x;
    out_streamlines[get_flat_index(seed_indice, current_length, 1, 0,
                                   n_seeds, MAX_LENGTH, 3)] = last_pos.y;
    out_streamlines[get_flat_index(seed_indice, current_length, 2, 0,
                                   n_seeds, MAX_LENGTH, 3)] = last_pos.z;
    ++current_length;

    float3 last_dir = (float3)(1.0f, 0.0f, 0.0f);
    current_length = propagate(last_pos, last_dir, current_length, true,
                               seed_indice, n_seeds, max_cos_theta_local,
                               tracking_mask, include_map, exclude_map,
                               sh_coeffs, sf_max, rand_f, vertices,
                               sh_to_sf_mat, out_streamlines);

#if !FORWARD_ONLY
    if(current_length > 1 && current_length < MAX_LENGTH)
    {
        reverse_streamline(current_length, n_seeds,
                           seed_indice, out_streamlines,
                           &last_pos, &last_dir);

        current_length = propagate(last_pos, last_dir, current_length, false,
                                   seed_indice, n_seeds, max_cos_theta_local,
                                   tracking_mask, include_map, exclude_map,
                                   sh_coeffs, sf_max, rand_f, vertices,
                                   sh_to_sf_mat, out_streamlines);
    }
#endif

    return current_length;
}

__kernel void pft_tracker(__global const float* sh_coeffs,
                          __global const float* vertices,
                          __global const float* sh_to_sf_mat,
                          __global const float* sf_max,
                          __global const float* tracking_mask,
                          __global const float* include_map,
                          __global const float* exclude_map,
                          __global const float* max_cos_theta,
                          __global const float* seed_positions,
                          __global const float* rand_f,
                          __global float* out_streamlines,
                          __global float* out_nb_points,
                          __global float* out_included)
{
    const size_t seed_indice = get_global_id(0);
    const int n_seeds = get_global_size(0);
    float max_cos_theta_local = max_cos_theta[0];

    const float3 seed_pos = {
        seed_positions[get_flat_index(seed_indice, 0, 0, 0, n_seeds, 3, 1)],
        seed_positions[get_flat_index(seed_indice, 1, 0, 0, n_seeds, 3, 1)],
        seed_positions[get_flat_index(seed_indice, 2, 0, 0, n_seeds, 3, 1)]
    };

    if(N_THETAS > 1)
    {
        float itpr;
        const float rand_v = fract(seed_pos.x + seed_pos.y + seed_pos.z, &itpr);
        max_cos_theta_local = max_cos_theta[(int)(rand_v * (float)N_THETAS)];
    }

    int current_length = track(seed_pos, seed_indice, n_seeds,
                               max_cos_theta_local,
                               tracking_mask,
                               include_map,
                               exclude_map,
                               sh_coeffs,
                               sf_max,
                               rand_f,
                               vertices,
                               sh_to_sf_mat,
                               out_streamlines);

    out_nb_points[seed_indice] = (float)current_length;

    float3 end_pos = seed_pos;
    if(current_length > 0)
    {
        const int end_idx = current_length - 1;
        end_pos.x = out_streamlines[get_flat_index(seed_indice, end_idx, 0, 0,
                                                   n_seeds, MAX_LENGTH, 3)];
        end_pos.y = out_streamlines[get_flat_index(seed_indice, end_idx, 1, 0,
                                                   n_seeds, MAX_LENGTH, 3)];
        end_pos.z = out_streamlines[get_flat_index(seed_indice, end_idx, 2, 0,
                                                   n_seeds, MAX_LENGTH, 3)];
    }

    out_included[seed_indice] = is_include_pos(include_map, exclude_map, end_pos)
                                 ? 1.0f : 0.0f;
}
