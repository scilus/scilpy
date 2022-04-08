/*
OpenCL kernel code for computing short-tracks tractogram from
SH volume. Tracking is performed in voxel space.
*/

// Compiler definitions with placeholder values
#define N_DIRS 0
#define IM_X_DIM 0
#define IM_Y_DIM 0
#define IM_Z_DIM 0
#define IM_N_COEFFS 0

#define FLOAT_TO_BOOL_EPSILON 0.1f

// TODO:
// * sh_to_sf
// * sample_sf, pre-filled random numbers from uniform dist,
//              then cumsum and search_sorted.
// is forward-tracking enough?

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

__kernel void track(__global const float* sh_coeffs, // whole brain fits easily
                    __global const float* vertices, // 724 x 3 floats
                    __global const float* sh_to_sf_mat, // 45 * 724 floats
                    __global const float* seed_pos, // n_seeds_batch * 3 floats
                    __global const float* tracking_mask, // dim.x*dim.y*dim.z floats
                    __global const float* rand_f, // n_seeds_batch * (max_length - 1) floats
                    __global float* out_streamlines, // n_seeds_batch * max_length * 3 floats
                    __global float* out_nb_points) // n_seeds_batch
{
    // 1. Get seed position from global_id.
    // 2. Sample SF at position.
    // 3. Try step.
    // 4. If new position is valid, write in out_streamlines and go to 2.
    //    Else go to 5.
    // 5. Write number of points for streamline in out_nb_points.
}
