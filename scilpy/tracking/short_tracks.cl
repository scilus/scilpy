/*
OpenCL kernel code for computing short-tracks tractogram from
SH volume.
*/

// Compiler definitions with placeholder values
#define IN_N_COEFFS 0
#define OUT_N_COEFFS 0
#define N_DIRS 0
#define IM_X_DIM 0
#define IM_Y_DIM 0
#define IM_Z_DIM 0

// TODO:
// * sh_to_sf
// * nearest_neighbour interpolate
// * is_inside_mask & is_inside_volume methods
// * sample_sf, pre-filled random numbers from uniform dist,
//              then cumsum and search_sorted.
// is forward tracking enough?


__kernel void track(__global const float* sh_coeffs, // whole brain fits easily
                    __global const float* vertices, // 724 x 3 floats
                    __global const float* sh_to_sf_mat, // 45 * 724 floats
                    __global const float* seed_pos, // n_seeds_batch * 3 floats
                    __global const float* tracking_mask, // dim.x*dim.y*dim.z floats
                    __global const float* rand_f, // n_seeds_batch * (max_length - 1) floats
                    __global float* out_streamlines) // n_seeds_batch * max_length * 3 floats
{
    // get_global_id(0)*3 is the index in seed_pos array and
    // is some precomputed random seed position.
}
