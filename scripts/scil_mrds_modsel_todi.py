import numpy as np
import nibabel as nib
import sys, itertools, argparse

parser = argparse.ArgumentParser(description='Apply NuFO map to MRDS output')
parser.add_argument('in_nufo', help='NuFO file to use as model selector map')
parser.add_argument('in_dwi', help='DWI file for anatomical reference.')
parser.add_argument('modsel', help='Model selector prefix')
parser.add_argument('--N1', nargs='*', help='MRDS output N=1 files')
parser.add_argument('--N2', nargs='*', help='MRDS output N=2 files')
parser.add_argument('--N3', nargs='*', help='MRDS output N=3 files')
parser.add_argument('--method', default='Diff', help='estimation method: Diff, Equal or Fixel. [Diff]')
parser.add_argument('--prefix', default='results', help='prefix of the MRDS results [results]')
parser.add_argument('--mask', help='optional mask filename')
args = parser.parse_args()

nufo_filename = args.in_nufo
dwi_filename = args.in_dwi
modsel = args.modsel

n1_files = args.N1
n2_files = args.N2
n3_files = args.N3
method = args.method
prefix = args.prefix
mask_filename = args.mask

mrds_files = [n1_files, n2_files, n3_files]

print('[INFO] Loading input files...')
compsize    = [nib.load(mrds_files[N][0]).get_fdata() for N in range(3)]
eigenvalues = [nib.load(mrds_files[N][1]).get_fdata() for N in range(3)]
isotropic   = [nib.load(mrds_files[N][2]).get_fdata() for N in range(3)]
numcomp     = [nib.load(mrds_files[N][3]).get_fdata() for N in range(3)]
pdds        = [nib.load(mrds_files[N][4]).get_fdata() for N in range(3)]

# load affine and header
dwi_file = nib.load( dwi_filename )
affine = dwi_file.affine
header = dwi_file.header
X,Y,Z  = dwi_file.shape[0:3]

print('[INFO] Loading nufo image...')

# load TODI nufo map
mosemap = nib.load(nufo_filename).get_fdata().astype(np.uint8)

# load mask
if args.mask:
    mask = nib.load(mask_filename).get_fdata().astype(np.uint8)
else:
    mask = np.ones((X,Y,Z), dtype=np.uint8)

# output data
compsize_out    = np.zeros((X,Y,Z,3))
eigenvalues_out = np.zeros((X,Y,Z,9))
isotropic_out   = np.zeros((X,Y,Z,2))
numcomp_out     = np.zeros((X,Y,Z), dtype=np.uint8)
pdds_out        = np.zeros((X,Y,Z,9))

# select data using mosemap
voxels = itertools.product( range(X),range(Y),range(Z) )
for (x,y,z) in voxels:
    if mask[x,y,z]:
        N = mosemap[x,y,z]-1

        if N > 2:
            N = 2

        if N > -1:
            compsize_out[x,y,z,:]    = compsize[N][x,y,z,:]
            eigenvalues_out[x,y,z,:] = eigenvalues[N][x,y,z,:]
            isotropic_out[x,y,z,:]   = isotropic[N][x,y,z,:]
            numcomp_out[x,y,z]       = int(numcomp[N][x,y,z])
            pdds_out[x,y,z,:]        = pdds[N][x,y,z,:]

# write output files
nib.save( nib.Nifti1Image(compsize_out,    affine, header), '%s_MRDS_%s_%s_COMP_SIZE.nii.gz'      % (prefix,method,modsel) )
nib.save( nib.Nifti1Image(eigenvalues_out, affine, header), '%s_MRDS_%s_%s_EIGENVALUES.nii.gz'    % (prefix,method,modsel) )
nib.save( nib.Nifti1Image(isotropic_out,   affine, header), '%s_MRDS_%s_%s_ISOTROPIC.nii.gz'      % (prefix,method,modsel) )
nib.save( nib.Nifti1Image(numcomp_out,     affine, header), '%s_MRDS_%s_%s_NUM_COMP.nii.gz'       % (prefix,method,modsel) )
nib.save( nib.Nifti1Image(pdds_out,        affine, header), '%s_MRDS_%s_%s_PDDs_CARTESIAN.nii.gz' % (prefix,method,modsel) )
