import numpy as np
import nibabel as nib
import os,argparse,itertools

parser = argparse.ArgumentParser(description='Run TODI and create model selector map for MRDS.')
parser.add_argument('in_tractogram', help='Tractogram filename. Format must be tck.')
parser.add_argument('in_dwi', help='DWI file for anatomical reference.')
parser.add_argument('out_tod', help='ouput TOD image.')
parser.add_argument('out_nufo', help='ouput TODI NuFO image.')
parser.add_argument('--prefix', default='results', help='Prefix of the MRDS results. [results]')
parser.add_argument('--method', default='Diff', help='Method used when estimating the fixel diffusivities in MRDS {Fixed,Equal,Diff}. [Diff]')
parser.add_argument('--modsel', default='bic', help='Model selector {aic,bic,ftest}. [bic]')
parser.add_argument('--lmax', type=int, default=16, help='lmax for the TOD. [16]')
parser.add_argument('--threshold', type=float, default=0.1, help='Threshold peak amplitude of positive TOD lobes. [0.1]')
args = parser.parse_args()

in_tractogram = args.in_tractogram
in_dwi = args.in_dwi
out_tod = args.out_tod
out_nufo = args.out_nufo

prefix = args.prefix
method = args.method
modsel = args.modsel
lmax = args.lmax
threshold = args.threshold

out_path = 'temp_fixels'

# compute TOD
os.system('tckmap %s %s -tod %d -template %s -force' % (in_tractogram,out_tod,lmax,in_dwi))

if os.path.exists( out_path ):
    os.system('rm -rf %s' % out_path)
os.system('mkdir %s' % out_path)

os.system('fod2fixel %s %s -nii -maxnum 3 -fmls_peak_value %f -force' % (out_tod,out_path,threshold))
    
os.system('fixel2voxel %s/directions.nii count %s/nufo.nii -force' % (out_path,out_path))

os.system('gzip %s/nufo.nii %s' % (out_path,out_nufo))

os.system('rm -rf %s' % out_path)