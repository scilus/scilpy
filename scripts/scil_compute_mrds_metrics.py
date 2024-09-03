import numpy as np
import nibabel as nib
import itertools, argparse

def lambdas2fa(lambdas):
    a = np.sqrt(0.5)
    b = np.sqrt( (lambdas[0]-lambdas[1])**2 + (lambdas[1]-lambdas[2])**2 + (lambdas[2]-lambdas[0])**2 )
    c = np.sqrt( lambdas[0]**2 + lambdas[1]**2 + lambdas[2]**2 )

    return a*b/c

def lambdas2md(lambdas):
    return ( lambdas[0]+lambdas[1]+lambdas[2] )/3

def main():
    parser = argparse.ArgumentParser(description='Convert MRDS output into fixel-AD, fixel-RD, fixel-FA and fixel-MD metrics.')
    parser.add_argument('in_eigenvalues', help='MRDS eigenvalues file')
    parser.add_argument('out_ad', help='MRDS fixel-AD file')
    parser.add_argument('out_rd', help='MRDS fixel-RD file')
    parser.add_argument('out_md', help='MRDS fixel-MD file')
    parser.add_argument('out_fa', help='MRDS fixel-FA file')
    parser.add_argument('--mask', help='optional mask file')
    args = parser.parse_args()

    eigenvalues_filename = args.in_eigenvalues
    ad_filename = args.out_ad
    rd_filename = args.out_rd
    md_filename = args.out_md
    fa_filename = args.out_fa

    eigenvalues_file = nib.load( eigenvalues_filename )
    lambdas = eigenvalues_file.get_fdata()
    affine = eigenvalues_file.affine
    header = eigenvalues_file.header

    X,Y,Z = lambdas.shape[0:3]
    voxels = itertools.product( range(X),range(Y),range(Z) )

    if args.mask:
        mask = nib.load(args.mask).get_fdata().astype(np.uint8)
    else:
        mask = np.ones( (X,Y,Z),dtype=np.uint8 )

    ad = np.zeros( (X,Y,Z,3) )
    rd = np.zeros( (X,Y,Z,3) )
    md = np.zeros( (X,Y,Z,3) )
    fa = np.zeros( (X,Y,Z,3) )

    for (x,y,z) in voxels:
        if mask[x,y,z]:
            ad[x,y,z, 0:3] = np.array([ lambdas[x,y,z, 0], lambdas[x,y,z, 3], lambdas[x,y,z, 6] ])
            rd[x,y,z, 0:3] = np.array([ (lambdas[x,y,z, 1]+lambdas[x,y,z, 2])/2, (lambdas[x,y,z, 4]+lambdas[x,y,z, 5])/2, (lambdas[x,y,z, 7]+lambdas[x,y,z, 8])/2 ])
            md[x,y,z, 0:3] = np.array([ lambdas2md(lambdas[x,y,z, 0:3]), lambdas2md(lambdas[x,y,z, 3:6]), lambdas2md(lambdas[x,y,z, 6:9]) ])
            fa[x,y,z, 0:3] = np.array([ lambdas2fa(lambdas[x,y,z, 0:3]), lambdas2fa(lambdas[x,y,z, 3:6]), lambdas2fa(lambdas[x,y,z, 6:9]) ])

    nib.save( nib.Nifti1Image(ad, affine, header), ad_filename )
    nib.save( nib.Nifti1Image(rd, affine, header), rd_filename )
    nib.save( nib.Nifti1Image(md, affine, header), md_filename )
    nib.save( nib.Nifti1Image(fa, affine, header), fa_filename )

if __name__ == '__main__':
    main()
