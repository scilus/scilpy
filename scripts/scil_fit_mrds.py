import argparse, os

def read_lambdas(lambdas_filename):
    with open(lambdas_filename, 'rt') as lambdas_file:
        line = lambdas_file.readlines()[0]
        lambdas = line.split(' ') [0:3]
        lambdas = [float(value) for value in lambdas]

               #lambda1    #lambda23 
        return lambdas[0], (lambdas[1]+lambdas[2])/2.0

def main():
    parser = argparse.ArgumentParser(description='Fit MRDS to DWI data.')
    parser.add_argument('in_dwi', help='DWI file for anatomical reference.')
    parser.add_argument('in_scheme', help='DWI file for anatomical reference.')
    parser.add_argument('--mask', help='optional mask filename')
    parser.add_argument('--modsel', default='bic', help='model selector. Options: aic, bic, ftest. [bic]')
    parser.add_argument('--method', default='Diff', help='estimation method for tensor diffusivities. Options: Diff, Equal, Fixel. [Diff]')
    parser.add_argument('--prefix', default='results', help='prefix of the MRDS results. [results]')
    parser.add_argument('-iso', help='include isotropic tensor fitting to MRDS', action='store_true')
    args = parser.parse_args()

    dwi = args.in_dwi
    scheme = args.in_scheme

    modsel = args.modsel
    method = args.method
    prefix = args.prefix

    if args.mask:
        mask = '-mask %s' % args.mask
    else:
        mask = ''

    if args.iso:
        iso = '-iso'
    else:
        iso = ''

    # run DTI
    cmd_dti  = 'dti %s %s %s %s -response 0 -correction 0' % (dwi, scheme, prefix, mask)
    os.system(cmd_dti)

    d_par,d_perp = read_lambdas('%s_DTInolin_ResponseAnisotropic.txt' % prefix)

    # run MRDS
    cmd_mrds = 'mdtmrds %s %s %s -correction 0 -response %.12f,%.12f,0.003 %s -modsel %s -each -intermediate %s -mse -method %s' % (dwi, scheme, prefix, d_par, d_perp, mask, modsel, iso, method)
    os.system(cmd_mrds)

if __name__ == '__main__':
    main()
