# -*- coding: utf-8 -*-
"""
Computing ihMT and non-ihMT maps
Make a more details description !
"""
#verifier si import dans le bon ordre...
import argparse
from bids import BIDSLayout
import json
import os
import math

import numpy as np
import nibabel as nib
from pathlib import Path
import scipy.ndimage

from scilpy.io.utils import (add_overwrite_arg,
                             assert_inputs_exist, assert_outputs_exist)


def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=args.RawTextHelpFormatter)
    p.add_argument('input', metavar='input',
                    help='Path of the ihMT maps from bids folder')
    p.add_argument('output', metavar='output',
                    help='Path to output folder')
    p.add_argument('subjid', metavar='subjid',
                    help='Id of subject from BIDS folder')
    p.add_argument('brainmask', metavar='input',
                    help='Path to T1 binary brainmask.'
                    'Generate by BET or other tools')
    p.add_argument('--resolution', action='store_true', metavar='string',
                    help="Give resolution ihmt: 'highres' or 'lowres'")
    p.add_argument('--filtering', action='store_true',
                    help='Gaussian filtering to remove Gibbs ringing,'
                    'sigma set at 0.5. Not recommanded. Default is None')

    add_overwrite_arg(p)

    return p


def merge_array(image, nb_echoes):
    """
    Function to concatenate n 3D-array matrix of the same size
    along the 4th dimension.
    Return a matrix (x,y,z,n) for n matrices (x,y,z) as input
    """
    merge_array=[]
    for k in range(0, nb_echoes):
        nii_image = nib.load(image[k])
        nii_array = np.array(nii_image.dataobj)
        merge_array.append(nii_array)
        #merge_array.append(np.array(nii_image.dataobj))
    merge_array = np.stack(merge_array, axis=-1)
    return merge_array


def py_fspecial_gauss(shape,sigma):
    """
    Function to mimic the 'fspecial gaussian' MATLAB function
    Returns a rotationally symmetric Gaussian lowpass filter of
    shape size with standard deviation sigma.
    see https://www.mathworks.com/help/images/ref/fspecial.html
    """
    m,n = [(ss-1.)/2. for ss in shape]
    y,x = np.ogrid[-m:m+1,-n:n+1]
    h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
    h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h

def threshold_by_intensity(computed_map,lower_threshold,upper_thresold):
    """
    Set at 0 values corresponding to lower and upper thresold and remove NaN
    """
    computed_map[isnan(computed_map)]=0
    computed_map[isinf(computed_map)]=0
    computed_map[computed_map<lower_threshold]=0
    computed_map[computed_map>upper_threshold]=0
    return computed_map

def threshold_by_contrast_maps(computed_map, contrast_maps_list):
    """
    Apply threshold based on a combination of specific contrasts maps
    Parameters
    ----------
    computed_map : name of ihmt computed maps
    contrast_maps_list : List of contrasts maps
       example : ['Positive, Negative']
    """
    for contrast_maps in contrast_maps_list:
        computed_map[contrast_maps==0]=0
    return computed_map

def save_maps(ref_image, data, output_name, resoltuion):
    ref_img = nib.load(ref_imgage)
    affine = ref_img.get_affine()
    hdr = nib.Nifti1Header()
    hdr.set_slope_inter(1.0, 0.0)
    hdr['glmax']=data.max()
    hdr['glmin']=data.min()
    img = nib.Nifti1Image(data.astype(np.float64), affine, hdr)
    if resolution is not None:
        nib.save(img,os.path.join(args.ouput,
                                  output_name +'_'+ resolution +'.nii'))
    else:
        nib.save(img,os.path.join(args.ouput,
                                  output_name +'.nii'))


def compute_contrast(contrast_maps, ref_image, arg_filtering=None):
    """ Return contrast map

    Parameters
    ----------
    contrast_map : List of BIDSFile object
        List of contrast maps
    ref_image : Give reference images to set header
    arg_filtering : Apply filtering. Default is None.

    Returns
    -------
    Contrast map in nifti format
    Contrast map into matrix format
    """

    resulting_map=[]

    # Merged different echo into 4d-array
    merged_map=merge_array(contrasts,NbEchoes)

    # Compute the contrast map
    contrast_map=np.sqrt(np.sum(np.squeeze(merged_map*merged_map),3))

    #Apply gaussian filtering if needed
    if args.filtering is not None:
        contrast_map=scipy.ndimage.convolve(contrasts,h).astype(float)

    resulting_map.append(contrast_map)

    # Save contrast maps in nii.gz format
    save_maps(positive_maps[0], contrast_map, contrast_map_name, acq)

    return resulting_map


# Define filtering parameters mainly to remove Gibbs ringing
    if args.filtering:
        h=py_fspecial_gauss((3,3), 0.5)

#Set constante
NbEchoes=3 # We may not want to include all echoes, depending on echo time/SNR
Contrast=5 # We know that there are 5 contrasts for ihMT


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    #Verify existence of input and output
    assert_inputs_exist(parser, [args.input,
                                 args.brainmask,
                                 args.subjid])
    assert_outputs_exist(parser,[args.output])

    ## Select Contrasts files
    acq = args.resolution
    layout = BIDSLayout(args.input, validate=False)  #(derivatives=True ? regle probleme ou pas)
        positive_maps = layout.get(subject=args.subjid,
                          datatype='anat',suffix='ihmt', aquisition='pos'+acq,
                          extension='nii.gz',return_type='file')
        negative_maps = layout.get(subject=args.subjid,
                          datatype='anat',suffix='ihmt', aquisition='neg'+acq,
                          extension='nii.gz',return_type='file')
        altpn_maps = layout.get(subject=args.subjid,
                          datatype='anat',suffix='ihmt', aquisition='altpn'+acq,
                          extension='nii.gz', return_type='file')
        altnp_maps = layout.get(subject=args.subjid,
                          datatype='anat',suffix='ihmt',aquisition='altnp'+acq,
                          extension='nii.gz', return_type='file')
        reference_maps = layout.get(subject=args.subjid,
                          datatype='anat',suffix='ihmt', aquisition='T1w'+acq,
                          extension='nii.gz', return_type='file')
        T1w_maps = layout.get(subject=args.subjid,
                          datatype='anat',suffix='ihmt',acquisition='mtoff'+acq,
                          extension='nii.gz', return_type='file')

    #set parameters of ihMT sequence
    TR_MT = layout.get_metadata(positive_maps[0])['RepetitionTime']
    TR_MT=TR_MT*1000
    FlipAMT = layout.get_metadata(positive_maps[0])['FlipAngle']
    FlipAngleMT=(FlipAMT*math.pi/180)

    #set parameters of T1w sequence
    TR_T1w = layout.get_metadata(T1w_maps[0])['RepetitionTime']
    TR_T1w = TR_T1w*1000
    FlipA = layout.get_metadata(T1w_maps[0])['FlipAngle']
    FlipAngleT1w = (FlipAT1*math.pi/180)

##### a voir si laisse pour le moment, besoin de tester
    # Set Parameters
    #TotalEchoes=int(len(allniigz)/Contrast)
    #NbEchoes=min(NbEchoes,TotalEchoes)

    #Load brainmask
    mask_image = nib.load(args.brainmask)
    mask = np.array(mask_image.dataobj)

    #Fix issue from the presence of NaN into array
    np.seterr(divide='ignore', invalid='ignore')

    ##Compute contrasts maps
    positive = compute_contrast(positive_maps, positive_maps[0])
    negative = compute_contrast(negative_maps, positive_maps[0])
    altpn = compute_contrast(altpn_maps, positive_maps[0])
    altnp = compute_contrast(altnp_maps, positive_maps[0])
    reference = compute_contrast(reference_maps, positive_maps[0])
    T1w = compute_contrast(T1w_maps, positive_maps[0])

    ##Compute ihMT maps
    #ihMTratio
    ihMTRatio=100*(positive+negative-altPN-altNP)/reference
    threshold_by_intensity(ihMTRatio, 0, 100)
    ihMTRatio=ihMTRatio*mask
    contrast_lists=['Positive','Negative','AltPN','reference']
    threshold_by_contrast_map(ihMTRatio, contrast_lists)

    #save image
    save_maps(positive_maps[0], ihMTRatio, 'ihMTRatio', acq)

    # Compute an dR1sat image (Varma et al. ISMRM 2015)
    cPD1a=(positive+negative)/2
    cPD1b=(altPN+altNP)/2
    cT1=T1w
    T1appa=((cPD1a/FlipAngleMT)-(cT1/FlipAngleT1))/((cT1*FlipAngleT1)/
            (2*TR_T1/1000)-(cPD1a*FlipAngleMT)/(2*TR_MT/1000))
    T1appb=((cPD1b/FlipAngleMT)-(cT1/FlipAngleT1))/((cT1*FlipAngleT1)/
            (2*TR_T1/1000)-(cPD1b*FlipAngleMT)/(2*TR_MT/1000))
    ihMTsat=(1/T1appb)-(1/T1appa)

    threshold_by_intensity(ihMTsat, 0, 10)
    ihMTsat=ihMTsat*mask
    contrast_lists=['positive','negative','altPN','altNP']
    threshold_by_contrast_map(ihMTsat, contrast_lists)

    #save image
    save_maps(positive_maps[0], ihMTsat, 'ihMT_dR1sat', acq)


    ##Compute non-ihMT maps
    # Compute MTR
    MTR=(reference-(positive+negative)/2)/Reference
    MTR=MTR*100
    threshold_by_intensity(MTR, 0, 100)
    contrast_lists=['positive','reference']
    threshold_by_contrast_map(MTR, contrast_lists)
    #save image
    save_maps(positive_maps[0], MTR, 'MTR', acq)

    # Compute MTsat
    cPD1=reference
    cPD2=(positive+negative)/2
    cT1=T1w

    Aapp=((2*TR_MT/(FlipAngleMT*FlipAngleMT))-(2*TR_T1/(FlipAngleT1*FlipAngleT1)))
          /(((2*TR_MT)/(FlipAngleMT*cPD1))-((2*TR_T1)/(FlipAngleT1*cT1)))
    T1app=((cPD1/FlipAngleMT)-(cT1/FlipAngleT1))/((cT1*FlipAngleT1)/
           (2*TR_T1)-(cPD1*FlipAngleMT)/(2*TR_MT))
    MTsat=((Aapp*FlipAngleMT*TR_MT/T1app)/cPD2)-(TR_MT/T1app)-
           (FlipAngleMT*FlipAngleMT)/2

    MTsat=MTsat*100
    threshold_by_intensity(MTsat, 0, 100)
    MTsat=MTsat*mask
    contrast_lists=['positive','reference']
    threshold_by_contrast_map(MTsat, contrast_lists)

    #save image
    save_maps(positive_maps[0], MTsat, 'MTsat', acq)


if __name__ == '__main__':
    main()
