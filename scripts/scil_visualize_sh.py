#!/usr/bin/env python
# -*- coding: utf-8 -*-

import nibabel as nib
from dipy.core.sphere import disperse_charges, Sphere, HemiSphere
from dipy.reconst.shm import sh_to_sf
import numpy as np
from fury import window, actor

filename = '/home/local/USHERBROOKE/poic2312/documents/data/toy_data/two_fibers_todi_sh.nii'

def get_sphere(n_pts):
    half_n_pts = int(n_pts/2)
    theta = np.pi * np.random.rand(half_n_pts)
    phi = 2 * np.pi * np.random.rand(half_n_pts)
    hemisphere, _ = disperse_charges(HemiSphere(theta=theta, phi=phi), 5000)
    return Sphere(xyz=np.vstack((hemisphere.vertices, -hemisphere.vertices)))

def main():
    print('Executing scil_visualize_sh script')
    img = nib.nifti1.load(filename)
    print('Loaded NIfTI1 image')
    data = img.get_fdata()
    print('Shape:', img.shape)
    print(data[0, 0, 0, :])
    print('NON ZEROS:', np.count_nonzero(data))

    sph_gtab = get_sphere(128)
    sfs = sh_to_sf(data, sph_gtab, sh_order=8)

    tods = []
    k = 0
    for i in range(sfs.shape[0]):
        for j in range(sfs.shape[1]):
            sf = sfs[j, i, k, :]
            if np.count_nonzero(sf):
                tods.append(actor.odf_slicer(sf[None, None, None, :], sphere=sph_gtab, colormap='jet'))
            else:
                tods.append(actor.dots(np.array([0.0, 0.0, 0.0]), dot_size=0.1)) 

    grid = actor.grid(tods, dim=(sfs.shape[0], sfs.shape[1]), cell_shape='square')

    scene = window.Scene()

    scene.add(grid)
    window.record(scene, out_path='visualize-todi.png')


if __name__ == '__main__':
    main()