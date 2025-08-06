#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
BundleSeg output visualization tool.

This script allows visualization of segmented bundles and after-the-fact
filtering to explore optimal thresholds for each bundle. It uses
FURY for 3D rendering.
It includes tools to slice images, filter streamlines, and interactively
visualize bundles and a reference image.

This script is intended to be used after running BundleSeg:
    scil_tractogram_segment_with_bundleseg.py
To explore thresholds the option --exploration_mode must be used to segment
using a much higher distance threshold.
"""

import argparse
import glob
import json
import logging
import os
import numpy as np


from dipy.io.streamline import load_tractogram, save_tractogram
from dipy.io.stateful_tractogram import StatefulTractogram, Space
import fury
from fury import window
from fury.data import read_viz_icons
import nibabel as nib

from scilpy.io.image import get_data_as_mask
from scilpy.io.utils import (add_overwrite_arg,
                             add_verbose_arg,
                             assert_inputs_exist,
                             assert_output_dirs_exist_and_empty, )
pil_logger = logging.getLogger('PIL')
pil_logger.setLevel(logging.WARNING)


def update_scene():
    """
    Update the scene with the current index and threshold.
    Renders the streamlines for the current index and threshold based on the
    filtering.
    """
    global current_index, mapping, sft, stream_actor, scene, filter_slider, \
        bundle_name_textbox
    try:
        # Remove previous streamline actor if it exists
        scene.rm(stream_actor)
    except NameError:
        pass

    key = list(mapping.keys())[current_index]
    indices = np.array(mapping[key]['indices'], dtype=int)
    scores = np.array(mapping[key]['scores'], dtype=float)

    # Filter the streamlines based on the threshold value from the slider
    thr = np.round(filter_slider.value, 2)
    indices = indices[np.where(scores < thr)]
    streamlines = sft.streamlines[indices]

    # Check if any streamlines remain after filtering
    if len(streamlines) == 0:
        print(f'No streamlines found for {key}')
    else:
        # Add the filtered streamlines to the scene
        stream_actor = fury.actor.line(list(streamlines))
        scene.add(stream_actor)

    # Update the bundle name display
    bundle_name_textbox.message = key


def build_label(text, font_size=18, justification='left'):
    """
    Utility function to create a TextBlock2D label with the provided text.
    """
    label = fury.ui.TextBlock2D()
    label.message = text
    label.font_size = font_size
    label.font_family = 'Arial'
    label.justification = justification
    label.bold = False
    label.italic = False
    label.shadow = False
    label.background_color = (0, 0, 0)
    label.color = (1, 1, 1)

    return label


def setup(sft, ref_img, mapping, mask):
    """
    Set up the visualization scene, including UI elements and event handlers.
    This function adds slicers, filters, and other UI components to interact
    with the 3D data.

    Parameters:
    - sft: StatefulTractogram, containing the streamlines to be visualized.
    - ref_img: NIfTI reference image to serve as background.
    - mapping: Dictionary containing bundle information.
    - mask: Optional mask image to highlight regions of interest.
    """
    global current_index, filter_slider, scene, size, dropdown, \
        bundle_name_textbox, image_actor_x, image_actor_y, image_actor_z

    data = ref_img.get_fdata()
    shape = ref_img.shape
    affine = ref_img.affine

    # This must be done in RASMM
    min_x, min_y, min_z = 0, 0, 0
    vector = np.array([min_x, min_y, min_z, 1])
    min_x, min_y, min_z = np.dot(affine, vector)[:3].astype(int)

    max_x, max_y, max_z = shape[0], shape[1], shape[2]
    vector = np.array([max_x, max_y, max_z, 1])
    max_x, max_y, max_z = np.dot(affine, vector)[:3].astype(int)

    min_x, max_x = np.min([min_x, max_x]), np.max([min_x, max_x])
    min_y, max_y = np.min([min_y, max_y]), np.max([min_y, max_y])
    min_z, max_z = np.min([min_z, max_z]), np.max([min_z, max_z])

    # Calculate the center slices for each axis
    center_x = int(np.round((max_x - min_x) / 2))
    center_y = int(np.round((max_y - min_y) / 2))
    center_z = int(np.round((max_z - min_z) / 2))

    # Initialize the main scene for 3D rendering
    scene = window.Scene()
    image_actor_x = fury.actor.slicer(data.copy(), affine=affine)
    image_actor_x.display(x=center_x)
    scene.add(image_actor_x)

    image_actor_y = fury.actor.slicer(data.copy(), affine=affine)
    image_actor_y.display(y=center_y)
    scene.add(image_actor_y)

    image_actor_z = fury.actor.slicer(data.copy(), affine=affine)
    image_actor_z.display(z=center_z)
    scene.add(image_actor_z)

    # Add the surface representation of the mask if provided
    if mask is not None:
        surface_actor = fury.actor.contour_from_roi(mask, affine=affine,
                                                    color=[1.0, 1.0, 1.0],
                                                    opacity=1.0)
        scene.add(surface_actor)
    else:
        surface_actor = None

    # ------------------- Slicing UI START -------------------
    # Slicers for X, Y, Z axes and opacity control
    line_slider_x = fury.ui.LineSlider2D(
        min_value=0, max_value=max_x, initial_value=center_x,
        text_template='{value:.0f}', length=140)

    line_slider_y = fury.ui.LineSlider2D(
        min_value=0, max_value=max_y, initial_value=center_y,
        text_template='{value:.0f}', length=140)

    line_slider_z = fury.ui.LineSlider2D(
        min_value=0, max_value=max_z, initial_value=center_z,
        text_template='{value:.0f}', length=140)

    opacity_slider = fury.ui.LineSlider2D(
        min_value=0.0, max_value=1.0, initial_value=0.6, length=140)

    # Callbacks for updating slice positions and opacity when sliders are moved
    def change_slice_x_callback(slider):
        global image_actor_x
        x = int(np.round(slider.value))
        image_actor_x.display(x=x)

    def change_slice_y_callback(slider):
        global image_actor_y
        y = int(np.round(slider.value))
        image_actor_y.display(y=y)

    def change_slice_z_callback(slider):
        global image_actor_z
        z = int(np.round(slider.value))
        image_actor_z.display(z=z)

    def change_opacity_callback(slider):
        slicer_opacity = slider.value
        image_actor_x.opacity(slicer_opacity)
        image_actor_y.opacity(slicer_opacity)
        image_actor_z.opacity(slicer_opacity)

    # Attach callbacks to sliders
    line_slider_z.on_change = change_slice_z_callback
    line_slider_x.on_change = change_slice_x_callback
    line_slider_y.on_change = change_slice_y_callback
    opacity_slider.on_change = change_opacity_callback

    # Labels for each slicer
    line_slider_label_x = build_label(text='X Slice')
    line_slider_label_y = build_label(text='Y Slice')
    line_slider_label_z = build_label(text='Z Slice')
    opacity_slider_label = build_label(text='Opacity')
    # ------------------- Slicing UI END -------------------

    # ------------------- Filtering UI START -------------------
    # Create filter slider for streamline filtering based on scores
    all_scores = []
    for key in list(mapping.keys()):
        basename = os.path.basename(key)
        all_scores.extend(mapping[key]['scores'])
        mapping[basename] = mapping.pop(key)
    max_score_value = np.max(all_scores)
    median_score_value = np.median(all_scores)

    filter_slider = fury.ui.LineSlider2D(min_value=0, max_value=max_score_value,
                                         initial_value=median_score_value,
                                         length=140)
    filter_slider.active_color = (1, 1, 1)

    # Callback for updating the scene when the filter slider is released
    def filter_release_callback(i_ren, _vtk_actor, _):
        global filter_slider
        update_scene()
        i_ren.force_render()

    # Callback for updating filter slider while being dragged
    def filter_drag_callback(i_ren, _vtk_actor, _):
        global filter_slider
        filter_slider.handle.color = filter_slider.active_color
        position = i_ren.event.position
        filter_slider.set_position(position)
        filter_slider.on_moving_slider(filter_slider)
        i_ren.force_render()
        i_ren.event.abort()  # Stop propagating the event.

    # Attach callbacks to the filter slider
    filter_slider.track.on_left_mouse_button_dragged = filter_drag_callback
    filter_slider.track.on_left_mouse_button_released = filter_release_callback
    filter_slider.handle.on_left_mouse_button_dragged = filter_drag_callback
    filter_slider.handle.on_left_mouse_button_released = filter_release_callback

    filter_slider_label = build_label(text='Filtering')
    # ------------------- Filtering UI END -------------------

    # ------------------- Selection UI START -------------------
    # UI elements for selecting different bundles
    name_list = list(mapping.keys())
    current_index = 0

    def next_callback(i_ren, _obj, _button):
        """Move to the next bundle in the list."""
        global current_index, dropdown
        current_index = (current_index + 1) % len(name_list)
        update_scene()
        dropdown._selection = name_list[current_index]
        i_ren.force_render()
        logging.debug(f'Next: {name_list[current_index]}')

    def prev_callback(i_ren, _obj, _button):
        """Move to the previous bundle in the list."""
        global current_index, dropdown
        current_index = (current_index - 1) % len(name_list)
        update_scene()
        i_ren.force_render()
        logging.debug(f'Previous: {name_list[current_index]}')

    # Create buttons for navigation
    next_button = fury.ui.Button2D(
        [('prev', read_viz_icons(fname='arrow-right.png'))])
    prev_button = fury.ui.Button2D(
        [('next', read_viz_icons(fname='arrow-left.png'))])
    next_button.on_left_mouse_button_clicked = next_callback
    prev_button.on_left_mouse_button_clicked = prev_callback

    # Dropdown for bundle selection
    def dropdown_callback(combo):
        """Select a bundle from the dropdown menu."""
        global current_index
        current_text = combo.selected_text
        current_index = name_list.index(current_text)
        update_scene()
        logging.debug(f'Selected: {name_list[current_index]}')

    dropdown = fury.ui.ComboBox2D(items=name_list, draggable=False)
    dropdown.on_change = dropdown_callback
    # ------------------- Selection UI END -------------------

    # ------------------- Saving UI START -------------------
    # UI elements for saving filtered results
    def save_all_callback(i_ren, _obj, _button):
        """Apply the filtering to all bundles and save the results."""
        for key in list(mapping.keys()):
            indices = np.array(mapping[key]['indices'], dtype=int)
            scores = np.array(mapping[key]['scores'], dtype=float)

            thr = np.round(filter_slider.value, 2)
            indices = indices[np.where(scores < thr)]
            new_sft = sft[indices]
            new_filename = os.path.join(out_dir, f'{key}_filtered.trk')
            save_tractogram(new_sft, new_filename)
        logging.info(
            f'Saved all filtered bundles at threshold {thr} to {out_dir}')

    def save_current_callback(i_ren, _obj, _button):
        """Save the currently selected bundle with the applied filtering."""
        key = list(mapping.keys())[current_index]
        indices = np.array(mapping[key]['indices'], dtype=int)
        scores = np.array(mapping[key]['scores'], dtype=float)

        thr = np.round(filter_slider.value, 2)
        indices = indices[np.where(scores < thr)]
        new_sft = sft[indices]
        new_filename = os.path.join(out_dir, f'{key}_filtered.trk')

        logging.info(f'Saved {key} with {len(new_sft.streamlines)} streamlines '
                     f'at threshold {thr} to {new_filename}')
        save_tractogram(new_sft, new_filename)

    # Create buttons for saving results
    save_all_button = fury.ui.Button2D(
        [('Save All', read_viz_icons(fname='database.png'))])
    save_all_button.on_left_mouse_button_clicked = save_all_callback

    save_current_button = fury.ui.Button2D(
        [('Save Current', read_viz_icons(fname='floppy-disk.png'))])
    save_current_button.on_left_mouse_button_clicked = save_current_callback

    save_all_label = build_label(text='Save\nAll', font_size=12,
                                 justification='center')
    save_current_label = build_label(text='Save\nCurrent', font_size=12,
                                     justification='center')
    # ------------------- Saving UI END -------------------

    # ------------------- Checkboxes UI START -------------------
    # UI elements to toggle visibility of anatomy slices and surfaces
    def toggle_anatomy(btn):
        """Show or hide anatomical slices based on checkbox state."""
        if len(btn.checked_labels) > 0:
            image_actor_x.SetVisibility(1)
            image_actor_y.SetVisibility(1)
            image_actor_z.SetVisibility(1)
        else:
            image_actor_x.SetVisibility(0)
            image_actor_y.SetVisibility(0)
            image_actor_z.SetVisibility(0)

    def toggle_surfaces(btn):
        """Show or hide the surface actor based on checkbox state."""
        if surface_actor is not None:
            if len(btn.checked_labels) > 0:
                surface_actor.GetProperty().SetOpacity(1.0)
            else:
                surface_actor.GetProperty().SetOpacity(0.0)

    # Create checkboxes for toggling visibility
    anatomy_button = fury.ui.Checkbox(
        labels=['Show/Hide Anatomy'], checked_labels=['Show/Hide Anatomy'])
    anatomy_button.on_change = toggle_anatomy
    surface_button = fury.ui.Checkbox(
        labels=['Show/Hide Surfaces'], checked_labels=['Show/Hide Surfaces'])
    surface_button.on_change = toggle_surfaces
    # ------------------- Checkboxes UI END -------------------

    # Add elements to the UI panels
    panel_slicers = fury.ui.Panel2D(
        size=(300, 200), color=(1, 1, 1), opacity=0.1, align='right')

    panel_slicers.center = (1030, 120)
    panel_slicers.add_element(line_slider_label_x, (0.1, 0.75))
    panel_slicers.add_element(line_slider_x, (0.38, 0.75))
    panel_slicers.add_element(line_slider_label_y, (0.1, 0.55))
    panel_slicers.add_element(line_slider_y, (0.38, 0.55))
    panel_slicers.add_element(line_slider_label_z, (0.1, 0.35))
    panel_slicers.add_element(line_slider_z, (0.38, 0.35))
    panel_slicers.add_element(opacity_slider_label, (0.1, 0.15))
    panel_slicers.add_element(opacity_slider, (0.38, 0.15))

    # Panel for filtering, bundle selection, and saving
    panel_options = fury.ui.Panel2D(
        size=(300, 300), color=(1, 1, 1), opacity=0.1, align='right')

    panel_options.center = (1030, 720)
    panel_options.add_element(anatomy_button, (0.1, 0.8))
    panel_options.add_element(surface_button, (0.1, 0.7))
    panel_options.add_element(filter_slider_label, (0.1, 0.5))
    panel_options.add_element(filter_slider, (0.38, 0.5))
    panel_options.add_element(save_current_button, (0.35, 0.25))
    panel_options.add_element(save_all_button, (0.55, 0.25))
    panel_options.add_element(next_button, (0.70, 0.25))
    panel_options.add_element(prev_button, (0.20, 0.25))
    panel_options.add_element(save_current_label, (0.40, 0.15))
    panel_options.add_element(save_all_label, (0.60, 0.15))
    panel_options.add_element(dropdown, (0.05, 0.05))

    # Display the name of the current bundle
    bundle_name_textbox = fury.ui.TextBlock2D(bold=True, position=(100, 90))
    bundle_name_textbox.message = name_list[current_index]

    # Add elements to the scene
    show_m = window.ShowManager(scene=scene, title='BundleSeg',
                                size=(1200, 900))
    show_m.scene.add(panel_slicers)
    show_m.scene.add(panel_options)
    show_m.scene.add(bundle_name_textbox)

    size = scene.GetSize()

    def win_callback(obj, _event):
        """Callback for handling window resizing events."""
        global size
        if size != obj.GetSize():
            size_old = size
            size = obj.GetSize()
            size_change = [size[0] - size_old[0], 0]
            panel_options.center = [panel_options.center[0] + size_change[0],
                                    panel_options.center[1]]
            panel_slicers.center = [panel_slicers.center[0] + size_change[0],
                                    panel_slicers.center[1]]

    # Adjust camera and set the initial view
    scene.zoom(1.5)
    scene.reset_clipping_range()

    # Render initial setup
    update_scene()

    # Start interaction
    show_m.add_window_callback(win_callback)
    show_m.start()


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('in_folder',
                   help='Folder generated by BundleSeg containing the results.')
    p.add_argument('in_ref_img',
                   help='Background reference image.')
    p.add_argument('out_dir',
                   help='Output directory for filtered bundles.')
    p.add_argument('--mask',
                   help='Binary mask used to render surfaces (VOI).')

    p.add_argument('--apply_config', metavar='JSON',
                   help='Apply a configuration file to automatically filter.')

    add_overwrite_arg(p)
    add_verbose_arg(p)

    return p


def main():
    """
    Main entry point of the script.
    Parses arguments, sets up logging, and calls the setup function.
    """
    parser = _build_arg_parser()
    args = parser.parse_args()
    args.verbose = 'INFO' if args.verbose != 'DEBUG' else 'DEBUG'
    logging.getLogger().setLevel(logging.getLevelName(args.verbose))

    global sft, ref_img, mapping, mask, out_dir
    # Ensure output directories exist and inputs are available
    assert_output_dirs_exist_and_empty(parser, args, [args.out_dir])
    assert_inputs_exist(parser, args.in_ref_img)
    out_dir = args.out_dir

    # Load reference image and mappinguration
    ref_img = nib.load(args.in_ref_img)
    with open(os.path.join(args.in_folder, 'results.json')) as f:
        mapping = json.load(f)

    # Load and merge streamline data from all bundles
    streamlines = []
    offset = 0
    count = 0
    for bundle in mapping.keys():
        filename = glob.glob(f'{os.path.join(args.in_folder, bundle)}.t?k')[0]

        if not os.path.exists(filename):
            logging.warning(f'File {filename} not found.')
            continue
        count += 1

        tmp_sft = load_tractogram(filename, ref_img)
        mapping[bundle]['indices'] = np.arange(offset,
                                               offset + len(tmp_sft.streamlines))
        offset += len(tmp_sft.streamlines)
        streamlines.extend(tmp_sft.streamlines)

        logging.info(f'Loaded {bundle} with {len(tmp_sft.streamlines)} '
                     'streamlines.')

    logging.info(f'Loaded {len(streamlines)} streamlines in total from {count} '
                 f'files.\n')
    sft = StatefulTractogram(streamlines, ref_img, space=Space.RASMM)

    # Load mask if provided
    if args.mask:
        mask_img = nib.load(args.mask)
        mask = get_data_as_mask(mask_img)
    else:
        mask = None

    if args.apply_config:
        with open(args.apply_config) as f:
            config = json.load(f)
        for key in config.keys():
            key_ext = os.path.splitext(key)[0]
            indices = np.array(mapping[key_ext]['indices'], dtype=int)
            scores = np.array(mapping[key_ext]['scores'], dtype=float)
            thr = config[key]
            indices = indices[np.where(scores < thr)]
            new_sft = sft[indices]
            new_filename = os.path.join(out_dir, f'{key}_filtered.trk')
            save_tractogram(new_sft, new_filename)
        logging.info(
            f'Saved all filtered bundles at threshold {thr} to {out_dir}')
        return

    # Set up and start the visualization
    setup(sft, ref_img, mapping, mask)


if __name__ == "__main__":
    main()
