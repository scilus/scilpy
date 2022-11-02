# -*- coding: utf-8 -*-

import logging
import numpy as np

from dipy.io.stateful_tractogram import StatefulTractogram

from scilpy.tractanalysis.reproducibility_measures import \
    get_endpoints_density_map
from scilpy.tractanalysis.streamlines_metrics import compute_tract_counts_map


def compute_f1_score(overlap, overreach):
    """
    Compute the F1 score between overlap and overreach (they must be
    percentages).

    Parameters
    ------
    overlap: float, The overlap value.
    overreach: float, The overreach value
        (Version normalized over bundle area, not version normalized over gt).

    Returns
    -------
    f1_score: float, The f1 score.

    Ref: https://en.wikipedia.org/wiki/F1_score
    """
    # Recall = True positive / (True positive + False negative)
    #        = |B inter A| / |A|
    #        = overlap
    recall = overlap
    # Precision = True positive / (True positive + False positive)
    #        = |B inter A| / |B|
    #        = 1 - |B except A| / |B|
    #        = 1 - overreach
    precision = 1.0 - overreach
    f1_score = 2.0 * (precision * recall) / (precision + recall)
    return f1_score


def compute_f1_overlap_overreach(current_vb_voxels, gt_mask, dimensions):
    """
    Compute f1, OL and OR/ORn based on a ground truth mask.

    Parameters
    ------
    current_vb_voxels: 3D array
        The voxels touched by at least one streamlines for a given bundle.
    gt_mask: 3D array
        The ground truth mask.
    dimensions: np.array
        The nibabel dimensions of the data (3D).

    Returns
    -------
    f1: float
        The f1 score.
    tp_nb_voxels: int
        The TP (true positive) count in number of voxels.
    fp_nb_voxels: int
        The FP (false positive) count in number of voxels.
        Hint: Divide it by the ground truth count to get the overreach, or
        by the recovered bundle count to get the ORn (scores used in the
        ismrm2015 tractography challenge).
    fn_nb_voxels: int
        The number of voxels from the gt_mask that have not been recovered;
        corresponds to the FN count (false negative).
    overlap: float
        TP divided by the ground truth count (i.e. TP + FN), in percentage.
    overreach_pct_total: float
        The overreach, normalized by the recovered bundle's area. (Or 0 if
        no streamline have been recovered for this bundle).
    overreach_pct_gt: float
        The overreach, normalized by the ground truth area.
    """
    # True positive = |B inter A|
    tp_mask = gt_mask * current_vb_voxels
    tp_nb_voxels = np.count_nonzero(tp_mask)

    # False positive = |B except A|
    fp_mask = np.zeros(dimensions)
    fp_mask[np.where(
        (gt_mask == 0) & (current_vb_voxels >= 1))] = 1
    fp_nb_voxels = np.count_nonzero(fp_mask)

    # False negative = |A except B|
    fn_mask = np.zeros(dimensions)
    fn_mask[np.where(
        (gt_mask == 1) & (current_vb_voxels == 0))] = 1
    fn_nb_voxels = np.count_nonzero(fn_mask)

    gt_total_nb_voxels = tp_nb_voxels + fn_nb_voxels
    # Same as np.count_nonzero(gt_mask)

    nb_voxels_total = tp_nb_voxels + fp_nb_voxels
    # Same as np.count_nonzero(current_vb_voxels)

    # Overlap = |B inter A| / |A|
    overlap = tp_nb_voxels / gt_total_nb_voxels

    # Overreach: two versions are sometimes used.
    # |B except A| / |A| or |B except A| / |B|
    if nb_voxels_total == 0:
        overreach_pct_total = 0
    else:
        overreach_pct_total = fp_nb_voxels / nb_voxels_total
    overreach_pct_gt = fp_nb_voxels / gt_total_nb_voxels

    # f1 score (=dice)
    f1 = compute_f1_score(overlap, overreach_pct_total)

    return (f1, tp_nb_voxels, fp_nb_voxels, fn_nb_voxels,
            overlap, overreach_pct_gt, overreach_pct_total)


def get_binary_maps(streamlines, sft):
    """
    Extract a mask from a bundle.

    Parameters
    ----------
    streamlines: list
        List of streamlines.
    sft : StatefulTractogram
        Reference tractogram.

    Returns
    -------
    bundles_voxels: numpy.ndarray
        Mask representing the bundle volume.
    endpoints_voxels: numpy.ndarray
        Mask representing the bundle's endpoints.
    """
    dimensions = sft.dimensions
    if not len(streamlines):
        return np.zeros(dimensions), np.zeros(dimensions)
    elif len(streamlines) == 1:
        streamlines = [streamlines]
    tmp_sft = StatefulTractogram.from_sft(streamlines, sft)
    tmp_sft.to_vox()
    tmp_sft.to_corner()

    if len(tmp_sft) == 1:
        return np.zeros(dimensions), np.zeros(dimensions)

    bundles_voxels = compute_tract_counts_map(tmp_sft.streamlines,
                                              dimensions).astype(np.int16)

    endpoints_voxels = get_endpoints_density_map(tmp_sft.streamlines,
                                                 dimensions).astype(np.int16)

    bundles_voxels[bundles_voxels > 0] = 1
    endpoints_voxels[endpoints_voxels > 0] = 1

    return bundles_voxels, endpoints_voxels


def compute_tractometry(all_vs_ids, all_wpc_ids, all_ic_ids, all_nc_ids,
                        vs_ids_list, wpc_ids_list, ic_ids_list,
                        vb_sft_list, wpc_sft_list, ib_sft_list, sft, args,
                        bundles_names, gt_masks, dimensions, comb_filename):

    """
    Tractometry stats: First in terms of connections (NC, IC, VS, WPC), then
    in terms of volume (OL, OR, Dice score)
    """
    logging.info("Computing tractometry")

    nb_bundles = len(bundles_names)

    # Total number of streamlines for each category
    vs_count = len(all_vs_ids)
    wpc_count = len(all_wpc_ids)
    ic_count = len(all_ic_ids)
    nc_count = len(all_nc_ids)
    total_count = len(sft)

    final_results = {
        "tractogram_filename": str(args.in_tractogram),
        "total_streamlines": total_count,
        "VB": len([x for x in vs_ids_list if len(x) > 0]),
        "VS": vs_count,
        "VS_ratio": vs_count / total_count,
        "IS": ic_count + nc_count,  # ic_count = 0 if not args.compute_ic
        "IS_ratio": (ic_count + nc_count) / total_count,
    }

    if args.compute_ic:
        final_results.update({
            "IB": len([x for x in ic_ids_list if len(x) > 0]),
            "IC": ic_count,
            "IC_ratio": ic_count / total_count,
            "NC": nc_count,
            "NC_ratio": nc_count / total_count})

    if args.save_wpc_separately:
        final_results.update({
            "WPC": wpc_count,
            "WPC_bundle": len([x for x in wpc_ids_list if len(x) > 0]),
            "WPC_ratio": wpc_count / total_count})

    # Tractometry stats over volume: OL, OR, Dice score
    mean_overlap = 0.0
    mean_overreach_gt = 0.0
    mean_overreach_n = 0.0
    mean_f1 = 0.0

    bundle_wise_dict = {}
    for i in range(nb_bundles):
        current_vb = vb_sft_list[i].streamlines
        bundle_results = {"VS": len(current_vb)}
        if gt_masks[i] is not None:
            # Getting the recovered mask
            current_vb_voxels, current_vb_endpoints_voxels = get_binary_maps(
                current_vb, sft)

            (f1, tp_nb_voxels, fp_nb_voxels, fn_nb_voxels,
             overlap, overreach_pct_gt, overreach_pct_total) = \
                compute_f1_overlap_overreach(
                    current_vb_voxels, gt_masks[i], dimensions)

            # Endpoints coverage
            # todo. What is this? Useful?
            endpoints_overlap = gt_masks[i] * current_vb_endpoints_voxels
            endpoints_overreach = np.zeros(dimensions)
            endpoints_overreach[np.where(
                (gt_masks[i] == 0) & (current_vb_endpoints_voxels >= 1))] = 1

            bundle_results.update({
                "TP": tp_nb_voxels,
                "FP": fp_nb_voxels,
                "FN": fn_nb_voxels,
                "OL": overlap,
                "OR_gt": overreach_pct_gt,
                "ORn": overreach_pct_total,
                "f1": f1,
                "endpoints_OL": np.count_nonzero(endpoints_overlap),
                "endpoints_OR": np.count_nonzero(endpoints_overreach)
            })

            # WPC
            if args.save_wpc_separately:
                wpc = wpc_sft_list[i]
                if wpc is not None and len(wpc.streamlines) > 0:
                    current_wpc_streamlines = wpc.streamlines
                    current_wpc_voxels, _ = get_binary_maps(
                        current_wpc_streamlines, sft)

                    # We could add an option to include wpc streamlines to the
                    # overreach count. But it seems more natural to exclude wpc
                    # streamlines from any count. Separating into a different
                    # statistic dict.
                    (_, tp_nb_voxels, fp_nb_voxels, _, overlap,
                     overreach_pct_gt, overreach_pct_total) = \
                        compute_f1_overlap_overreach(
                            current_vb_voxels, gt_masks[i], dimensions)

                    wpc_results = {
                        "Count": len(current_wpc_streamlines),
                        "TP": tp_nb_voxels,
                        "FP": fp_nb_voxels,
                        "OL": overlap,
                        "OR_gt": overreach_pct_gt,
                        "ORn": overreach_pct_total,
                    }
                    bundle_results.update({"WPC": wpc_results})
                else:
                    bundle_results.update({"WPC": None})

        mean_overlap += bundle_results["OL"]
        mean_overreach_gt += bundle_results["OR_gt"]
        mean_overreach_n += bundle_results["ORn"]
        mean_f1 += bundle_results["f1"]
        bundle_wise_dict.update({bundles_names[i]: bundle_results})

    if args.compute_ic:
        # -----------
        # False connections stats: number of voxels
        # -----------
        ic_results = {}
        for i, filename in enumerate(comb_filename):
            current_ib = ib_sft_list[i].streamlines

            if len(current_ib):
                current_ib_voxels, _ = get_binary_maps(current_ib, sft)

                bundle_results = {
                    "filename": filename,
                    "IC": len(current_ib),
                    "nb_voxels": np.count_nonzero(current_ib_voxels)
                }
                ic_results.update({str(filename): bundle_results})

        bundle_wise_dict.update({"IB": ic_results})

    final_results.update({
        "bundle_wise": bundle_wise_dict,
        "mean_OL": mean_overlap / nb_bundles,
        "mean_OR_gt": mean_overreach_gt / nb_bundles,
        "mean_ORn": mean_overreach_n / nb_bundles,
        "mean_f1": mean_f1 / nb_bundles
    })

    return final_results
