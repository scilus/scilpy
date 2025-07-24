# -*- coding: utf-8 -*-

"""
Tractometry
-----------
Global connectivity metrics:

- Computed by default:
    - VS: valid streamlines, belonging to a bundle (i.e. respecting all the
        criteria for that bundle; endpoints, limit_mask, gt_mask.).
    - IS: invalid streamlines. All other streamlines. IS = IC + NC.

- Optional:
    - WPC: wrong path connections, streamlines connecting correct ROIs but not
        respecting the other criteria for that bundle. Such streamlines always
        exist but they are only saved separately if specified in the options.
        Else, they are merged back with the IS.
        By definition. WPC are only computed if "limits masks" are provided.
    - IC: invalid connections, streamlines joining an incorrect combination of
        ROIs. Use carefully, quality depends on the quality of your ROIs and no
        analysis is done on the shape of the streamlines.
    - NC: no connections. Invalid streamlines minus invalid connections.

- Fidelity metrics:
    - OL: Overlap. Percentage of ground truth voxels containing streamline(s)
        for a given bundle.
    - OR: Overreach. Amount of voxels containing streamline(s) when they
        shouldn't, for a given bundle. We compute two versions :
        OR_pct_vs = divided by the total number of voxel covered by the bundle.
        (percentage of the voxels touched by VS).
        Values range between 0 and 100%. Values are not defined when we
        recovered no streamline for a bundle, but we set the OR_pct_vs to 0
        in that case.
        OR_pct_gt = divided by the total size of the ground truth bundle mask.
        Values could be higher than 100%.
    - f1 score: which is the same as the Dice score.
"""

import logging

import numpy as np

from scilpy.tractanalysis.streamlines_metrics import compute_tract_counts_map

from scilpy.tractograms.streamline_and_mask_operations import \
    get_endpoints_density_map


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
    # In the case where overlap = 0 (found the bundle but entirely out of the
    # mask; overreach = 100%), we avoid division by 0 and define f1 as 0.
    if overlap == 0 and overreach == 1:
        return 0.

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
    overreach_pct_gt: float
        The overreach, normalized by the ground truth area.
    overreach_pct_vs: float
        The overreach, normalized by the recovered bundle's area. (Or 0 if
        no streamline have been recovered for this bundle).
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
        overreach_pct_vs = 0
    else:
        overreach_pct_vs = fp_nb_voxels / nb_voxels_total
    overreach_pct_gt = fp_nb_voxels / gt_total_nb_voxels

    # f1 score (=dice)
    f1 = compute_f1_score(overlap, overreach_pct_vs)

    return (f1, tp_nb_voxels, fp_nb_voxels, fn_nb_voxels,
            overlap, overreach_pct_gt, overreach_pct_vs)


def get_binary_maps(sft):
    """
    Extract a mask from a bundle.

    Parameters
    ----------
    sft: StatefulTractogram
        Bundle.

    Returns
    -------
    bundles_voxels: numpy.ndarray
        Mask representing the bundle volume.
    endpoints_voxels: numpy.ndarray
        Mask representing the bundle's endpoints.
    """
    sft.to_vox()
    sft.to_corner()
    _, dimensions, _, _ = sft.space_attributes

    if len(sft) == 0:
        return np.zeros(dimensions), np.zeros(dimensions)

    bundles_voxels = compute_tract_counts_map(sft.streamlines,
                                              dimensions).astype(np.int16)

    endpoints_voxels = get_endpoints_density_map(sft).astype(np.int16)

    bundles_voxels[bundles_voxels > 0] = 1
    endpoints_voxels[endpoints_voxels > 0] = 1

    return bundles_voxels, endpoints_voxels


def compute_tractometry(
        vb_sft_list, wpc_sft_list, ib_sft_list, nc_sft, args,
        bundles_names, gt_masks, dimensions, ib_names):
    """
    Tractometry stats: First in terms of connections (NC, IC, VS, WPC), then
    in terms of volume (OL, OR, Dice score)
    """
    logging.info("Computing tractometry")

    vs_per_bundle = [len(x) if x is not None else 0 for x in vb_sft_list]
    vb_count = np.count_nonzero(vs_per_bundle)
    vs_count = np.sum(vs_per_bundle)

    if wpc_sft_list is not None:
        wpc_per_bundle = [len(x) if x is not None else 0 for x in wpc_sft_list]
        wpb_count = np.count_nonzero(wpc_per_bundle)
        wpc_count = np.sum(wpc_per_bundle)
    else:
        wpb_count = 0
        wpc_count = 0

    ic_per_ib_bundle = [len(x) for x in ib_sft_list]
    ib_count = np.count_nonzero(ic_per_ib_bundle)
    ic_count = np.sum(ic_per_ib_bundle)

    nc_count = len(nc_sft) if nc_sft is not None else 0
    total_count = vs_count + wpc_count + ic_count + nc_count

    nb_bundles = len(bundles_names)

    final_results = {
        "total_streamlines": int(total_count),
        "VB": int(vb_count),
        "VS": int(vs_count),
        "VS_ratio": vs_count / total_count,
        "IS": int(ic_count + nc_count),  # ic_count = 0 if not args.compute_ic
        "IS_ratio": (ic_count + nc_count) / total_count,
    }

    if args.compute_ic:
        final_results.update({
            "IB": int(ib_count),
            "IC": int(ic_count),
            "IC_ratio": ic_count / total_count,
            "NC": int(nc_count),
            "NC_ratio": nc_count / total_count})

    if args.save_wpc_separately:
        final_results.update({
            "WPC": int(wpc_count),
            "WPC_bundle": wpb_count,
            "WPC_ratio": wpc_count / total_count})

    # Tractometry stats over volume: OL, OR, Dice score
    mean_overlap = 0.0
    mean_overreach_gt = 0.0
    mean_overreach_vs = 0.0
    mean_f1 = 0.0
    nb_bundles_in_stats = 0

    bundle_wise_dict = {}
    for i in range(nb_bundles):
        logging.debug("Scoring bundle {}".format({bundles_names[i]}))

        current_vb = vb_sft_list[i]
        bundle_results = {"VS": len(current_vb)}

        if gt_masks[i] is not None:
            if current_vb is None or len(current_vb) == 0:
                logging.debug("   Empty bundle or bundle not found.")
                bundle_results.update({
                    "TP": 0, "FP": 0, "FN": 0,
                    "OL": 0, "OR_pct_gt": 0, "OR_pct_vs": 0, "f1": 0,
                    "endpoints_OL": 0, "endpoints_OR": 0
                })
                nb_bundles_in_stats += 1
                bundle_wise_dict.update({bundles_names[i]: bundle_results})
                continue

            # Getting the recovered mask
            current_vb_voxels, current_vb_endpoints_voxels = get_binary_maps(
                current_vb)

            (f1, tp_nb_voxels, fp_nb_voxels, fn_nb_voxels,
             overlap_count, overreach_pct_gt, overreach_pct_vs) = \
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
                "OL": overlap_count,
                "OR_pct_vs": overreach_pct_vs,
                "OR_pct_gt": overreach_pct_gt,
                "f1": f1,
                "endpoints_OL": np.count_nonzero(endpoints_overlap),
                "endpoints_OR": np.count_nonzero(endpoints_overreach)
            })

            # WPC
            if args.save_wpc_separately:
                wpc_sft = wpc_sft_list[i]
                if wpc_sft is not None and len(wpc_sft) > 0:
                    current_wpc_voxels, _ = get_binary_maps(wpc_sft)

                    # We could add an option to include wpc streamlines to the
                    # overreach count. But it seems more natural to exclude wpc
                    # streamlines from any count. Separating into a different
                    # statistic dict. Else, user may simply not include a "ALL"
                    # mask, there won't be any wpc.
                    (_, tp_nb_voxels, fp_nb_voxels, _, overlap_count,
                     overreach_pct_gt, overreach_pct_vs) = \
                        compute_f1_overlap_overreach(
                            current_vb_voxels, gt_masks[i], dimensions)

                    wpc_results = {
                        "Count": len(wpc_sft),
                        "TP": tp_nb_voxels,
                        "FP": fp_nb_voxels,
                        "OL": overlap_count,
                        "OR_pct_vs": overreach_pct_vs,
                        "OR_pct_gt": overreach_pct_gt,
                    }
                    bundle_results.update({"WPC": wpc_results})
                else:
                    bundle_results.update({"WPC": None})

            mean_overlap += bundle_results["OL"]
            mean_overreach_gt += bundle_results["OR_pct_gt"]
            mean_overreach_vs += bundle_results["OR_pct_vs"]
            mean_f1 += bundle_results["f1"]
            nb_bundles_in_stats += 1
        else:
            bundle_results.update({"Scoring skipped": "No gt_mask provided"})

        bundle_wise_dict.update({bundles_names[i]: bundle_results})

    if args.compute_ic:
        # -----------
        # False connections stats: number of voxels
        # -----------
        ic_results = {}
        for i in range(len(ib_names)):
            current_ib = ib_sft_list[i]
            if len(current_ib) > 0:
                current_ib_voxels, _ = get_binary_maps(current_ib)

                bundle_results = {
                    "IC": len(current_ib),
                    "nb_voxels": np.count_nonzero(current_ib_voxels)
                }
                ic_results.update({ib_names[i]: bundle_results})

        bundle_wise_dict.update({"IB": ic_results})

    final_results.update({"bundle_wise": bundle_wise_dict})

    if nb_bundles_in_stats > 0:
        final_results.update({
            "mean_OL": mean_overlap / nb_bundles_in_stats,
            "mean_OR_gt": mean_overreach_gt / nb_bundles_in_stats,
            "mean_OR_vs": mean_overreach_vs / nb_bundles_in_stats,
            "mean_f1": mean_f1 / nb_bundles_in_stats
        })

    return final_results
