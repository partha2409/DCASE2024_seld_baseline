# Implements the localization and detection metrics proposed in [1] with extensions to support multi-instance of the same class from [2].
#
# [1] Joint Measurement of Localization and Detection of Sound Events
# Annamaria Mesaros, Sharath Adavanne, Archontis Politis, Toni Heittola, Tuomas Virtanen
# WASPAA 2019
#
# [2] Overview and Evaluation of Sound Event Localization and Detection in DCASE 2019
# Politis, Archontis, Annamaria Mesaros, Sharath Adavanne, Toni Heittola, and Tuomas Virtanen.
# IEEE/ACM Transactions on Audio, Speech, and Language Processing (2020).
#
# This script has MIT license
#

import numpy as np

eps = np.finfo(float).eps
from scipy.optimize import linear_sum_assignment
from IPython import embed


class SELDMetricsSegmentLevel(object):
    def __init__(self, doa_threshold=20, nb_classes=11, average='macro'):
        '''
            This class implements both the class-sensitive localization and location-sensitive detection metrics.
            Additionally, based on the user input, the corresponding averaging is performed within the segment.
            Used till DCASE2024.

        :param nb_classes: Number of sound classes. In the paper, nb_classes = 11
        :param doa_thresh: DOA threshold for location sensitive detection.
        '''
        self._nb_classes = nb_classes

        # Variables for Location-senstive detection performance
        self._TP = np.zeros(self._nb_classes)
        self._FP = np.zeros(self._nb_classes)
        self._FP_spatial = np.zeros(self._nb_classes)
        self._FN = np.zeros(self._nb_classes)

        self._Nref = np.zeros(self._nb_classes)

        self._spatial_T = doa_threshold

        self._S = 0
        self._D = 0
        self._I = 0

        # Variables for Class-sensitive localization performance
        self._total_DE = np.zeros(self._nb_classes)

        self._DE_TP = np.zeros(self._nb_classes)
        self._DE_FP = np.zeros(self._nb_classes)
        self._DE_FN = np.zeros(self._nb_classes)

        self._average = average

    def early_stopping_metric(self, _er, _f, _le, _lr):
        """
        Compute early stopping metric from sed and doa errors.

        :param sed_error: [error rate (0 to 1 range), f score (0 to 1 range)]
        :param doa_error: [doa error (in degrees), frame recall (0 to 1 range)]
        :return: early stopping metric result
        """
        seld_metric = np.mean([
            _er,
            1 - _f,
            _le / 180,
            1 - _lr
        ], 0)
        return seld_metric

    def compute_seld_scores(self):
        '''
        Collect the final SELD scores

        :return: returns both location-sensitive detection scores and class-sensitive localization scores
        '''
        ER = (self._S + self._D + self._I) / (self._Nref.sum() + eps)
        classwise_results = []
        if self._average == 'micro':
            # Location-sensitive detection performance
            F = self._TP.sum() / (eps + self._TP.sum() + self._FP_spatial.sum() + 0.5 * (self._FP.sum() + self._FN.sum()))

            # Class-sensitive localization performance
            LE = self._total_DE.sum() / float(self._DE_TP.sum() + eps) if self._DE_TP.sum() else 180
            LR = self._DE_TP.sum() / (eps + self._DE_TP.sum() + self._DE_FN.sum())

            SELD_scr = self.early_stopping_metric(ER, F, LE, LR)

        elif self._average == 'macro':
            # Location-sensitive detection performance
            F = self._TP / (eps + self._TP + self._FP_spatial + 0.5 * (self._FP + self._FN))

            # Class-sensitive localization performance
            LE = self._total_DE / (self._DE_TP + eps)
            LE[self._DE_TP==0] = 180.0
            LR = self._DE_TP / (eps + self._DE_TP + self._DE_FN)

            SELD_scr = self.early_stopping_metric(np.repeat(ER, self._nb_classes), F, LE, LR)
            classwise_results = np.array([np.repeat(ER, self._nb_classes), F, LE, LR, SELD_scr])
            F, LE, LR, SELD_scr = F.mean(), LE.mean(), LR.mean(), SELD_scr.mean()
        return ER, F, LE, LR, SELD_scr, classwise_results

    def update_seld_scores(self, pred, gt, eval_dist=False):
        '''
        Implements the spatial error averaging according to equation 5 in the paper [1] (see papers in the title of the code).
        Adds the multitrack extensions proposed in paper [2]

        The input pred/gt must be in Cartesian coordinates

        :param pred: dictionary containing class-wise prediction results for each N-seconds segment block
        :param gt: dictionary containing class-wise groundtruth for each N-seconds segment block
        :param eval_dist: boolean, if True, the distance estimation is also evaluated
        '''
        assert not eval_dist, 'Distance evaluation is not supported in segment level SELD evaluation'
        for block_cnt in range(len(gt.keys())):
            loc_FN, loc_FP = 0, 0
            for class_cnt in range(self._nb_classes):
                # Counting the number of referece tracks for each class in the segment
                nb_gt_doas = max([len(val) for val in gt[block_cnt][class_cnt][0][1]]) if class_cnt in gt[block_cnt] else None
                nb_pred_doas = max([len(val) for val in pred[block_cnt][class_cnt][0][1]]) if class_cnt in pred[block_cnt] else None
                if nb_gt_doas is not None:
                    self._Nref[class_cnt] += nb_gt_doas
                if class_cnt in gt[block_cnt] and class_cnt in pred[block_cnt]:
                    # True positives or False positive case

                    # NOTE: For multiple tracks per class, associate the predicted DOAs to corresponding reference
                    # DOA-tracks using hungarian algorithm and then compute the average spatial distance between
                    # the associated reference-predicted tracks.

                    # Reference and predicted track matching
                    matched_track_dist = {}
                    matched_track_cnt = {}
                    gt_ind_list = gt[block_cnt][class_cnt][0][0]
                    pred_ind_list = pred[block_cnt][class_cnt][0][0]
                    for gt_ind, gt_val in enumerate(gt_ind_list):
                        if gt_val in pred_ind_list:
                            gt_arr = np.array(gt[block_cnt][class_cnt][0][1][gt_ind])
                            gt_ids = np.arange(len(gt_arr[:, -1])) #TODO if the reference has track IDS use here - gt_arr[:, -1]
                            gt_doas = gt_arr[:, 1:]

                            pred_ind = pred_ind_list.index(gt_val)
                            pred_arr = np.array(pred[block_cnt][class_cnt][0][1][pred_ind])
                            pred_doas = pred_arr[:, 1:]

                            # Extract distance
                            if gt_doas.shape[-1] == 4:
                                gt_dist = gt_doas[:, 3]
                                gt_doas = gt_doas[:, :3]
                            if pred_doas.shape[-1] == 4:
                                pred_dist = pred_doas[:, 3]
                                pred_doas = pred_doas[:, :3]

                            dist_list, row_inds, col_inds = least_distance_between_gt_pred(gt_doas, pred_doas)

                            # Collect the frame-wise distance between matched ref-pred DOA pairs
                            for dist_cnt, dist_val in enumerate(dist_list):
                                matched_gt_track = gt_ids[row_inds[dist_cnt]]
                                if matched_gt_track not in matched_track_dist:
                                    matched_track_dist[matched_gt_track], matched_track_cnt[matched_gt_track] = [], []
                                matched_track_dist[matched_gt_track].append(dist_val)
                                matched_track_cnt[matched_gt_track].append(pred_ind)

                    # Update evaluation metrics based on the distance between ref-pred tracks
                    if len(matched_track_dist) == 0:
                        # if no tracks are found. This occurs when the predicted DOAs are not aligned frame-wise to the reference DOAs
                        loc_FN += nb_pred_doas
                        self._FN[class_cnt] += nb_pred_doas
                        self._DE_FN[class_cnt] += nb_pred_doas
                    else:
                        # for the associated ref-pred tracks compute the metrics
                        for track_id in matched_track_dist:
                            total_spatial_dist = sum(matched_track_dist[track_id])
                            total_framewise_matching_doa = len(matched_track_cnt[track_id])
                            avg_spatial_dist = total_spatial_dist / total_framewise_matching_doa

                            # Class-sensitive localization performance
                            self._total_DE[class_cnt] += avg_spatial_dist
                            self._DE_TP[class_cnt] += 1

                            # Location-sensitive detection performance
                            if avg_spatial_dist <= self._spatial_T:
                                self._TP[class_cnt] += 1
                            else:
                                loc_FP += 1
                                self._FP_spatial[class_cnt] += 1
                        # in the multi-instance of same class scenario, if the number of predicted tracks are greater
                        # than reference tracks count as FP, if it less than reference count as FN
                        if nb_pred_doas > nb_gt_doas:
                            # False positive
                            loc_FP += (nb_pred_doas-nb_gt_doas)
                            self._FP[class_cnt] += (nb_pred_doas-nb_gt_doas)
                            self._DE_FP[class_cnt] += (nb_pred_doas-nb_gt_doas)
                        elif nb_pred_doas < nb_gt_doas:
                            # False negative
                            loc_FN += (nb_gt_doas-nb_pred_doas)
                            self._FN[class_cnt] += (nb_gt_doas-nb_pred_doas)
                            self._DE_FN[class_cnt] += (nb_gt_doas-nb_pred_doas)
                elif class_cnt in gt[block_cnt] and class_cnt not in pred[block_cnt]:
                    # False negative
                    loc_FN += nb_gt_doas
                    self._FN[class_cnt] += nb_gt_doas
                    self._DE_FN[class_cnt] += nb_gt_doas
                elif class_cnt not in gt[block_cnt] and class_cnt in pred[block_cnt]:
                    # False positive
                    loc_FP += nb_pred_doas
                    self._FP[class_cnt] += nb_pred_doas
                    self._DE_FP[class_cnt] += nb_pred_doas

            self._S += np.minimum(loc_FP, loc_FN)
            self._D += np.maximum(0, loc_FN - loc_FP)
            self._I += np.maximum(0, loc_FP - loc_FN)
        return


class SELDMetrics(object):
    def __init__(self, doa_threshold=20, dist_threshold=np.inf, reldist_threshold=np.inf, nb_classes=11, eval_dist=True,
                 average='macro',):
        '''
            This class implements both the class-sensitive localization and location-sensitive detection metrics.

        :param nb_classes: Number of sound classes. In the paper, nb_classes = 11
        :param doa_thresh: DOA threshold for location sensitive detection.
        :param dist_thresh: Relative distance threshold for distance estimation
        '''
        self._nb_classes = nb_classes
        self.eval_dist = eval_dist

        # Variables for Location-senstive detection performance
        self._TP = np.zeros(self._nb_classes)
        self._FP = np.zeros(self._nb_classes)
        self._FP_spatial = np.zeros(self._nb_classes)
        self._FN = np.zeros(self._nb_classes)

        self._Nref = np.zeros(self._nb_classes)

        self._ang_T = doa_threshold
        self._dist_T = dist_threshold
        self._reldist_T = reldist_threshold

        self._S = 0
        self._D = 0
        self._I = 0

        # Variables for Class-sensitive localization performance
        self._total_AngE = np.zeros(self._nb_classes)
        self._total_DistE = np.zeros(self._nb_classes)
        self._total_RelDistE = np.zeros(self._nb_classes)

        self._DE_TP = np.zeros(self._nb_classes)
        self._DE_FP = np.zeros(self._nb_classes)
        self._DE_FN = np.zeros(self._nb_classes)

        self._idss = np.zeros(self._nb_classes)

        self._average = average

    def early_stopping_metric(self, _er, _f, _ae, _lr, _rde):
        """
        Compute early stopping metric from sed and doa errors.
        """
        if self.eval_dist:  # 2024 Challenge
            seld_metric = np.nanmean([
                1 - _f,
                _ae / 180,
                _rde
            ], 0)
        else:  # 2023 Challenge
            seld_metric = np.nanmean([
                _er,
                1 - _f,
                _ae / 180,
                1 - _lr
            ], 0)
        return seld_metric

    def compute_seld_scores(self):
        '''
        Collect the final SELD scores
        ISDR is not returned and hasn't been tested

        :return: returns both location-sensitive detection scores and class-sensitive localization scores
        '''
        ER = (self._S + self._D + self._I) / (self._Nref.sum() + eps)
        classwise_results = []
        if self._average == 'micro':
            # Location-sensitive detection performance
            F = self._TP.sum() / (eps + self._TP.sum() + self._FP_spatial.sum() + 0.5 * (self._FP.sum() + self._FN.sum()))

            # Class-sensitive localization performance
            AngE = self._total_AngE.sum() / float(self._DE_TP.sum() + eps) if self._DE_TP.sum() else np.NaN
            DistE = self._total_DistE.sum() / float(self._DE_TP.sum() + eps) if self._DE_TP.sum() else np.NaN
            RelDistE = self._total_RelDistE.sum() / float(self._DE_TP.sum() + eps) if self._DE_TP.sum() else np.NaN
            LR = self._DE_TP.sum() / (eps + self._DE_TP.sum() + self._DE_FN.sum())

            SELD_scr = self.early_stopping_metric(ER, F, AngE, LR, RelDistE)

            IDSR = self._idss.sum() / self._Nref.sum() if self._Nref.sum() else np.NaN

        elif self._average == 'macro':
            # Location-sensitive detection performance
            F = self._TP / (eps + self._TP + self._FP_spatial + 0.5 * (self._FP + self._FN))

            # Class-sensitive localization performance
            AngE = self._total_AngE / (self._DE_TP + eps)
            AngE[self._DE_TP==0] = np.NaN
            DistE = self._total_DistE / (self._DE_TP + eps)
            DistE[self._DE_TP==0] = np.NaN
            RelDistE = self._total_RelDistE / (self._DE_TP + eps)
            RelDistE[self._DE_TP==0] = np.NaN
            LR = self._DE_TP / (eps + self._DE_TP + self._DE_FN)

            SELD_scr = self.early_stopping_metric(np.repeat(ER, self._nb_classes), F, AngE, LR, RelDistE)

            IDSR = self._idss / (self._Nref + eps)
            IDSR[self._Nref==0] = np.NaN

            classwise_results = np.array(
                [np.repeat(ER, self._nb_classes), F, AngE, DistE, RelDistE, LR, SELD_scr] if self.eval_dist else [
                    np.repeat(ER, self._nb_classes), F, AngE, LR, SELD_scr])

            non_zero_F_indices = np.where(np.round(F,2) != 0)

            F, AngE, LR, SELD_scr, IDSR = F.mean(), np.nanmean(AngE[non_zero_F_indices]), LR[non_zero_F_indices].mean(), SELD_scr[non_zero_F_indices].mean(), IDSR.mean()
            DistE, RelDistE = np.nanmean(DistE[non_zero_F_indices]), np.nanmean(RelDistE[non_zero_F_indices])
        return (ER, F, AngE, DistE, RelDistE, LR, SELD_scr, classwise_results) if self.eval_dist else (
                    ER, F, AngE, LR, SELD_scr, classwise_results)

    def update_seld_scores(self, pred, gt, eval_dist=False):
        '''
        Implements the spatial error averaging according to equation 5 in the paper [1] (see papers in the title of the code).
        Adds the multitrack extensions proposed in paper [2]

        The input pred/gt must be Cartesian coordinates

        :param pred: dictionary containing the predictions for every frame
            pred[frame-index][class-index][track-index] = [x, y, z, (distance)]
        :param gt: dictionary containing the groundtruth for every frame
            gt[frame-index][class-index][track-index] = [x, y, z, (distance)]
        :param eval_dist: boolean, if True, the distance estimation is also evaluated
        '''
        assignations = [{} for i in range(self._nb_classes)]
        assignations_pre = [{} for i in range(self._nb_classes)]
        for frame_cnt in range(len(gt.keys())):
            loc_FN, loc_FP = 0, 0
            for class_cnt in range(self._nb_classes):
                # Counting the number of referece tracks for each class
                nb_gt_doas = len(gt[frame_cnt][class_cnt]) if class_cnt in gt[frame_cnt] else None
                nb_pred_doas = len(pred[frame_cnt][class_cnt]) if class_cnt in pred[frame_cnt] else None
                if nb_gt_doas is not None:
                    self._Nref[class_cnt] += nb_gt_doas
                if class_cnt in gt[frame_cnt] and class_cnt in pred[frame_cnt]:
                    # True positives or False positive case

                    # NOTE: For multiple tracks per class, associate the predicted DOAs to corresponding reference
                    # DOA-tracks using hungarian algorithm and then compute the average spatial distance between
                    # the associated reference-predicted tracks.

                    # Reference and predicted track matching

                    gt_doas = np.array(list(gt[frame_cnt][class_cnt].values()))
                    gt_ids = np.array(list(gt[frame_cnt][class_cnt].keys()))
                    pred_doas = np.array(list(pred[frame_cnt][class_cnt].values()))
                    pred_ids = np.array(list(pred[frame_cnt][class_cnt].keys()))

                    # Extract distance
                    if gt_doas.shape[-1] == 4:
                        gt_dist = gt_doas[:, 3] if eval_dist else None
                        gt_doas = gt_doas[:, :3]
                    else:
                        assert not eval_dist, 'Distance evaluation was requested but the ground-truth distance was not provided.'
                        gt_dist = None
                    if pred_doas.shape[-1] == 4:
                        pred_dist = pred_doas[:, 3] if eval_dist else None
                        pred_doas = pred_doas[:, :3]
                    else:
                        assert not eval_dist, 'Distance evaluation was requested but the predicted distance was not provided.'
                        pred_dist = None

                    doa_err_list, row_inds, col_inds = least_distance_between_gt_pred(gt_doas, pred_doas, gt_dist, pred_dist)
                    assignations[class_cnt] = {gt_ids[row_inds[i]] : pred_ids[col_inds[i]] for i in range(len(doa_err_list))}
                    for gt_id, pred_id in assignations[class_cnt].items():
                        if gt_id in assignations_pre[class_cnt] and assignations_pre[class_cnt][gt_id] != pred_id:
                            self._idss[class_cnt] += 1
                    if eval_dist:
                        dist_err_list = np.abs(gt_dist[row_inds] - pred_dist[col_inds])
                        rel_dist_err_list = dist_err_list / (gt_dist[row_inds] + eps)

                    # https://dcase.community/challenge2022/task-sound-event-localization-and-detection-evaluated-in-real-spatial-sound-scenes#evaluation
                    Pc = len(pred_doas)
                    Rc = len(gt_doas)
                    FNc = max(0, Rc - Pc)
                    FPcinf = max(0, Pc - Rc)
                    Kc = min(Pc, Rc)
                    TPc = Kc
                    Lc = np.sum(doa_err_list > self._ang_T or (eval_dist and dist_err_list > self._dist_T)
                                                           or (eval_dist and rel_dist_err_list > self._reldist_T))
                    FPct = Lc
                    FPc = FPcinf + FPct
                    TPct = Kc - FPct
                    assert Pc == TPct + FPc
                    assert Rc == TPct + FPct + FNc

                    self._total_AngE[class_cnt] += doa_err_list.sum()
                    self._total_DistE[class_cnt] += dist_err_list.sum() if eval_dist else 0
                    self._total_RelDistE[class_cnt] += rel_dist_err_list.sum() if eval_dist else 0

                    self._TP[class_cnt] += TPct
                    self._DE_TP[class_cnt] += TPc

                    self._FP[class_cnt] += FPcinf
                    self._DE_FP[class_cnt] += FPcinf
                    self._FP_spatial[class_cnt] += FPct
                    loc_FP += FPc

                    self._FN[class_cnt] += FNc
                    self._DE_FN[class_cnt] += FNc
                    loc_FN += FNc

                    assignations_pre[class_cnt] = assignations[class_cnt]

                elif class_cnt in gt[frame_cnt] and class_cnt not in pred[frame_cnt]:
                    # False negative
                    loc_FN += nb_gt_doas
                    self._FN[class_cnt] += nb_gt_doas
                    self._DE_FN[class_cnt] += nb_gt_doas
                    assignations_pre[class_cnt] = {}
                elif class_cnt not in gt[frame_cnt] and class_cnt in pred[frame_cnt]:
                    # False positive
                    loc_FP += nb_pred_doas
                    self._FP[class_cnt] += nb_pred_doas
                    self._DE_FP[class_cnt] += nb_pred_doas
                    assignations_pre[class_cnt] = {}
                else:
                    # True negative
                    assignations_pre[class_cnt] = {}

            self._S += np.minimum(loc_FP, loc_FN)
            self._D += np.maximum(0, loc_FN - loc_FP)
            self._I += np.maximum(0, loc_FP - loc_FN)
        return


def distance_between_spherical_coordinates_rad(az1, ele1, az2, ele2):
    """
    Angular distance between two spherical coordinates
    MORE: https://en.wikipedia.org/wiki/Great-circle_distance

    :return: angular distance in degrees
    """
    dist = np.sin(ele1) * np.sin(ele2) + np.cos(ele1) * np.cos(ele2) * np.cos(np.abs(az1 - az2))
    # Making sure the dist values are in -1 to 1 range, else np.arccos kills the job
    dist = np.clip(dist, -1, 1)
    dist = np.arccos(dist) * 180 / np.pi
    return dist


def distance_between_cartesian_coordinates(x1, y1, z1, x2, y2, z2):
    """
    Angular distance between two cartesian coordinates
    MORE: https://en.wikipedia.org/wiki/Great-circle_distance
    Check 'From chord length' section

    :return: angular distance in degrees
    """
    # Normalize the Cartesian vectors
    N1 = np.sqrt(x1**2 + y1**2 + z1**2 + 1e-10)
    N2 = np.sqrt(x2**2 + y2**2 + z2**2 + 1e-10)
    x1, y1, z1, x2, y2, z2 = x1/N1, y1/N1, z1/N1, x2/N2, y2/N2, z2/N2

    #Compute the distance
    dist = x1*x2 + y1*y2 + z1*z2
    dist = np.clip(dist, -1, 1)
    dist = np.arccos(dist) * 180 / np.pi
    return dist


def distance_3d_between_doas(x1, y1, z1, x2, y2, z2, dist1, dist2):
    """
    3D distance between two cartesian DOAs with their respective distances
    :return: 3D distance in meters
    """
    N1 = np.sqrt(x1**2 + y1**2 + z1**2 + 1e-10)
    x1, y1, z1 = x1/N1 * dist1, y1/N1 * dist1, z1/N1 * dist1
    N2 = np.sqrt(x2**2 + y2**2 + z2**2 + 1e-10)
    x2, y2, z2 = x2/N2 * dist2, y2/N2 * dist2, z2/N2 * dist2
    return np.sqrt((x1-x2)**2 + (y1-y2)**2 + (z1-z2)**2)

def least_distance_between_gt_pred(gt_list, pred_list, gt_dist=None, pred_dist=None,
                                   opt_3d_dist=False, ret_3d_dist=False):
    """
        Shortest distance between two sets of DOA coordinates. Given a set of groundtruth coordinates,
        and its respective predicted coordinates, we calculate the distance between each of the
        coordinate pairs resulting in a matrix of distances, where one axis represents the number of groundtruth
        coordinates and the other the predicted coordinates. The number of estimated peaks need not be the same as in
        groundtruth, thus the distance matrix is not always a square matrix. We use the hungarian algorithm to find the
        least cost in this distance matrix.
        :param gt_list_xyz: list of ground-truth DOA in Cartesian coordinates
        :param pred_list_xyz: list of predicted DOA in Carteisan coordinates
        :param gt_dist: list of ground-truth distances in meters (optional, for distance evaluation)
        :param pred_dist: list of predicted distances in meters (optional, for distance evaluation)
        :param opt_3d_dist: boolean, if True, the 3D distance is used for matching the predicted and groundtruth DOAs
        :param ret_3d_dist: boolean, if True, the 3D distance [meters] is returned instead of angular distance [degrees]
        :return: cost - distance
        :return: less - number of DOA's missed
        :return: extra - number of DOA's over-estimated
    """
    if opt_3d_dist or ret_3d_dist:
        assert gt_dist is not None and pred_dist is not None, 'Distance information is needed to compute 3D distances.'

    gt_len, pred_len = gt_list.shape[0], pred_list.shape[0]
    ind_pairs = np.array([[x, y] for y in range(pred_len) for x in range(gt_len)])
    cost_mat = np.zeros((gt_len, pred_len))
    dist_mat = np.zeros((gt_len, pred_len))

    if gt_len and pred_len:
        x1, y1, z1, x2, y2, z2 = gt_list[ind_pairs[:, 0], 0], gt_list[ind_pairs[:, 0], 1], gt_list[ind_pairs[:, 0], 2], pred_list[ind_pairs[:, 1], 0], pred_list[ind_pairs[:, 1], 1], pred_list[ind_pairs[:, 1], 2]
        if opt_3d_dist or ret_3d_dist:
            dist1 = gt_dist[ind_pairs[:, 0]]
            dist2 = pred_dist[ind_pairs[:, 1]]
            distances_3d = distance_3d_between_doas(x1, y1, z1, x2, y2, z2, dist1, dist2)
            if opt_3d_dist:
                cost_mat[ind_pairs[:, 0], ind_pairs[:, 1]] = distances_3d
            if ret_3d_dist:
                dist_mat[ind_pairs[:, 0], ind_pairs[:, 1]] = distances_3d
        if not (opt_3d_dist and ret_3d_dist):
            distances_ang = distance_between_cartesian_coordinates(x1, y1, z1, x2, y2, z2)
            if not opt_3d_dist:
                cost_mat[ind_pairs[:, 0], ind_pairs[:, 1]] = distances_ang
            if not ret_3d_dist:
                dist_mat[ind_pairs[:, 0], ind_pairs[:, 1]] = distances_ang

    row_ind, col_ind = linear_sum_assignment(cost_mat)
    cost = dist_mat[row_ind, col_ind]
    return cost, row_ind, col_ind
