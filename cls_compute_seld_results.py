import os
import SELD_evaluation_metrics
import cls_feature_class
import parameters
import numpy as np
from scipy import stats
from IPython import embed


def jackknife_estimation(global_value, partial_estimates, significance_level=0.05):
    """
    Compute jackknife statistics from a global value and partial estimates.
    Original function by Nicolas Turpault

    :param global_value: Value calculated using all (N) examples
    :param partial_estimates: Partial estimates using N-1 examples at a time
    :param significance_level: Significance value used for t-test

    :return:
    estimate: estimated value using partial estimates
    bias: Bias computed between global value and the partial estimates
    std_err: Standard deviation of partial estimates
    conf_interval: Confidence interval obtained after t-test
    """

    mean_jack_stat = np.mean(partial_estimates)
    n = len(partial_estimates)
    bias = (n - 1) * (mean_jack_stat - global_value)

    std_err = np.sqrt(
        (n - 1) * np.mean((partial_estimates - mean_jack_stat) * (partial_estimates - mean_jack_stat), axis=0)
    )

    # bias-corrected "jackknifed estimate"
    estimate = global_value - bias

    # jackknife confidence interval
    if not (0 < significance_level < 1):
        raise ValueError("confidence level must be in (0, 1).")

    t_value = stats.t.ppf(1 - significance_level / 2, n - 1)

    # t-test
    conf_interval = estimate + t_value * np.array((-std_err, std_err))

    return estimate, bias, std_err, conf_interval


class ComputeSELDResults(object):
    def __init__(self, params, ref_files_folder=None):
        self._desc_dir = ref_files_folder if ref_files_folder is not None else os.path.join(params['dataset_dir'],
                                                                                            'metadata_dev')
        self._doa_thresh = params['lad_doa_thresh']
        self._dist_thresh = params['lad_dist_thresh']  if 'lad_dist_thresh' in params else float('inf')
        self._reldist_thresh = params['lad_reldist_thresh'] if 'lad_reldist_thresh' in params else float('inf')
        self.segment_level = params['segment_based_metrics'] if 'segment_based_metrics' in params else True
        self.evaluate_distance = params['evaluate_distance'] if 'evaluate_distance' in params else False
        assert not (self.segment_level and self.evaluate_distance), 'Segment level evaluation is not supported for distance evaluation'

        # Load feature class
        self._feat_cls = cls_feature_class.FeatureClass(params)

        # collect reference files
        self._ref_labels = {}
        for split in os.listdir(self._desc_dir):
            for ref_file in os.listdir(os.path.join(self._desc_dir, split)):
                # Load reference description file
                gt_dict = self._feat_cls.load_output_format_file(os.path.join(self._desc_dir, split, ref_file), cm2m=True)  # TODO: Reconsider the cm2m conversion
                gt_dict = self._feat_cls.convert_output_format_polar_to_cartesian(gt_dict)
                nb_ref_frames = max(list(gt_dict.keys()))
                if self.segment_level:
                    self._ref_labels[ref_file] = [self._feat_cls.segment_labels(gt_dict, nb_ref_frames), nb_ref_frames]
                else:
                    self._ref_labels[ref_file] = [self._feat_cls.organize_labels(gt_dict, nb_ref_frames), nb_ref_frames]

        self._nb_ref_files = len(self._ref_labels)
        self._average = params['average']

    @staticmethod
    def get_nb_files(file_list, tag='all'):
        '''
        Given the file_list, this function returns a subset of files corresponding to the tag.

        Tags supported
        'all' -
        'ir'

        :param file_list: complete list of predicted files
        :param tag: Supports two tags 'all', 'ir'
        :return: Subset of files according to chosen tag
        '''
        _group_ind = {'room': 10}
        _cnt_dict = {}
        for _filename in file_list:

            if tag == 'all':
                _ind = 0
            else:
                _ind = int(_filename[_group_ind[tag]])

            if _ind not in _cnt_dict:
                _cnt_dict[_ind] = []
            _cnt_dict[_ind].append(_filename)

        return _cnt_dict

    def get_SELD_Results(self, pred_files_path, is_jackknife=False):
        # collect predicted files info
        pred_files = os.listdir(pred_files_path)
        pred_labels_dict = {}
        if self.segment_level:
            eval = SELD_evaluation_metrics.SELDMetricsSegmentLevel(nb_classes=self._feat_cls.get_nb_classes(),
                                                       doa_threshold=self._doa_thresh, average=self._average)
        else:
            eval = SELD_evaluation_metrics.SELDMetrics(nb_classes=self._feat_cls.get_nb_classes(),
                                    doa_threshold=self._doa_thresh, average=self._average, eval_dist=self.evaluate_distance,
                                    dist_threshold=self._dist_thresh, reldist_threshold=self._reldist_thresh)

        for pred_cnt, pred_file in enumerate(pred_files):
            # Load predicted output format file
            pred_dict = self._feat_cls.load_output_format_file(os.path.join(pred_files_path, pred_file))
            pred_dict = self._feat_cls.convert_output_format_polar_to_cartesian(pred_dict)
            if self.segment_level:
                pred_labels = self._feat_cls.segment_labels(pred_dict, self._ref_labels[pred_file][1])
                # pred_labels[segment-index][class-index] := list(frame-cnt-within-segment, azimuth, elevation)
            else:
                pred_labels = self._feat_cls.organize_labels(pred_dict, self._ref_labels[pred_file][1])
                # pred_labels[frame-index][class-index][track-index] := [azimuth, elevation]
            # Calculated scores
            eval.update_seld_scores(pred_labels, self._ref_labels[pred_file][0], eval_dist=self.evaluate_distance)
            if is_jackknife:
                pred_labels_dict[pred_file] = pred_labels
        # Overall SED and DOA scores

        if self.evaluate_distance:
            ER, F, AngE, DistE, RelDistE, LR, seld_scr, classwise_results = eval.compute_seld_scores()
        else:
            ER, F, AngE, LR, seld_scr, classwise_results = eval.compute_seld_scores()

        if is_jackknife:
            global_values = [ER, F, AngE, DistE, RelDistE, LR, seld_scr] if self.evaluate_distance \
                else [ER, F, AngE, LR, seld_scr]
            if len(classwise_results):
                global_values.extend(classwise_results.reshape(-1).tolist())
            partial_estimates = []
            # Calculate partial estimates by leave-one-out method
            for leave_file in pred_files:
                leave_one_out_list = pred_files[:]
                leave_one_out_list.remove(leave_file)
                if self.segment_level:
                    eval = SELD_evaluation_metrics.SELDMetricsSegmentLevel(nb_classes=self._feat_cls.get_nb_classes(),
                                                               doa_threshold=self._doa_thresh, average=self._average)
                else:
                    eval = SELD_evaluation_metrics.SELDMetrics(nb_classes=self._feat_cls.get_nb_classes(),
                                    doa_threshold=self._doa_thresh, average=self._average, eval_dist=self.evaluate_distance,
                                    dist_threshold=self._dist_thresh, reldist_threshold=self._reldist_thresh)
                for pred_cnt, pred_file in enumerate(leave_one_out_list):
                    # Calculated scores
                    eval.update_seld_scores(pred_labels_dict[pred_file], self._ref_labels[pred_file][0],
                                            eval_dist=self.evaluate_distance)
                if self.evaluate_distance:
                    ER, F, AngE, DistE, RelDistE, LR, seld_scr, classwise_results = eval.compute_seld_scores()
                    leave_one_out_est = [ER, F, AngE, DistE, RelDistE, LR, seld_scr]
                else:
                    ER, F, AngE, LR, seld_scr, classwise_results = eval.compute_seld_scores()
                    leave_one_out_est = [ER, F, AngE, LR, seld_scr]
                if len(classwise_results):
                    leave_one_out_est.extend(classwise_results.reshape(-1).tolist())

                # Overall SED and DOA scores
                partial_estimates.append(leave_one_out_est)
            partial_estimates = np.array(partial_estimates)

            estimate, bias, std_err, conf_interval = [-1] * len(global_values), [-1] * len(global_values), [-1] * len(
                global_values), [-1] * len(global_values)
            for i in range(len(global_values)):
                estimate[i], bias[i], std_err[i], conf_interval[i] = jackknife_estimation(
                    global_value=global_values[i],
                    partial_estimates=partial_estimates[:, i],
                    significance_level=0.05
                )

            if self.evaluate_distance:
                return ([ER, conf_interval[0]], [F, conf_interval[1]], [AngE, conf_interval[2]],
                        [DistE, conf_interval[3]], [RelDistE, conf_interval[4]], [LR, conf_interval[5]],
                        [seld_scr, conf_interval[6]], [classwise_results, np.array(conf_interval)[7:].reshape(7, 13, 2)
                                                            if len(classwise_results) else []])
            else:
                return [ER, conf_interval[0]], [F, conf_interval[1]], [AngE, conf_interval[2]], [LR, conf_interval[3]], [
                    seld_scr, conf_interval[4]], [classwise_results, np.array(conf_interval)[5:].reshape(5, 13, 2) if len(
                    classwise_results) else []]

        else:
            return (ER, F, AngE, DistE, RelDistE, LR, seld_scr, classwise_results) if self.evaluate_distance \
                else (ER, F, AngE, LR, seld_scr, classwise_results)

    def get_consolidated_SELD_results(self, pred_files_path, score_type_list=['all', 'room']):
        '''
            Get all categories of results.
            TODO: Check if it works at frame level

            ;score_type_list: Supported
                'all' - all the predicted files
                'room' - for individual rooms

        '''

        # collect predicted files info
        pred_files = os.listdir(pred_files_path)
        nb_pred_files = len(pred_files)

        # Calculate scores for different splits, overlapping sound events, and impulse responses (reverberant scenes)

        print('Number of predicted files: {}\nNumber of reference files: {}'.format(nb_pred_files, self._nb_ref_files))
        print('\nCalculating {} scores for {}'.format(score_type_list, os.path.basename(pred_output_format_files)))

        for score_type in score_type_list:
            print('\n\n---------------------------------------------------------------------------------------------------')
            print('------------------------------------  {}   ---------------------------------------------'.format('Total score' if score_type == 'all' else 'score per {}'.format(score_type)))
            print('---------------------------------------------------------------------------------------------------')

            split_cnt_dict = self.get_nb_files(pred_files, tag=score_type)  # collect files corresponding to score_type
            # Calculate scores across files for a given score_type
            for split_key in np.sort(list(split_cnt_dict)):
                # Load evaluation metric class
                if self.segment_level:
                    eval = SELD_evaluation_metrics.SELDMetricsSegmentLevel(nb_classes=self._feat_cls.get_nb_classes(),
                                                doa_threshold=self._doa_thresh, average=self._average)
                else:
                    eval = SELD_evaluation_metrics.SELDMetrics(nb_classes=self._feat_cls.get_nb_classes(),
                                    doa_threshold=self._doa_thresh, average=self._average, eval_dist=self.evaluate_distance,
                                    dist_threshold=self._dist_thresh, reldist_threshold=self._reldist_thresh)
                for pred_cnt, pred_file in enumerate(split_cnt_dict[split_key]):
                    # Load predicted output format file
                    pred_dict = self._feat_cls.load_output_format_file(os.path.join(pred_output_format_files, pred_file))
                    pred_dict = self._feat_cls.convert_output_format_polar_to_cartesian(pred_dict)
                    if self.segment_level:
                        pred_labels = self._feat_cls.segment_labels(pred_dict, self._ref_labels[pred_file][1])
                    else:
                        pred_labels = self._feat_cls.organize_labels(pred_dict, self._ref_labels[pred_file][1])

                    # Calculated scores
                    eval.update_seld_scores(pred_labels, self._ref_labels[pred_file][0],
                                            eval_dist=self.evaluate_distance)

                # Overall SED and DOA scores
                if self.evaluate_distance:
                    ER, F, AngE, DistE, RelDistE, LR, seld_scr, classwise_results = eval.compute_seld_scores()
                else:
                    ER, F, AngE, LR, seld_scr, classwise_results = eval.compute_seld_scores()

                print('\nAverage score for {} {} data using {} coordinates'.format(score_type,
                                                                                   'fold' if score_type == 'all' else split_key,
                                                                                   'Cartesian'))
                print('SELD score (early stopping metric): {:0.2f}'.format(seld_scr))
                print('SED metrics: Error rate: {:0.2f}, F-score:{:0.1f}'.format(ER, 100 * F))
                print('DOA metrics: DOA error: {:0.1f}, Localization Recall: {:0.1f}'.format(AngE, 100 * LR))
                if self.evaluate_distance:
                    print('Distance metrics: Distance error: {:0.1f}, Relative distance error: {:0.1f}'.format(DistE, RelDistE))


def reshape_3Dto2D(A):
    return A.reshape(A.shape[0] * A.shape[1], A.shape[2])


if __name__ == "__main__":
    pred_output_format_files = 'Submissions/Task_A/Politis_TAU_task3a_1/Politis_TAU_task3a_1'  # Path of the DCASEoutput format files
    params = parameters.get_params()
    # Compute just the DCASE final results
    use_jackknife = False
    eval_dist = params['evaluate_distance'] if 'evaluate_distance' in params else False
    score_obj = ComputeSELDResults(params, ref_files_folder='metadata_eval_shuffled')
    if eval_dist:
        ER, F, AngE, DistE, RelsDistE, LR, seld_scr, classwise_test_scr = score_obj.get_SELD_Results(pred_output_format_files,
                                                                                                     is_jackknife=use_jackknife)
    else:
        ER, F, AngE, LR, seld_scr, classwise_test_scr = score_obj.get_SELD_Results(pred_output_format_files,
                                                                                   is_jackknife=use_jackknife)

    print('SELD score (early stopping metric): {:0.2f} {}'.format(
        seld_scr[0] if use_jackknife else seld_scr,
        '[{:0.2f}, {:0.2f}]'.format(seld_scr[1][0], seld_scr[1][1]) if use_jackknife else ''))
    print('SED metrics: F-score: {:0.1f} {}'.format(
        100 * F[0] if use_jackknife else 100 * F,
        '[{:0.2f}, {:0.2f}]'.format(100 * F[1][0], 100 * F[1][1]) if use_jackknife else ''))
    print('DOA metrics: DOA error: {:0.1f} {}'.format(
        AngE[0] if use_jackknife else AngE,
        '[{:0.2f}, {:0.2f}]'.format(AngE[1][0], AngE[1][1]) if use_jackknife else ''))
    if eval_dist:
        print('Distance metrics: Distance error: {:0.2f} {}, Relative distance error: {:0.2f} {}'.format(
            DistE[0] if use_jackknife else DistE,
            '[{:0.2f}, {:0.2f}]'.format(DistE[1][0], DistE[1][1]) if use_jackknife else '',
            RelsDistE[0] if use_jackknife else RelsDistE,
            '[{:0.2f}, {:0.2f}]'.format(RelsDistE[1][0], RelsDistE[1][1]) if use_jackknife else '')
        )
    if params['average'] == 'macro':
        print('Classwise results on unseen test data')
        print('Class\tF\tAngE\tDistE\tRelDistE\tSELD_score')
        for cls_cnt in range(params['unique_classes']):
            if eval_dist:
                print('{}\t{:0.2f} {}\t{:0.2f} {}\t{:0.2f} {}\t{:0.2f} {}\t{:0.2f} {}\t{:0.2f} {}\t{:0.2f} {}'.format(
                    cls_cnt,
                    #classwise_test_scr[0][0][cls_cnt] if use_jackknife else classwise_test_scr[0][cls_cnt],
                    #'[{:0.2f}, {:0.2f}]'.format(classwise_test_scr[1][0][cls_cnt][0],
                    #                            classwise_test_scr[1][0][cls_cnt][1]) if use_jackknife else '',
                    classwise_test_scr[0][1][cls_cnt] if use_jackknife else classwise_test_scr[1][cls_cnt],
                    '[{:0.2f}, {:0.2f}]'.format(classwise_test_scr[1][1][cls_cnt][0],
                                                classwise_test_scr[1][1][cls_cnt][1]) if use_jackknife else '',
                    classwise_test_scr[0][2][cls_cnt] if use_jackknife else classwise_test_scr[2][cls_cnt],
                    '[{:0.2f}, {:0.2f}]'.format(classwise_test_scr[1][2][cls_cnt][0],
                                                classwise_test_scr[1][2][cls_cnt][1]) if use_jackknife else '',
                    classwise_test_scr[0][3][cls_cnt] if use_jackknife else classwise_test_scr[3][cls_cnt],
                    '[{:0.2f}, {:0.2f}]'.format(classwise_test_scr[1][3][cls_cnt][0],
                                                classwise_test_scr[1][3][cls_cnt][1]) if use_jackknife else '',
                    classwise_test_scr[0][4][cls_cnt] if use_jackknife else classwise_test_scr[4][cls_cnt],
                    '[{:0.2f}, {:0.2f}]'.format(classwise_test_scr[1][4][cls_cnt][0],
                                                classwise_test_scr[1][4][cls_cnt][1]) if use_jackknife else '',
                    #classwise_test_scr[0][5][cls_cnt] if use_jackknife else classwise_test_scr[5][cls_cnt],
                    #'[{:0.2f}, {:0.2f}]'.format(classwise_test_scr[1][5][cls_cnt][0],
                    #                            classwise_test_scr[1][5][cls_cnt][1]) if use_jackknife else '',
                    classwise_test_scr[0][6][cls_cnt] if use_jackknife else classwise_test_scr[6][cls_cnt],
                    '[{:0.2f}, {:0.2f}]'.format(classwise_test_scr[1][6][cls_cnt][0],
                                                classwise_test_scr[1][6][cls_cnt][1]) if use_jackknife else ''))
            else:
                print('{}\t{:0.2f} {}\t{:0.2f} {}\t{:0.2f} {}\t{:0.2f} {}\t{:0.2f} {}'.format(
                    cls_cnt,
                    classwise_test_scr[0][0][cls_cnt] if use_jackknife else classwise_test_scr[0][cls_cnt],
                    '[{:0.2f}, {:0.2f}]'.format(classwise_test_scr[1][0][cls_cnt][0],
                                                classwise_test_scr[1][0][cls_cnt][1]) if use_jackknife else '',
                    classwise_test_scr[0][1][cls_cnt] if use_jackknife else classwise_test_scr[1][cls_cnt],
                    '[{:0.2f}, {:0.2f}]'.format(classwise_test_scr[1][1][cls_cnt][0],
                                                classwise_test_scr[1][1][cls_cnt][1]) if use_jackknife else '',
                    classwise_test_scr[0][2][cls_cnt] if use_jackknife else classwise_test_scr[2][cls_cnt],
                    '[{:0.2f}, {:0.2f}]'.format(classwise_test_scr[1][2][cls_cnt][0],
                                                classwise_test_scr[1][2][cls_cnt][1]) if use_jackknife else '',
                    classwise_test_scr[0][3][cls_cnt] if use_jackknife else classwise_test_scr[3][cls_cnt],
                    '[{:0.2f}, {:0.2f}]'.format(classwise_test_scr[1][3][cls_cnt][0],
                                                classwise_test_scr[1][3][cls_cnt][1]) if use_jackknife else '',
                    classwise_test_scr[0][4][cls_cnt] if use_jackknife else classwise_test_scr[4][cls_cnt],
                    '[{:0.2f}, {:0.2f}]'.format(classwise_test_scr[1][4][cls_cnt][0],
                                                classwise_test_scr[1][4][cls_cnt][1]) if use_jackknife else ''))

    # UNCOMMENT to Compute DCASE results along with room-wise performance
    # score_obj.get_consolidated_SELD_results(pred_output_format_files)

