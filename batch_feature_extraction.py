# Extracts the features, labels, and normalizes the development and evaluation split features.

import cls_feature_class
import parameters
import sys


def main(argv):
    # Expects one input - task-id - corresponding to the configuration given in the parameter.py file.
    # Extracts features and labels relevant for the task-id
    # It is enough to compute the feature and labels once. 

    # use parameter set defined by user
    task_id = '1' if len(argv) < 2 else argv[1]
    params = parameters.get_params(task_id)

    # -------------- Extract features and labels for development set -----------------------------
    if params['mode'] == 'dev':
        dev_feat_cls = cls_feature_class.FeatureClass(params, is_eval=False)

        # # Extract features and normalize them
        dev_feat_cls.extract_all_feature()
        dev_feat_cls.preprocess_features()

        # # Extract labels
        dev_feat_cls.extract_all_labels()

        # # Extract visual features
        if params['modality'] == 'audio_visual':
            dev_feat_cls.extract_visual_features()

    else:
        dev_feat_cls = cls_feature_class.FeatureClass(params, is_eval=True)

        # # Extract features and normalize them
        dev_feat_cls.extract_all_feature()
        dev_feat_cls.preprocess_features()

        # # Extract visual features
        if params['modality'] == 'audio_visual':
            dev_feat_cls.extract_visual_features()


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv))
    except (ValueError, IOError) as e:
        sys.exit(e)

