"""A medical image analysis pipeline.

The pipeline is used for brain tissue segmentation using a decision forest classifier.
"""
import argparse
import datetime
import os
import sys
import timeit
import warnings

import SimpleITK as sitk
import sklearn.ensemble as sk_ensemble
import numpy as np
import pymia.data.conversion as conversion
import pymia.evaluation.writer as writer

import joblib
from sklearn.model_selection import GridSearchCV
import configparser
import logging

if sys.version_info[0:2] > (3, 9):
    raise Exception('Python version above 3.9 may cause problems with SimpleITK. [BufferError: memoryview has 1 exported buffer]')

try:
    import mialab.data.structure as structure
    import mialab.utilities.file_access_utilities as futil
    import mialab.utilities.pipeline_utilities as putil
except ImportError:
    # Append the MIALab root directory to Python path
    sys.path.insert(0, os.path.join(os.path.dirname(sys.argv[0]), '..'))
    import mialab.data.structure as structure
    import mialab.utilities.file_access_utilities as futil
    import mialab.utilities.pipeline_utilities as putil

LOADING_KEYS = [structure.BrainImageTypes.T1w,
                structure.BrainImageTypes.T2w,
                structure.BrainImageTypes.GroundTruth,
                structure.BrainImageTypes.BrainMask,
                structure.BrainImageTypes.RegistrationTransform]  # the list of data we will load

timestamp = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
logging.basicConfig(filename=f'script_log_{timestamp}.txt', level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
def read_config(config_path):
    config = configparser.ConfigParser()
    config.read(config_path)
    current_directory = os.getcwd()
    logger.info(f"Current Working Directory: {current_directory}")
    logger.debug("Sections in the configuration file: %s", config.sections())
    if 'main' not in config:
        logger.error("The 'main' section is missing in the configuration file.")
        raise ValueError("The 'main' section is missing in the configuration file.")

    return config['main']

config = read_config("./config.ini")

# Define the parameter grid
param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [None, 10, 20, 30]
}

label_mapping = {
            0: 'Background',
            1: 'WhiteMatter',
            2: 'GreyMatter',
            3: 'Hippocampus',
            4: 'Amygdala',
            5: 'Thalamus'
        }


def get_dice_scores_for_all_labels(results):
    unique_labels = set(
        result.label for result in results if result.metric == 'DICE')
    dice_scores_per_label = {}
    for label in unique_labels:
        dice_scores_per_label[label] = [result.value for result in results if result.metric == 'DICE' and result.label == label]
    return dice_scores_per_label

def get_voxel_count_per_label(img):
    image_array = sitk.GetArrayFromImage(img)
    unique_labels = np.unique(image_array)
    #volume_per_label = []
    volume_per_label = {}
    for label in unique_labels:
        # Extract the voxel indices for the current label
        label_indices = np.where(image_array == label)

        # Calculate the size of the label in each dimension
        size_x = (np.max(label_indices[0]) - np.min(label_indices[0]) + 1)
        size_y = (np.max(label_indices[1]) - np.min(label_indices[1]) + 1)
        size_z = (np.max(label_indices[2]) - np.min(label_indices[2]) + 1)

        volume = size_x * size_y * size_z
        volume_per_label[label] = volume
    return volume_per_label

def map_fields(init_dict, map_dict, res_dict=None):

    res_dict = res_dict or {}
    for k, v in init_dict.items():
        if isinstance(v, dict):
            v = map_fields(v, map_dict[k])
        elif k in map_dict.keys():
            k = str(map_dict[k])
        res_dict[k] = v
    return res_dict

def order_dicts(dict1, dict2):

    ordered_keys = sorted(dict1.keys())
    return {key: dict2[key] for key in ordered_keys}


def calculate_weighted_dice_score_per_label(dice_scores_per_label, volume_per_label):
    if len(dice_scores_per_label) != len(dice_scores_per_label):
        raise ValueError("Lengths of dice_scores and voxel_counts must be the same.")

    # Calculate the total voxel count
    total_voxel_count = sum(volume_per_label.values())

    if list(dice_scores_per_label.keys()) != list(volume_per_label.keys()):
        voxel_counts_ordered = order_dicts(dice_scores_per_label, volume_per_label)



    # Calculate the weighted Dice scores per label
    weighted_dice_scores = {label: (1 - voxel_counts_ordered[label] / total_voxel_count)*dice_scores_per_label[label] for label in
                            dice_scores_per_label.keys()}
    return weighted_dice_scores

def main(result_dir: str, data_atlas_dir: str, data_train_dir: str, data_test_dir: str, label: int = 0):
    """Brain tissue segmentation using decision forests.

    The main routine executes the medical image analysis pipeline:

        - Image loading
        - Registration
        - Pre-processing
        - Feature extraction
        - Decision forest classifier model building
        - Segmentation using the decision forest classifier model on unseen images
        - Post-processing of the segmentation
        - Evaluation of the segmentation
    """

    # Cutting of data for faster testing
    CUT_DATA = config.getboolean('CUT_DATA')
    CUTOFF_NUMBER = config.getint('CUTOFF_NUMBER')  # 3

    # use joblib
    USE_JOBLIB_FOREST = config.getboolean('USE_JOBLIB_FOREST')
    USE_JOBLIB_PREPROCESSING = config.getboolean('USE_JOBLIB_PREPROCESSING')

    USE_GRIDSEARCH = config.getboolean('USE_GRIDSEARCH')

    # use weighted dice score
    WEIGHTED_DICE = config.getboolean("WEIGHTED_DICE")

    # load atlas images
    putil.load_atlas_images(data_atlas_dir)

    print('-' * 5, 'Training...')
    logger.info('------Training...')


    # crawl the training image directories
    crawler = futil.FileSystemDataCrawler(data_train_dir,
                                          LOADING_KEYS,
                                          futil.BrainImageFilePathGenerator(),
                                          futil.DataDirectoryFilter())
    logger.info('Crawler was run')
    pre_process_params = {'skullstrip_pre': True,
                          'normalization_pre': True,
                          'registration_pre': True,
                          'coordinates_feature': True,
                          'intensity_feature': True,
                          'gradient_intensity_feature': True}
    logger.info('pre_process_params were set')

    if not USE_JOBLIB_FOREST:

        if not USE_JOBLIB_PREPROCESSING:
            images = putil.pre_process_batch(crawler.data, pre_process_params, multi_process=False, label=label)
            joblib.dump(images, './bin/preprocessed_images.joblib')
            logger.info('preprocessed_images were dumped to joblib')

        if USE_JOBLIB_PREPROCESSING:
            images = joblib.load('./bin/preprocessed_images.joblib')
            print('[INFO] Loaded preprocessed images from joblib')
            logger.info('[INFO] Loaded preprocessed images from joblib')

        # generate feature matrix and label vector
        data_train = np.concatenate([img.feature_matrix[0] for img in images])
        labels_train = np.concatenate([img.feature_matrix[1] for img in images]).squeeze()
        logger.info('data_train and labels_train were generated')

        # Create the Random Forest model

        if USE_GRIDSEARCH:
            initforest = sk_ensemble.RandomForestClassifier(max_features=images[0].feature_matrix[0].shape[1])
            start_time = timeit.default_timer()
            print('[INFO] Start grid search (this may take some time)')
            logger.info('[INFO] Start grid search (this may take some time)')
            grid_search = GridSearchCV(initforest, param_grid=param_grid, cv=3, scoring='accuracy')
            model_grid = grid_search.fit(data_train, labels_train)
            print(' Time elapsed:', timeit.default_timer() - start_time, 's')

            # Get the best hyperparameters
            best_params = model_grid.best_params_
            best_params['max_features']= images[0].feature_matrix[0].shape[1]
            print("Best Hyperparameters:", best_params)
            logger.info("Best Hyperparameters:", best_params)

            # Train the model with the best hyperparameters
            forest = sk_ensemble.RandomForestClassifier(**model_grid.best_params_)
        else:
            N_ESTIMATORS= config.getint('N_ESTIMATORS')
            MAX_DEPTH= config.getint('MAX_DEPTH')
            forest = sk_ensemble.RandomForestClassifier(max_features=images[0].feature_matrix[0].shape[1], n_estimators=N_ESTIMATORS, max_depth=MAX_DEPTH)

        start_time = timeit.default_timer()
        print('[INFO] Start training (this may take some time)')
        logger.info('[INFO] Start training (this may take some time)')
        forest.fit(data_train, labels_train)
        print(' Time elapsed:', timeit.default_timer() - start_time, 's')

        joblib.dump(forest, './bin/forest.pkl')

    else:
        try:
            forest = joblib.load('./bin/forest.pkl')
            print('[INFO] Use old forest.pkl')
        except:
            print('[ERROR] No pickle file found, cancelling program')
            print('[ERROR] Please set "USE_JOBLIB_FOREST=False" before next run')
            exit(1)

    # create a result directory with timestamp
    t = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    result_dir = os.path.join(result_dir, t)
    os.makedirs(result_dir, exist_ok=True)

    print('-' * 5, 'Testing...')
    logger.info('------Testing...')

    # initialize evaluator
    if binary_classification:
        evaluator = putil.init_evaluator(label)
        print('[INFO] Binary classification')
    else:
        evaluator = putil.init_evaluator()
        print('[INFO] Multi-label classification')

    # crawl the training image directories
    crawler = futil.FileSystemDataCrawler(data_test_dir,
                                          LOADING_KEYS,
                                          futil.BrainImageFilePathGenerator(),
                                          futil.DataDirectoryFilter())

    # load images for testing and pre-process
    pre_process_params['training'] = False
    if CUT_DATA == True:
        data_list = list(crawler.data.items())[:CUTOFF_NUMBER]
        subset_data = dict(data_list)
        images_test = putil.pre_process_batch(subset_data, pre_process_params, multi_process=False, label=label)
        print(f'[DEBUG] Data was cut to {CUTOFF_NUMBER} image(s)')
    else:
    # load images for training and pre-process
        images_test = putil.pre_process_batch(crawler.data, pre_process_params, multi_process=False, label=label)

    images_prediction = []
    images_probabilities = []

    for img in images_test:
        print('-' * 10, 'Testing', img.id_)
        logger.info('-----------Testing... img')

        start_time = timeit.default_timer()
        predictions = forest.predict(img.feature_matrix[0])
        probabilities = forest.predict_proba(img.feature_matrix[0])
        print(' Time elapsed:', timeit.default_timer() - start_time, 's')
        timeelapsed = timeit.default_timer() - start_time

        # convert prediction and probabilities back to SimpleITK images
        image_prediction = conversion.NumpySimpleITKImageBridge.convert(predictions.astype(np.uint8),
                                                                        img.image_properties)
        image_probabilities = conversion.NumpySimpleITKImageBridge.convert(probabilities, img.image_properties)

        # evaluate segmentation without post-processing
        evaluator.evaluate(image_prediction, img.images[structure.BrainImageTypes.GroundTruth], img.id_)

        images_prediction.append(image_prediction)
        images_probabilities.append(image_probabilities)

    # post-process segmentation and evaluate with post-processing
    post_process_params = {'simple_post': True}
    images_post_processed = putil.post_process_batch(images_test, images_prediction, images_probabilities,
                                                     post_process_params, multi_process=True)

    # evaluate segmentation with post-processing
    for i, img in enumerate(images_test):
        evaluator.evaluate(images_post_processed[i], img.images[structure.BrainImageTypes.GroundTruth],
                           img.id_ + '-PP')
        # save results
        sitk.WriteImage(images_prediction[i], os.path.join(result_dir, images_test[i].id_ + '_SEG.mha'), True)

    # use two writers to report the results
    os.makedirs(result_dir, exist_ok=True)  # generate result directory, if it does not exists
    result_file = os.path.join(result_dir, 'results.csv')
    writer.CSVWriter(result_file).write(evaluator.results)

    # Access some parameters (get either from joblib or from freshly trained model)
    best_params={'n_estimators': forest.n_estimators, 'max_depth': forest.max_depth, 'max_features': forest.max_features}
    print("Best Hyperparameters:\n")
    for key, value in best_params.items():
        print(f"{key}: {value}\n")

    # write run info file
    runinfofile = os.path.join(result_dir, 'RunInfo.txt')
    with open(runinfofile, 'w') as f:
        f.write("General info:\n")

        f.write("binary_classification: " + str(binary_classification) + "\n")
        f.write("label: "+str(label)+"\n")
        f.write("\nBest Hyperparameters:\n")
        for key, value in best_params.items():
            f.write(f'{key}: {value}\n')

    if WEIGHTED_DICE:
        # get dice scores for all labels
        dice_scores_all_labels = get_dice_scores_for_all_labels(evaluator.results)

        # Remove the second entry [1] from each list (for some reason I get double entries for each label)
        dice_scores_all_labels = {label: score[0] for label, score in dice_scores_all_labels.items()}

        #Get voxel count per label
        volume_per_img = [map_fields(get_voxel_count_per_label(img),label_mapping) for img in images_prediction]

        # Calculate the total voxel count per label
        voxel_count_per_label = {}
        for volume_dict in volume_per_img:
            for label, volume in volume_dict.items():
                if label not in voxel_count_per_label:
                    voxel_count_per_label[label] = []
                voxel_count_per_label[label].append(volume)

        # Calculate the mean voxel count per label
        mean_voxel_count_per_label = {label: np.mean(volumes) for label, volumes in voxel_count_per_label.items()}

        del mean_voxel_count_per_label["Background"]#remove background #fixme background same as total_voxels_avg???

        weighted_dice_scores = calculate_weighted_dice_score_per_label(dice_scores_all_labels, mean_voxel_count_per_label)

        print("Weighted Dice Scores per Label:")
        for label, score in weighted_dice_scores.items():
            print(f"Label {label}: {score}")
        with open(runinfofile, 'a') as f:
            f.write("\nWeighted Dice Scores:\n")
            for label, score in weighted_dice_scores.items():
                f.write(f"Label {label}: {score}\n")

    print('\nSubject-wise results...')
    writer.ConsoleWriter(use_logging=True).write(evaluator.results)

    # report also mean and standard deviation among all subjects
    result_summary_file = os.path.join(result_dir, 'results_summary.csv')
    functions = {'MEAN': np.mean, 'STD': np.std}
    writer.CSVStatisticsWriter(result_summary_file, functions=functions).write(evaluator.results)
    print('\nAggregated statistic results...')
    writer.ConsoleStatisticsWriter(functions=functions).write(evaluator.results)

    # clear results such that the evaluator is ready for the next evaluation
    evaluator.clear()

if __name__ == "__main__":
    """The program's entry point."""
    ##TODO: use binary classification


    binary_classification = config.getboolean('binary_classification')
    label = config.getint('label')

    script_dir = os.path.dirname(sys.argv[0])

    parser = argparse.ArgumentParser(description='Medical image analysis pipeline for brain tissue segmentation')

    parser.add_argument(
        '--result_dir',
        type=str,
        default=os.path.normpath(os.path.join(script_dir, './mia-result')),
        help='Directory for results.'
    )

    parser.add_argument(
        '--data_atlas_dir',
        type=str,
        default=os.path.normpath(os.path.join(script_dir, '../data/atlas')),
        help='Directory with atlas data.'
    )

    parser.add_argument(
        '--data_train_dir',
        type=str,
        default=os.path.normpath(os.path.join(script_dir, '../data/train/')),
        help='Directory with training data.'
    )

    parser.add_argument(
        '--data_test_dir',
        type=str,
        default=os.path.normpath(os.path.join(script_dir, '../data/test/')),
        help='Directory with testing data.'
    )

    args = parser.parse_args()
    if binary_classification:
        main(
            args.result_dir,
            args.data_atlas_dir,
            args.data_train_dir,
            args.data_test_dir,
            label,
        )
    else:
        main(
            args.result_dir,
            args.data_atlas_dir,
            args.data_train_dir,
            args.data_test_dir,
        )

