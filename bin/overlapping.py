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
import pymia.filtering.filter as pymia_filter

import joblib
from sklearn.model_selection import GridSearchCV
import configparser
import logging
import copy
import itertools
import pymia.filtering.filter as pymia_filter
import SimpleITK as sitk
import numpy as np

if sys.version_info[0:2] > (3, 9):
    raise Exception(
        "Python version above 3.9 may cause problems with SimpleITK. [BufferError: memoryview has 1 exported buffer]"
    )

try:
    import mialab.data.structure as structure
    import mialab.utilities.file_access_utilities as futil
    import mialab.utilities.pipeline_utilities as putil
except ImportError:
    # Append the MIALab root directory to Python path
    sys.path.insert(0, os.path.join(os.path.dirname(sys.argv[0]), ".."))
    import mialab.data.structure as structure
    import mialab.utilities.file_access_utilities as futil
    import mialab.utilities.pipeline_utilities as putil

LOADING_KEYS = [
    structure.BrainImageTypes.T1w,
    structure.BrainImageTypes.T2w,
    structure.BrainImageTypes.GroundTruth,
    structure.BrainImageTypes.BrainMask,
    structure.BrainImageTypes.RegistrationTransform,
]  # the list of data we will load


def compute_intersection(structure_a, structure_b):
    """Computes the intersection of two structures."""

    # Set the same origin, spacing, and direction
    structure_a.SetOrigin(structure_b.GetOrigin())
    structure_a.SetSpacing(structure_b.GetSpacing())
    structure_a.SetDirection(structure_b.GetDirection())

    # Compute the intersection
    intersection = sitk.And(structure_a, structure_b)

    # Count non-zero voxels in the intersection
    voxel_count_intersection = sitk.GetArrayFromImage(intersection).sum()

    return voxel_count_intersection


def calculate_overlap_matrix(segmentation_images):
    """Calculates the overlap matrix for the given segmentation images."""

    # Initialize the overlap matrix
    num_labels = len(segmentation_images)
    overlap_matrix = [[0] * num_labels for _ in range(num_labels)]

    # Iterate through all combinations of segmentation images
    for (i, segmentation_image_i), (j, segmentation_image_j) in itertools.product(
        enumerate(segmentation_images), repeat=2
    ):
        # Calculate the intersection (overlap) instead of voxel count difference
        intersection_count = compute_intersection(
            segmentation_image_i, segmentation_image_j
        )
        overlap_matrix[i][j] = intersection_count
        overlap_matrix[j][i] = intersection_count  # Symmetric matrix

    return overlap_matrix


def rename_folders(input_folder, legend):
    """Renames the folders in the input folder based on the legend."""
    # List all folders in the directory
    folders = [
        f
        for f in os.listdir(input_folder)
        if os.path.isdir(os.path.join(input_folder, f))
    ]

    # Iterate through the folders and rename them based on the legend
    for folder_name in folders:
        # Check if the folder name is in the legend
        if folder_name in legend:
            new_name = legend[folder_name]

            # Construct the full paths
            old_path = os.path.join(input_folder, folder_name)
            new_path = os.path.join(input_folder, new_name)

            # Rename the folder
            os.rename(old_path, new_path)

            print(f"Renamed folder: {folder_name} -> {new_name}")
        else:
            print(f"No legend entry for folder: {folder_name}")

    # Get specific folder paths
    folder_paths = [
        os.path.join(input_folder, folder_name) for folder_name in legend.values()
    ]

    return folder_paths

def subtract_overlap(larger_structure, smaller_structure):
    # Convert images to arrays
    larger_array = sitk.GetArrayFromImage(larger_structure)
    smaller_array = sitk.GetArrayFromImage(smaller_structure)

    # Identify overlapping regions
    overlap_array = np.logical_and(larger_array > 0, smaller_array > 0)

    # Set overlapping parts to 0 in the larger structure
    larger_array = np.where(overlap_array, 0, larger_array)

    # Identify non-overlapping regions and keep them in the larger structure
    larger_array = np.where(np.logical_not(overlap_array), larger_array, 0)

    return sitk.GetImageFromArray(larger_array)

def calculate_weighted_dice_scores(evaluator_results, images_prediction, runinfofile):
    # get dice scores for all labels
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
            dice_scores_per_label[label] = [result.value for result in results if
                                            result.metric == 'DICE' and result.label == label]
        return dice_scores_per_label


    def get_voxel_count_per_label(img):
        image_array = sitk.GetArrayFromImage(img)
        unique_labels = np.unique(image_array)
        # volume_per_label = []
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


    def calculate_weighted_dice_score_per_label(dice_scores_per_label, volume_per_label):

        background_label=volume_per_label["Background"]
        del volume_per_label["Background"]  # remove background
        if len(dice_scores_per_label) != len(volume_per_label):
            volume_per_label = {label: volume_per_label[label] for label in dice_scores_per_label}

        # Calculate the total voxel count
        #total_voxel_count = sum(volume_per_label.values())
        total_voxel_count = background_label

        voxel_counts_ordered = {label: volume_per_label[label] for label in dice_scores_per_label}

        weighted_dice_scores = {
            label: (total_voxel_count - voxel_counts_ordered[label])  / total_voxel_count * dice_scores_per_label[label]
            for label in dice_scores_per_label if dice_scores_per_label[label] != 0.0
        }


        return weighted_dice_scores


    dice_scores_all_labels = get_dice_scores_for_all_labels(evaluator_results)

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
    #fixme this one might be pointless
    mean_voxel_count_per_label = {label: np.mean(volumes) for label, volumes in voxel_count_per_label.items()}
    # fixme background same as total_voxels_avg???
    #del mean_voxel_count_per_label["Background"]#remove background

    weighted_dice_scores = calculate_weighted_dice_score_per_label(dice_scores_all_labels, mean_voxel_count_per_label)

    print("Weighted Dice Scores per Label:")
    for label, score in weighted_dice_scores.items():
        print(f"Label {label}: {score}")
    with open(runinfofile, 'a') as f:
        f.write("\nWeighted Dice Scores:\n")
        for label, score in weighted_dice_scores.items():
            f.write(f"Label {label}: {score}\n")

def main():
    """Overlapping calulation"""

    # Define the input and output folders
    output_folder = r".\overlapping_output"
    input_folder = r".\from_ubelix"
    os.makedirs(input_folder, exist_ok=True)
    os.makedirs(output_folder, exist_ok=True)

    # Define a legend to map folder names to structure names
    # todo get from runinfo.txt instead
    legend = {
        "2023-11-21-01-51-25": "white_matter",
        "2023-11-21-01-49-34": "grey_matter",
        "2023-11-21-01-46-42": "thalamus_matter",
        "2023-11-21-01-46-28": "hippocampus_matter",
        "2023-11-21-01-44-52": "amygdala_matter",
        # Add more mappings as needed
    }

    # Rename the folders
    folder_paths = rename_folders(input_folder, legend)

    # get patient list
    import glob

    # List all segmentation files in the label folder
    segmentation_files = glob.glob(
        os.path.join(folder_paths[0], "*.mha")
    )  # just take a random folder
    segmentation_files_basenames = [
        os.path.basename(segmentation_file) for segmentation_file in segmentation_files
    ]
    patient_list = [
        segmentation_files_basename.split("_")[0]
        for segmentation_files_basename in segmentation_files_basenames
    ]


    #################### Calculate substractions ####################
    #patient_list = [patient_list[0]]  #shorten-up # TODO remove
    # Iterate through the patients
    #patient_list = patient_list[4:]
    for patient in patient_list:
        print("==patient==: ", patient)

        # Get the segmentation files for the current patient
        segmentation_files = [
            os.path.join(label_folder, f"{patient}_seg.mha")
            for label_folder in folder_paths
        ]

        # Load the segmentation images
        segmentation_images = [
            sitk.ReadImage(segmentation_file)
            for segmentation_file in segmentation_files
        ]

        # Get a deepcopy of the segmentation images
        segmentation_images_original = copy.deepcopy(segmentation_images)

        # Calculate the overlap matrix
        overlap_matrix = calculate_overlap_matrix(segmentation_images)
        print("Overlap Matrix - original:")
        for row in overlap_matrix:
            print(row)

        # Convert to numpy arrays
        segmentation_arrays = [
            sitk.GetArrayFromImage(segmentation_image)
            for segmentation_image in segmentation_images
        ]

        # Count voxels for each structure
        structure_voxel_count = [
            np.count_nonzero(segmentation_array)
            for segmentation_array in segmentation_arrays
        ]

        # Create a folder for the patient inside the output_folder
        patient_folder = os.path.join(output_folder, patient)
        os.makedirs(patient_folder, exist_ok=True)

        #1) white 2) gray 3) thalamus 4) amygdala 5) hyppocampus
        #definition shorthand #check if order always correct todo
        white_matter = segmentation_images_original[0]
        gray_matter = segmentation_images_original[1]
        thalamus = segmentation_images_original[2]
        amygdala = segmentation_images_original[3]
        hippocampus = segmentation_images_original[4]

        #substract overlap
        white_matter = subtract_overlap(white_matter, gray_matter)
        white_matter = subtract_overlap(white_matter, thalamus)
        white_matter = subtract_overlap(white_matter, hippocampus)
        white_matter = subtract_overlap(white_matter, amygdala)

        gray_matter = subtract_overlap(gray_matter, thalamus)
        gray_matter = subtract_overlap(gray_matter, hippocampus)
        gray_matter = subtract_overlap(gray_matter, amygdala)

        thalamus = subtract_overlap(thalamus, hippocampus)
        thalamus = subtract_overlap(thalamus, amygdala)

        amygdala = subtract_overlap(amygdala, hippocampus)

        # Calculate the overlap matrix - after substraction
        segmentation_images = [white_matter, gray_matter, thalamus, hippocampus, amygdala]
        overlap_matrix = calculate_overlap_matrix(segmentation_images)
        print("Overlap Matrix - after substraction:")
        for row in overlap_matrix:
            print(row)

        # Copy information from original images
        white_matter.CopyInformation(segmentation_images_original[0])
        gray_matter.CopyInformation(segmentation_images_original[1])
        thalamus.CopyInformation(segmentation_images_original[2])
        hippocampus.CopyInformation(segmentation_images_original[3])
        amygdala.CopyInformation(segmentation_images_original[4])

        # Save the images
        sitk.WriteImage(white_matter, os.path.join(patient_folder, f'_new_white_matter.mha'))
        sitk.WriteImage(gray_matter, os.path.join(patient_folder, f'_new_gray_matter.mha'))
        sitk.WriteImage(thalamus, os.path.join(patient_folder, f'_new_thalamus.mha'))
        sitk.WriteImage(hippocampus, os.path.join(patient_folder, f'_new_hippocampus.mha'))
        sitk.WriteImage(amygdala, os.path.join(patient_folder, f'_new_amygdala.mha'))


    ##################### Calculate dice scores ####################
    import mialab.utilities.pipeline_utilities as putil
    data_atlas_dir = os.path.normpath(os.path.join(script_dir, '../data/atlas'))
    putil.load_atlas_images(data_atlas_dir)
    LOADING_KEYS = [structure.BrainImageTypes.T1w,
                    structure.BrainImageTypes.T2w,
                    structure.BrainImageTypes.GroundTruth,
                    structure.BrainImageTypes.BrainMask,
                    structure.BrainImageTypes.RegistrationTransform]
    import mialab.utilities.file_access_utilities as futil
    data_test_dir = os.path.normpath(os.path.join(script_dir, '../data/test/'))
    crawler = futil.FileSystemDataCrawler(data_test_dir,
                                          LOADING_KEYS,
                                          futil.BrainImageFilePathGenerator(),
                                          futil.DataDirectoryFilter())
    pre_process_params = {'skullstrip_pre': True,
                          'normalization_pre': True,
                          'registration_pre': True,
                          'coordinates_feature': True,
                          'intensity_feature': True,
                          'gradient_intensity_feature': True}

    #patient_list = [patient_list[0]]  # TODO remove
    for patient in patient_list:
        print("patient: ",patient)

        # Get the segmentation files for the current patient
        segmentation_files = [
            os.path.join(label_folder, f"{patient}_seg.mha")
            for label_folder in folder_paths
        ]
        # Define the order of structures
        desired_order = ['white_matter', 'grey_matter', 'hippocampus', 'amygdala', 'thalamus']
        #fixme must be in same order as labels!!!

        # Define a custom sorting key function
        def custom_sort_key(file_path):
            for index, structure_name in enumerate(desired_order, start=1):
                if structure_name in file_path:
                    return index
            # If the structure name is not found in desired_order, return a high value
            return len(desired_order) + 1

        # Sort the segmentation files based on the custom key
        sorted_segmentation_files = sorted(segmentation_files, key=custom_sort_key)

        # Load the segmentation images
        segmentation_images = [
            sitk.ReadImage(segmentation_file)
            for segmentation_file in sorted_segmentation_files
        ]

        # loop through the binary labels
        for binary_label, segmentation_image in enumerate(segmentation_images, start=1):
            print("binary_label",binary_label)
            print("init preprocess with bin label")
            data = copy.deepcopy(crawler.data)  # deepcopy to prevent overwriting trough pre_process_batch
            images_test = putil.pre_process_batch(data, pre_process_params, multi_process=False, label=binary_label)
            evaluator = putil.init_evaluator(binary_label)  # todo white matter

            for img in images_test:
                image_prediction=segmentation_image
                evaluator.evaluate(image_prediction, img.images[structure.BrainImageTypes.GroundTruth], img.id_)

            binary_folder = os.path.join(output_folder, f'{binary_label}')
            result_dir = binary_folder
            print('\nSubject-wise results...')
            os.makedirs(result_dir, exist_ok=True)
            result_file = os.path.join(result_dir, f'results_{binary_label}_{patient}.csv')
            import pymia.evaluation.writer as writer
            writer.CSVWriter(result_file).write(evaluator.results)



            print('\nSubject-wise results...')
            writer.ConsoleWriter(use_logging=True).write(evaluator.results)

            # report also mean and standard deviation among all subjects
            result_summary_file = os.path.join(result_dir, f'results_summary_{binary_label}_{patient}.csv')
            functions = {'MEAN': np.mean, 'STD': np.std}
            writer.CSVStatisticsWriter(result_summary_file, functions=functions).write(evaluator.results)
            print('\nAggregated statistic results...')
            writer.ConsoleStatisticsWriter(functions=functions).write(evaluator.results)

            WEIGHTED_DICE = True
            images_prediction = []
            images_prediction.append(image_prediction)

            # # write run info file
            runinfofile = os.path.join(result_dir, f'RunInfo_{binary_label}_{patient}.txt')
            with open(runinfofile, 'a') as f:
                f.write("General info:\n")
                f.write("label: " + str(binary_label) + "\n")

            if WEIGHTED_DICE:
                calculate_weighted_dice_scores(evaluator.results, images_prediction, runinfofile)

            # clear results such that the evaluator is ready for the next evaluation
            evaluator.clear()

if __name__ == "__main__":
    """The program's entry point."""

    script_dir = os.path.dirname(sys.argv[0])

    main()
