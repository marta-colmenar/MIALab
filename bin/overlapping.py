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
import copy
import itertools

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

def compute_overlap_and_subtract(larger_structure, smaller_structure):
    # Threshold to create binary masks
    threshold1 = sitk.BinaryThreshold(larger_structure, lowerThreshold=0.5)
    larger_binary = sitk.Cast(threshold1, sitk.sitkUInt8)

    threshold2 = sitk.BinaryThreshold(smaller_structure, lowerThreshold=0.5)
    smaller_binary = sitk.Cast(threshold2, sitk.sitkUInt8)

    # Compute the intersection
    intersection = sitk.And(larger_binary, smaller_binary)

    # Subtract the overlapping region from the larger structure
    larger_structure -= intersection

    return intersection, larger_structure
def load_and_average_mha_files(folder_path):
    files = [f for f in os.listdir(folder_path) if f.endswith('.mha')]
    images = [sitk.ReadImage(os.path.join(folder_path, file)) for file in files]

    # Ensure all images have the same origin, spacing, and direction
    reference_image = images[0]
    for image in images[1:]:
        if image.GetOrigin() != reference_image.GetOrigin() or \
           image.GetSpacing() != reference_image.GetSpacing() or \
           image.GetDirection() != reference_image.GetDirection():
            raise RuntimeError("Images do not occupy the same physical space.")

    # Assuming all images have the same size, use the size of the first image
    size = reference_image.GetSize()
    average_image = sitk.Image(size, sitk.sitkFloat32)

    # Set the physical space of the average image based on the first image
    average_image = sitk.Image(images[0].GetSize(), sitk.sitkFloat32)
    average_image.SetOrigin(images[0].GetOrigin())
    average_image.SetSpacing(images[0].GetSpacing())
    average_image.SetDirection(images[0].GetDirection())

    # Create a separate variable for the sum
    sum_image = sitk.Image(images[0].GetSize(), sitk.sitkFloat32)
    sum_image.SetOrigin(images[0].GetOrigin())
    sum_image.SetSpacing(images[0].GetSpacing())
    sum_image.SetDirection(images[0].GetDirection())

    for image in images:
        # Convert to float for averaging
        image_float = sitk.Cast(image, sitk.sitkFloat32)
        average_image += image_float

    # Calculate the average
    average_image /= len(images)

    return average_image

def compute_overlap_and_subtract(seg1, seg2):
    # Threshold to create binary masks
    threshold1 = sitk.BinaryThreshold(seg1, lowerThreshold=0.5)
    seg1_binary = sitk.Cast(threshold1, sitk.sitkUInt8)

    threshold2 = sitk.BinaryThreshold(seg2, lowerThreshold=0.5)
    seg2_binary = sitk.Cast(threshold2, sitk.sitkUInt8)

    # Compute the intersection
    intersection = sitk.And(seg1_binary, seg2_binary)

    # Cast intersection to 32-bit float
    intersection_float = sitk.Cast(intersection, sitk.sitkFloat32)

    # Compute the size of the intersection region in voxels
    overlap_size = sitk.GetArrayFromImage(intersection_float).sum()

    # Determine the smaller and larger structures
    if seg1.GetSize() < seg2.GetSize():
        smaller_structure = seg1
        larger_structure = seg2
    else:
        smaller_structure = seg2
        larger_structure = seg1

    # Cast smaller_structure to 32-bit float
    smaller_structure_float = sitk.Cast(smaller_structure, sitk.sitkFloat32)


    # Subtract the overlapping region from the smaller structure
    smaller_structure_float -= intersection_float

    return overlap_size, smaller_structure_float
def dice_similarity_coefficient(seg1, seg2):
    # Threshold to create binary masks
    threshold = sitk.BinaryThreshold(seg1, lowerThreshold=0.5)
    seg1_binary = sitk.Cast(threshold, sitk.sitkUInt8)

    threshold = sitk.BinaryThreshold(seg2, lowerThreshold=0.5)
    seg2_binary = sitk.Cast(threshold, sitk.sitkUInt8)

    # Perform AND operation on binary masks
    intersection = sitk.And(seg1_binary, seg2_binary)

    # Calculate Dice coefficient
    dice = (2.0 * sitk.GetArrayFromImage(intersection).sum()) / (
                sitk.GetArrayFromImage(seg1_binary).sum() + sitk.GetArrayFromImage(seg2_binary).sum())

    return dice

def calculate_pairwise_dice_similarity(average_images):
    num_labels = len(average_images)
    dice_matrix = [[0.0] * num_labels for _ in range(num_labels)]

    for i, j in itertools.combinations(range(num_labels), 2):
        dice_value = dice_similarity_coefficient(average_images[i], average_images[j])
        dice_matrix[i][j] = dice_value
        dice_matrix[j][i] = dice_value

    return dice_matrix

def calculate_pairwise_overlap_substr(structures):
    num_structures = len(structures)
    overlap_matrix = [[0] * num_structures for _ in range(num_structures)]

    for i, j in itertools.combinations(range(num_structures), 2):
        overlap_size, smaller_structure = compute_overlap_and_subtract(structures[i], structures[j])
        overlap_matrix[i][j] = overlap_size
        overlap_matrix[j][i] = overlap_size

        # Optionally, you can use or save the modified smaller_structure

    return overlap_matrix

def compute_overlap_size(seg1, seg2):
    # Threshold to create binary masks
    threshold1 = sitk.BinaryThreshold(seg1, lowerThreshold=0.5)
    seg1_binary = sitk.Cast(threshold1, sitk.sitkUInt8)

    threshold2 = sitk.BinaryThreshold(seg2, lowerThreshold=0.5)
    seg2_binary = sitk.Cast(threshold2, sitk.sitkUInt8)

    # Compute the intersection
    intersection = sitk.And(seg1_binary, seg2_binary)

    # Compute the size of the intersection region in voxels
    overlap_size = sitk.GetArrayFromImage(intersection).sum()

    return overlap_size
def calculate_pairwise_overlap(structures):
    num_structures = len(structures)
    overlap_matrix = [[0] * num_structures for _ in range(num_structures)]

    for i, j in itertools.combinations(range(num_structures), 2):
        overlap_size = compute_overlap_size(structures[i], structures[j])
        overlap_matrix[i][j] = overlap_size
        overlap_matrix[j][i] = overlap_size

    return overlap_matrix

def calculate_voxel_size(structure):
    spacing = structure.GetSpacing()
    voxel_size = np.prod(spacing)
    return voxel_size
def calculate_and_save_overlaps_old(structures, output_folder):
    # Sort structures by voxel size in descending order
    structures.sort(key=calculate_voxel_size, reverse=True)

    num_structures = len(structures)
    overlap_matrix = [[None] * num_structures for _ in range(num_structures)]

    for i in range(num_structures):
        for j in range(i + 1, num_structures):
            overlap_region, structures[i] = compute_overlap_and_subtract(structures[i], structures[j])
            overlap_matrix[i][j] = overlap_region

    # Save the modified structures as .mha files
    for i, structure in enumerate(structures):
        output_path = os.path.join(output_folder, f"modified_structure_{i}.mha")
        sitk.WriteImage(structure, output_path )

    return overlap_matrix

def calculate_and_save_overlaps(label_folders, output_folder):
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    num_labels = len(label_folders)
    overlap_matrix = [[None] * num_labels for _ in range(num_labels)]

    for i, label_folder_i in enumerate(label_folders):
        structures_i = []
        for label_path_i in os.listdir(label_folder_i):
            if label_path_i.endswith(".mha"):  # Only consider image files
                structure_i = sitk.ReadImage(os.path.join(label_folder_i, label_path_i))
                structures_i.append(structure_i)

        for j, label_folder_j in enumerate(label_folders):
            if i != j:
                structures_j = []
                for label_path_j in os.listdir(label_folder_j):
                    if label_path_j.endswith(".mha"):  # Only consider image files
                        structure_j = sitk.ReadImage(os.path.join(label_folder_j, label_path_j))
                        structures_j.append(structure_j)

                for idx, (structure_i, structure_j) in enumerate(zip(structures_i, structures_j)):
                    overlap_region, structures_i[idx] = compute_overlap_and_subtract(structure_i, structure_j)
                    overlap_matrix[i][j] = overlap_region

        # Save the modified structures for each label and patient
        for idx, (structure_i, patient_id_i) in enumerate(zip(structures_i, os.listdir(label_folder_i))):
            output_path = os.path.join(output_folder, os.path.basename(label_folder_i), patient_id_i)
            os.makedirs(output_path, exist_ok=True)
            output_path = os.path.join(output_path, f"modified_structure_{idx}.mha")
            sitk.WriteImage(structure_i, output_path)

    return overlap_matrix
def main(result_dir: str, data_atlas_dir: str, data_train_dir: str, data_test_dir: str, config_section: str):
    """ Overlapping calulation """

    output_folder = r"C:\Users\FlipFlop\Documents\UniBE\Sem5_MedImLab\Repo\MIALab\overlapping_output"

    #get folders
    white_matter_folder = r'C:\Users\FlipFlop\Documents\UniBE\Sem5_MedImLab\Repo\MIALab\fromubelix\2023-11-21-01-51-25'
    grey_matter_folder = r'C:\Users\FlipFlop\Documents\UniBE\Sem5_MedImLab\Repo\MIALab\fromubelix\2023-11-21-01-49-34'
    thalamus_matter_folder = r'C:\Users\FlipFlop\Documents\UniBE\Sem5_MedImLab\Repo\MIALab\fromubelix\2023-11-21-01-46-42'
    hippocampus_matter_folder =r'C:\Users\FlipFlop\Documents\UniBE\Sem5_MedImLab\Repo\MIALab\fromubelix\2023-11-21-01-46-28'
    amygdala_matter_folder=r'C:\Users\FlipFlop\Documents\UniBE\Sem5_MedImLab\Repo\MIALab\fromubelix\2023-11-21-01-44-52'

    label_folders = [white_matter_folder, grey_matter_folder, thalamus_matter_folder, hippocampus_matter_folder,
                       amygdala_matter_folder]

    overlap_matrix = calculate_and_save_overlaps(label_folders, output_folder)


    # Print or use the overlap_matrix as needed
    print("Overlapping Regions (larger to smaller structures):")
    for row in overlap_matrix:
        print(row)




if __name__ == "__main__":
    """The program's entry point."""

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

    parser.add_argument(
        '--config_section',
        type=str,
        default="default",
        help='Choose config file section.'
    )

    args = parser.parse_args()
    main(
            args.result_dir,
            args.data_atlas_dir,
            args.data_train_dir,
            args.data_test_dir,
            args.config_section
        )

