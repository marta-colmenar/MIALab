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
            enumerate(segmentation_images), repeat=2):
        # Calculate the intersection (overlap) instead of voxel count difference
        intersection_count = compute_intersection(segmentation_image_i, segmentation_image_j)
        overlap_matrix[i][j] = intersection_count
        overlap_matrix[j][i] = intersection_count  # Symmetric matrix

    return overlap_matrix


def rename_folders(input_folder, legend):
    """Renames the folders in the input folder based on the legend."""
    # List all folders in the directory
    folders = [f for f in os.listdir(input_folder) if os.path.isdir(os.path.join(input_folder, f))]

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
    folder_paths = [os.path.join(input_folder, folder_name) for folder_name in legend.values()]

    return folder_paths
def main():
    """ Overlapping calulation """

    # Define the input and output folders
    output_folder = r'.\overlapping_output'
    input_folder = r'.\from_ubelix'
    os.makedirs(input_folder, exist_ok=True)
    os.makedirs(output_folder, exist_ok=True)

    # Define a legend to map folder names to structure names
    # todo get from runinfo.txt instead
    legend = {
        '2023-11-21-01-51-25': 'white_matter',
        '2023-11-21-01-49-34': 'grey_matter',
        '2023-11-21-01-46-42': 'thalamus_matter',
        '2023-11-21-01-46-28': 'hippocampus_matter',
        '2023-11-21-01-44-52': 'amygdala_matter',
        # Add more mappings as needed
    }

    # Rename the folders
    folder_paths = rename_folders(input_folder, legend)


    # get patient list
    import glob
    # List all segmentation files in the label folder
    segmentation_files = glob.glob(os.path.join(folder_paths[0], '*.mha'))#just take a random folder
    segmentation_files_basenames=[os.path.basename(segmentation_file) for segmentation_file in segmentation_files]
    patient_list=[segmentation_files_basename.split('_')[0] for segmentation_files_basename in segmentation_files_basenames]

    # Iterate through the patients
    patient_list=[patient_list[0]]#TODO remove
    for patient in patient_list:

        print("==patient==: ", patient)

        # Get the segmentation files for the current patient
        segmentation_files = [os.path.join(label_folder, f"{patient}_seg.mha") for label_folder in folder_paths]

        # Load the segmentation images
        segmentation_images = [sitk.ReadImage(segmentation_file) for segmentation_file in segmentation_files]

        # Calculate the overlap matrix
        overlap_matrix = calculate_overlap_matrix(segmentation_images)
        print("Overlap Matrix - original:")
        for row in overlap_matrix: print(row)

        # Convert to numpy arrays
        segmentation_arrays = [sitk.GetArrayFromImage(segmentation_image) for segmentation_image in segmentation_images]

        # Count voxels for each structure
        structure_voxel_count = [np.count_nonzero(segmentation_array) for segmentation_array in segmentation_arrays]

        # Convert the structure_voxel_count list to a NumPy array
        structure_voxel_count = np.array(structure_voxel_count)

        # Get the indices that would sort the structure_voxel_count array
        sorted_indices = np.argsort(structure_voxel_count)[::-1]

        # Use the sorted indices to reorder arrays
        sorted_segmentation_matter_arrays = [segmentation_arrays[i] for i in sorted_indices]

        # print the order of the structures
        print("original order: ", list(legend.values()))
        sorted_structure_names=np.array(list(legend.values()))[sorted_indices]
        print("order sorted by voxel count (descending): ", sorted_structure_names)

        # Use the sorted indices to reorder images
        sorted_segmentation_images = [segmentation_images[i] for i in sorted_indices]

        # Create a folder for the patient inside the output_folder
        patient_folder = os.path.join(output_folder, patient)
        os.makedirs(patient_folder, exist_ok=True)


        # Subtract the overlap from the larger structure and add it to the smaller structure
        new_sorted_segmentation_matter_array_images = []

        for count, (array, image) in enumerate(
                zip(sorted_segmentation_matter_arrays[:-1], sorted_segmentation_images[:-1])):
            overlap_array = np.logical_and(array > 0, sorted_segmentation_matter_arrays[count + 1] > 0)

            # Conditionally subtract overlap from the larger structure
            # new_array = array - overlap_array
            #new_array = array & ~overlap_array
            #new_array = np.where(overlap_array, array & ~overlap_array, array)
            new_array = np.where(overlap_array, 0, array)

            # Conditionally add overlap to the smaller structure using logical_or
            #sorted_segmentation_matter_arrays[count + 1] += overlap_array
            #sorted_segmentation_matter_arrays[count + 1] &= ~overlap_array
            #sorted_segmentation_matter_arrays[count + 1] |= overlap_array
            sorted_segmentation_matter_arrays[count + 1] = np.where(overlap_array, 1,
                                                                    sorted_segmentation_matter_arrays[count + 1])

            new_image = sitk.GetImageFromArray(new_array)
            new_image.CopyInformation(image)

            new_sorted_segmentation_matter_array_images.append(new_image)

            output_filename = os.path.join(patient_folder, f'_new_{sorted_structure_names[count]}.mha')
            sitk.WriteImage(new_image, output_filename)

        last_array = sorted_segmentation_matter_arrays[-1]
        last_image = sitk.GetImageFromArray(last_array)
        last_image.CopyInformation(sorted_segmentation_images[-1])

        # Construct the output filename for the last image
        last_output_filename = os.path.join(patient_folder, f'_new_{sorted_structure_names[-1]}.mha')

        # Save the last image to the specified filename
        sitk.WriteImage(last_image, last_output_filename)

        # Append the last image to the list if needed
        new_sorted_segmentation_matter_array_images.append(last_image)

        # Calculate the overlap matrix
        overlap_matrix = calculate_overlap_matrix(new_sorted_segmentation_matter_array_images)
        print("Overlap Matrix - after substraction:")
        for row in overlap_matrix:
            print(row)




if __name__ == "__main__":
    """The program's entry point."""

    script_dir = os.path.dirname(sys.argv[0])

    main()

