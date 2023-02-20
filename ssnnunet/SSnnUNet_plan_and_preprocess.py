import nnunet
from batchgenerators.utilities.file_and_folder_operations import *
from nnunet.paths import *
import shutil
from nnunet.utilities.task_name_id_conversion import convert_id_to_task_name
from ssnnunet.SS_DatasetAnalyzer import SS_DatasetAnalyzer
from nnunet.preprocessing.sanity_checks import verify_dataset_integrity
from nnunet.training.model_restore import recursive_find_python_class


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", help="Task id for task you wish to run planning and preprocessing for. Task must have"
                                   " a corresponding Task_XXX folder in nnUNet_raw_data")
    parser.add_argument("-c", type=int, default=6, help="Number of CPU cores you would like to use for preprocessing.")
    parser.add_argument("-pl", type=str, default="SS_ExperimentPlanner3D_v21", help="Name of the Experiment Planner class you would like to use.")
    parser.add_argument("-s", type=int, default=36, help="Amount of GPU memory")
    parser.add_argument("-u", action='store_true', help="Set this flag to ignore unlabeled data")
    args = parser.parse_args()
    task_id = int(args.t)
    task_name = convert_id_to_task_name(task_id)
    planner_name = args.pl
    cores = args.c
    vram = args.s
    ignore_unlabeled = args.u

    search_in = join(nnunet.__path__[0], "experiment_planning")
    planner_3d = recursive_find_python_class([search_in], planner_name, current_module="nnunet.experiment_planning")
    if planner_3d is None:
        raise RuntimeError("Could not find the Planner class %s. Make sure it is located somewhere in "
                           "nnunet.experiment_planning" % planner_name)

    #do planning
    print("\n\n\n", task_name)
    cropped_out_dir = os.path.join(nnUNet_cropped_data, task_name)
    preprocessing_output_dir_this_task = os.path.join(preprocessing_output_dir, task_name)

    # we need to figure out if we need the intensity propoerties. We collect them only if one of the modalities is CT
    dataset_json = load_json(join(cropped_out_dir, 'dataset.json'))
    modalities = list(dataset_json["modality"].values())
    collect_intensityproperties = True if (("CT" in modalities) or ("ct" in modalities)) else False

    #analyze dataset with SS_DatasetAnalyzer
    print("analyzing dataset, creating dataset fingerprint")
    # this class creates the fingerprint
    dataset_analyzer = SS_DatasetAnalyzer(cropped_out_dir, overwrite=True,
                                       num_processes=cores, ignore_unlabeled=ignore_unlabeled)
    _ = dataset_analyzer.analyze_dataset(
        collect_intensityproperties)  # this will write output files that will be used by the ExperimentPlanner

    maybe_mkdir_p(preprocessing_output_dir_this_task)
    shutil.copy(join(cropped_out_dir, "dataset_properties.pkl"), preprocessing_output_dir_this_task)
    shutil.copy(join(nnUNet_raw_data, task_name, "dataset.json"), preprocessing_output_dir_this_task)
    maybe_mkdir_p(join(preprocessing_output_dir_this_task, "imagesTrL"))
    if not ignore_unlabeled:
        maybe_mkdir_p(join(preprocessing_output_dir_this_task, "imagesTrUL"))
    maybe_mkdir_p(join(preprocessing_output_dir_this_task, "imagesVal"))

    print("number of threads: ", cores, "\n")
    if planner_3d is not None:
        exp_planner = planner_3d(cropped_out_dir, preprocessing_output_dir_this_task, vram, ignore_unlabeled=ignore_unlabeled)
        print("planning experiment for 3D:")
        exp_planner.plan_experiment()

        print("\nrunning preprocessing for 3D:")
        exp_planner.run_preprocessing((cores, cores))


if __name__ == "__main__":
    main()