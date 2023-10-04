from nnunet.utilities.task_name_id_conversion import convert_id_to_task_name
from ssnnunet.SS_utils import crop


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", help="Task id for task you wish to run cropping for. Task must have a corresponding "
                                   "Task_XXX folder in nnUNet_raw_data")
    # parser.add_argument("-s", help="Subset of the data you wish to crop. Can be either TrL, TrUL or Val.")
    parser.add_argument("-c", default=6, help="Number of CPU cores you would like to use for cropping.")
    parser.add_argument("-u", action='store_true', help="Set this flag to ignore unlabeled data")
    args = parser.parse_args()
    task_id = int(args.t)
    ignore_unlabeled = args.u
    task_name = convert_id_to_task_name(task_id)
    crop(task_name, num_threads=args.c, ignore_unlabeled=ignore_unlabeled)


if __name__ == "__main__":
    main()