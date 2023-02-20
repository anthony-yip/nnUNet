from nnunet.experiment_planning.DatasetAnalyzer import DatasetAnalyzer
from batchgenerators.utilities.file_and_folder_operations import *
from nnunet.configuration import default_num_threads
from nnunet.preprocessing.cropping import get_patient_identifiers_from_cropped_files


class SS_DatasetAnalyzer(DatasetAnalyzer):
    def __init__(self, folder_with_cropped_data, overwrite=True, num_processes=default_num_threads, ignore_unlabeled=False):
        super().__init__(folder_with_cropped_data, overwrite, num_processes)
        self.folder_with_cropped_data_original = self.folder_with_cropped_data
        self.labeled_subfolder = join(self.folder_with_cropped_data, "imagesTrL")
        self.labeled_patient_identifiers = get_patient_identifiers_from_cropped_files(self.labeled_subfolder)
        self.ignore_unlabeled = ignore_unlabeled
        if not self.ignore_unlabeled:
            self.unlabeled_subfolder = join(self.folder_with_cropped_data, "imagesTrUL")
            self.unlabeled_patient_identifiers = get_patient_identifiers_from_cropped_files(self.unlabeled_subfolder)
        else:
            self.unlabeled_subfolder = None
            self.unlabeled_patient_identifiers = None

    def analyze_dataset(self, collect_intensityproperties=True):
        # operate on the data as a whole
        self.folder_with_cropped_data = self.folder_with_cropped_data_original
        # get all classes and what classes are in what patients
        # class min size
        # region size per class
        classes = self.get_classes()
        all_classes = [int(i) for i in classes.keys() if int(i) > 0]
        # modalities
        modalities = self.get_modalities()

        # operate on labeled data
        self.folder_with_cropped_data = self.labeled_subfolder
        self.patient_identifiers = self.labeled_patient_identifiers
        all_sizes, all_spacings = self.get_sizes_and_spacings_after_cropping()
        # collect intensity information
        if collect_intensityproperties:
            intensityproperties = self.collect_intensity_properties(len(modalities))
        else:
            intensityproperties = None
        # size reduction by cropping
        size_reductions = self.get_size_reduction_by_cropping()

        if not self.ignore_unlabeled:
            assert self.unlabeled_subfolder is not None
            # operate on unlabeled data
            self.folder_with_cropped_data = self.unlabeled_subfolder
            self.patient_identifiers = self.unlabeled_patient_identifiers
            unlabeled_sizes, unlabeled_spacings = self.get_sizes_and_spacings_after_cropping()
            # size reduction by cropping
            unlabeled_size_reductions = self.get_size_reduction_by_cropping()
            size_reductions.update(unlabeled_size_reductions)
            all_sizes += unlabeled_sizes
            all_spacings += unlabeled_spacings

        # return to original directory
        self.folder_with_cropped_data = self.folder_with_cropped_data_original
        dataset_properties = dict()
        dataset_properties['all_sizes'] = all_sizes
        dataset_properties['all_spacings'] = all_spacings
        dataset_properties['all_classes'] = all_classes
        dataset_properties['modalities'] = modalities  # {idx: modality name}
        dataset_properties['intensityproperties'] = intensityproperties
        # {patient_id: size_reduction}
        dataset_properties['size_reductions'] = size_reductions
        save_pickle(dataset_properties, join(self.folder_with_cropped_data, "dataset_properties.pkl"))
        return dataset_properties

