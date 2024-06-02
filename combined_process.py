import json
import os
import subprocess
from pathlib import Path


import numpy as np
import SimpleITK as sitk
import torch
from evalutils import SegmentationAlgorithm
from evalutils.validators import (UniqueImagesValidator,
                                  UniquePathIndicesValidator)
from picai_baseline.unet.training_setup.default_hyperparam import \
    get_default_hyperparams
from picai_baseline.unet.training_setup.neural_network_selector import \
    neural_network_for_run
from picai_baseline.unet.training_setup.preprocess_utils import z_score_norm
from picai_prep.data_utils import atomic_image_write
from picai_prep.preprocessing import Sample, PreprocessingSettings, crop_or_pad, resample_img
from report_guided_annotation import extract_lesion_candidates
from scipy.ndimage import gaussian_filter


class MissingSequenceError(Exception):
    """Exception raised when a sequence is missing."""
    def __init__(self, name, folder):
        message = f"Could not find scan for {name} in {folder} (files: {os.listdir(folder)})"
        super().__init__(message)


class MultipleScansSameSequencesError(Exception):
    """Exception raised when multiple scans of the same sequences are provided."""
    def __init__(self, name, folder):
        message = f"Found multiple scans for {name} in {folder} (files: {os.listdir(folder)})"
        super().__init__(message)


def strip_metadata(img: sitk.Image) -> None:
    for key in img.GetMetaDataKeys():
        img.EraseMetaData(key)


class csPCaAlgorithm(SegmentationAlgorithm):
    """
    Wrapper to deploy trained baseline U-Net model from
    https://github.com/DIAGNijmegen/picai_baseline as a
    grand-challenge.org algorithm.
    """

    def __init__(self):
        super().__init__(
            validators=dict(
                input_image=(
                    UniqueImagesValidator(),
                    UniquePathIndicesValidator(),
                )
            ),
        )

        # set expected i/o paths in gc env (image i/p, algorithms, prediction o/p)
        # see grand-challenge.org/algorithms/interfaces/ for expected path per i/o interface
        # note: these are fixed paths that should not be modified

        # directory to model weights
        self.algorithm_weights_dir = Path("/opt/algorithm/weights/")

        # path to image files
        self.image_input_dirs = [
            "/input/images/transverse-t2-prostate-mri/",
            "/input/images/transverse-adc-prostate-mri/",
            "/input/images/transverse-hbv-prostate-mri/",
            # "/input/images/coronal-t2-prostate-mri/",  # not used in this algorithm
            # "/input/images/sagittal-t2-prostate-mri/"  # not used in this algorithm
        ]
        self.image_input_paths = [list(Path(x).glob("*.mha"))[0] for x in self.image_input_dirs]

        # load clinical information
        with open("/input/clinical-information-prostate-mri.json") as fp:
            self.clinical_info = json.load(fp)

        # path to output files
        self.detection_map_output_path = Path("/output/images/cspca-detection-map/cspca_detection_map.mha")
        self.case_level_likelihood_output_file = Path("/output/cspca-case-level-likelihood.json")

        # create output directory
        self.detection_map_output_path.parent.mkdir(parents=True, exist_ok=True)

        # define compute used for training/inference ('cpu' or 'cuda')
        if torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'

        # extract available clinical metadata (not used for this example)
        self.age = self.clinical_info["patient_age"]

        if "PSA_report" in self.clinical_info:
            self.psa = self.clinical_info["PSA_report"]
        else:
            self.psa = None  # value is missing, if not reported

        if "PSAD_report" in self.clinical_info:
            self.psad = self.clinical_info["PSAD_report"]
        else:
            self.psad = None  # value is missing, if not reported

        if "prostate_volume_report" in self.clinical_info:
            self.prostate_volume = self.clinical_info["prostate_volume_report"]
        else:
            self.prostate_volume = None  # value is missing, if not reported

        # extract available acquisition metadata (not used for this example)
        self.scanner_manufacturer = self.clinical_info["scanner_manufacturer"]
        self.scanner_model_name = self.clinical_info["scanner_model_name"]
        self.diffusion_high_bvalue = self.clinical_info["diffusion_high_bvalue"]

        # define input data specs [image shape, spatial res, num channels, num classes]
        self.img_spec = {
            'image_shape': [20, 128, 128],
            'spacing': [3.0, 0.5, 0.5],
            'num_channels': 3,
            'num_classes': 2,
        }

        # load trained algorithm architecture + weights
        self.models = []
        model_arch = ['unet']
        model_folds = [range(5)]

        # for each given architecture
        for arch_name, folds in zip(model_arch, model_folds):

            # for each trained 5-fold instance of a given architecture
            for fold in folds:
                # path to trained weights for this architecture + fold (e.g. 'unet_F4.pt')
                checkpoint_path = self.algorithm_weights_dir / f'{arch_name}_F{fold}.pt'

                # skip if model was not trained for this fold
                if not checkpoint_path.exists():
                    continue

                # define the model specifications used for initialization at train-time
                # note: if the default hyperparam listed in picai_baseline was used,
                # passing arguments 'image_shape', 'num_channels', 'num_classes' and
                # 'model_type' via function 'get_default_hyperparams' is enough.
                # otherwise arguments 'model_strides' and 'model_features' must also
                # be explicitly passed directly to function 'neural_network_for_run'
                args = get_default_hyperparams({
                    'model_type': arch_name,
                    **self.img_spec
                })

                model = neural_network_for_run(args=args, device=self.device)

                # load trained weights for the fold
                checkpoint = torch.load(checkpoint_path)
                print(f"Loading weights from {checkpoint_path}")
                model.load_state_dict(checkpoint['model_state_dict'])
                model.to(self.device)
                self.models += [model]
                print("Complete.")
                print("-"*100)

        # display error/success message
        if len(self.models) == 0:
            raise Exception("No models have been found/initialized.")
        else:
            print(f"Success! {len(self.models)} model(s) have been initialized.")
            print("-"*100)

    # generate + save predictions, given images
    def predict(self):

        print("Preprocessing Images ...")

        # First, segment the prostate and align the images
        prostate_segmentation_path = self.segment_prostate_and_align_images()

        # TODO: load the segmentation
        seg = ProstateSegmentationAlgorithm().process()

        # TODO: adapt https://github.com/DIAGNijmegen/picai_prep/blob/b0c5fb00fe9d150987a92f457dee3c5f11df395e/src/picai_prep/preprocessing.py#L107
        # to accept/use the centroid. getOrigin is bottom left corner // centroid(x, y, z) = half
        # tip: sitk.TransformContinuousIndexToPhysicalPoint

        # read images (axial sequences used for this example only)
        sample = Sample(
            scans=[
                sitk.ReadImage(str(path))
                for path in self.image_input_paths
            ],
            settings=PreprocessingSettings(
                matrix_size=self.img_spec['image_shape'],
                spacing=self.img_spec['spacing'],
                align_segmentation=seg,
            )
        )

        # preprocess - align, center-crop, resample
        sample.preprocess()
        cropped_img = [
            sitk.GetArrayFromImage(x)
            for x in sample.scans
        ]

        # preprocessing - intensity normalization + expand channel dim
        preproc_img = [
            z_score_norm(np.expand_dims(x, axis=0), percentile=99.5)
            for x in cropped_img
        ]

        # preprocessing - concatenate + expand batch dim
        preproc_img = np.expand_dims(np.vstack(preproc_img), axis=0)

        # preprocessing - convert to PyTorch tensor
        preproc_img = torch.from_numpy(preproc_img)

        print("Complete.")

        # test-time augmentation (horizontal flipping)
        img_for_pred = [preproc_img.to(self.device)]
        img_for_pred += [torch.flip(preproc_img, [4]).to(self.device)]

        # begin inference
        outputs = []
        print("Generating Predictions ...")

        # for each member model in ensemble
        for p in range(len(self.models)):

            # switch model to evaluation mode
            self.models[p].eval()

            # scope to disable gradient updates
            with torch.no_grad():
                # aggregate predictions for all tta samples
                preds = [
                    torch.sigmoid(self.models[p](x))[:, 1, ...].detach().cpu().numpy()
                    for x in img_for_pred
                ]

                # revert horizontally flipped tta image
                preds[1] = np.flip(preds[1], [3])

                # gaussian blur to counteract checkerboard artifacts in
                # predictions from the use of transposed conv. in the U-Net
                outputs += [
                    np.mean([
                        gaussian_filter(x, sigma=1.5)
                        for x in preds
                    ], axis=0)[0]
                ]

        # ensemble softmax predictions
        ensemble_output = np.mean(outputs, axis=0).astype('float32')

        # read and resample images (used for reverting predictions only)
        sitk_img = [
            sitk.ReadImage(str(path)) for path in self.image_input_paths
        ]
        resamp_img = [
            sitk.GetArrayFromImage(
                resample_img(x, out_spacing=self.img_spec['spacing'])
            )
            for x in sitk_img
        ]

        # revert softmax prediction to original t2w - reverse center crop
        cspca_det_map_sitk: sitk.Image = sitk.GetImageFromArray(crop_or_pad(
            ensemble_output, size=resamp_img[0].shape))
        cspca_det_map_sitk.SetSpacing(list(reversed(self.img_spec['spacing'])))

        # revert softmax prediction to original t2w - reverse resampling
        cspca_det_map_sitk = resample_img(cspca_det_map_sitk,
                                          out_spacing=list(reversed(sitk_img[0].GetSpacing())))

        # process softmax prediction to detection map
        cspca_det_map_npy = extract_lesion_candidates(
            sitk.GetArrayFromImage(cspca_det_map_sitk), threshold='dynamic')[0]

        # remove (some) secondary concentric/ring detections
        cspca_det_map_npy[cspca_det_map_npy<(np.max(cspca_det_map_npy)/5)] = 0

        # make sure that expected shape was matched after reverse resampling (can deviate due to rounding errors)
        cspca_det_map_sitk: sitk.Image = sitk.GetImageFromArray(crop_or_pad(
            cspca_det_map_npy, size=sitk.GetArrayFromImage(sitk_img[0]).shape))

        # works only if the expected shape matches
        cspca_det_map_sitk.CopyInformation(sitk_img[0])

        # save detection map
        atomic_image_write(cspca_det_map_sitk, self.detection_map_output_path)

        # save case-level likelihood
        with open(str(self.case_level_likelihood_output_file), 'w') as f:
            json.dump(float(np.max(cspca_det_map_npy)), f)

    def segment_prostate_and_align_images(self):
        # This method performs prostate segmentation and image alignment

        # Initialize paths
        prostate_segmentation_path = Path("/output/images/transverse-whole-prostate-mri/prostate_gland.mha")
        nnunet_inp_dir = Path("/opt/algorithm/nnunet/input")
        nnunet_out_dir = Path("/opt/algorithm/nnunet/output")
        nnunet_results = Path("/opt/algorithm/results")

        # Ensure required folders exist
        nnunet_inp_dir.mkdir(exist_ok=True, parents=True)
        nnunet_out_dir.mkdir(exist_ok=True, parents=True)
        prostate_segmentation_path.parent.mkdir(exist_ok=True, parents=True)

        # Input validation for multiple inputs
        scan_glob_format = "*.mha"
        scan_paths = []
        for folder in self.image_input_dirs:
            file_paths = list(Path(folder).glob(scan_glob_format))
            if len(file_paths) == 0:
                raise MissingSequenceError(name=folder.split("/")[-1], folder=folder)
            elif len(file_paths) >= 2:
                raise MultipleScansSameSequencesError(name=folder.split("/")[-1], folder=folder)
            else:
                # Append scan path to algorithm input paths
                scan_paths += [file_paths[0]]

        # Set up Sample
        sample = Sample(
            scans=[
                sitk.ReadImage(str(path))
                for path in scan_paths
            ],
            settings=PreprocessingSettings(
                physical_size=[81.0, 192.0, 192.0],
                crop_only=True
            )
        )

        # Perform preprocessing
        sample.preprocess()

        # Write preprocessed scans to nnUNet input directory
        for i, scan in enumerate(sample.scans):
            path = nnunet_inp_dir / f"scan_{i:04d}.nii.gz"
            atomic_image_write(scan, path)

        # Perform inference using nnUNet
        self.predict_prostate_segmentation(
            task="Task2202_prostate_segmentation",
            trainer="nnUNetTrainerV2_Loss_FL_and_CE_checkpoints",
            checkpoint="model_final_checkpoint",
            nnunet_inp_dir=nnunet_inp_dir,
            nnunet_out_dir=nnunet_out_dir,
            nnunet_results=nnunet_results
        )

        # Read binarized and postprocessed prediction
        pred_path = str(nnunet_out_dir / "scan.nii.gz")
        pred: sitk.Image = sitk.ReadImage(pred_path)

        # Transform prediction to original space
        reference_scan = sitk.ReadImage(str(scan_paths[0]))
        pred = resample_to_reference_scan(pred, reference_scan_original=reference_scan)

        # Remove metadata to get rid of SimpleITK warning
        strip_metadata(pred)

        # Save prediction to output folder
        atomic_image_write(pred, str(prostate_segmentation_path))

        return prostate_segmentation_path

    def predict_prostate_segmentation(self, task, trainer="nnUNetTrainerV2", network="3d_fullres",
                                      checkpoint="model_final_checkpoint", folds="0,1,2,3,4",
                                      store_probability_maps=True, disable_augmentation=False,
                                      disable_patch_overlap=False, nnunet_inp_dir=None,
                                      nnunet_out_dir=None, nnunet_results=None):
        """
        Use trained nnUNet network to generate segmentation masks
        """

        # Set environment variables
        os.environ['RESULTS_FOLDER'] = str(nnunet_results)

        # Run prediction script
        cmd = [
            'nnUNet_predict',
            '-t', task,
            '-i', str(nnunet_inp_dir),
            '-o', str(nnunet_out_dir),
            '-m', network,
            '-tr', trainer,
            '--num_threads_preprocessing', '2',
            '--num_threads_nifti_save', '1'
        ]

        if folds:
            cmd.append('-f')
            cmd.extend(folds.split(','))

        if checkpoint:
            cmd.append('-chk')
            cmd.append(checkpoint)

        if store_probability_maps:
            cmd.append('--save_npz')

        if disable_augmentation:
            cmd.append('--disable_tta')

        if disable_patch_overlap:
            cmd.extend(['--step_size', '1'])

        subprocess.check_call(cmd)


if __name__ == "__main__":
    csPCaAlgorithm().predict()