import torchio as tio
import nibabel as nib
def resample_nii(input_path: str, output_path: str, target_spacing: tuple = (1.5, 1.5, 1.5), n=None, reference_image=None):
    """
    Resample a nii.gz file to a specified spacing using torchio.

    Parameters:
    - input_path: Path to the input .nii.gz file.
    - output_path: Path to save the resampled .nii.gz file.
    - target_spacing: Desired spacing for resampling. Default is (1.5, 1.5, 1.5).
    """

    # Load the nii.gz file using torchio
    subject = tio.Subject(
        img=tio.ScalarImage(input_path)
    )

    # Resample the image to the target spacing
    resampler = tio.Resample(target=target_spacing)
    resampled_subject = resampler(subject)
    if(n!=None):
        image = resampled_subject.img
        tensor_data = image.data
        if(isinstance(n, int)):
            n = [n]
        for ni in n:
            tensor_data[tensor_data == ni] = -1
        tensor_data[tensor_data != -1] = 0
        tensor_data[tensor_data != 0] = 1
        save_image = tio.ScalarImage(tensor=tensor_data, affine=image.affine)
        reference_size = reference_image.shape[1:]  # omitting the channel dimension
        # padding_amount = [ref_dim - in_dim for ref_dim, in_dim in zip(reference_size, save_image.shape[1:])] + [0,0,0]
        # padding_transform = tio.Pad(padding_amount)
        cropper_or_padder = tio.CropOrPad(reference_size)
        save_image = cropper_or_padder(save_image)
    else:
        save_image = resampled_subject.img

    # Save the resampled image to the specified output path
    save_image.save(output_path)
imagepath = ''
labelpath = ''
targetimagepath = ''
targetlabelpath = ''
gt_img = nib.load(labelpath)
spacing = tuple(gt_img.header['pixdim'][1:4])
spacing_voxel = spacing[0] * spacing[1] * spacing[2]
gt_arr = gt_img.get_fdata()
volume = gt_arr.sum()*spacing_voxel
if(volume<10):
    print("the volume of label is too small")
else:
    resample_nii(imagepath,targetimagepath)
    reference_image = tio.ScalarImage(targetimagepath)
    resample_nii(labelpath, targetlabelpath, n=1, reference_image=reference_image)
