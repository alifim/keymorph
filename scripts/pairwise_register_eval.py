import os
import torch
import numpy as np
import torchio as tio
import nibabel as nib

from keymorph.utils import align_img, one_hot_eval, one_hot_eval_abdominal, one_hot
from keymorph.viz_tools import imshow_registration_2d, imshow_registration_3d
from keymorph.augmentation import affine_augment, random_affine_augment
import keymorph.loss_ops as loss_ops
from keymorph.pytorch_ssim import SSIM3D
from keymorph.loss_ops import DiceOrgan

from scripts.script_utils import (
    load_dict_from_json,
    save_dict_as_json,
    parse_test_aug,
)

@torch.no_grad()
def run_eval(
    loader,
    registration_model,
    list_of_eval_metrics,
    list_of_eval_names,
    list_of_eval_augs,
    list_of_eval_aligns,
    args,
    save_dir_prefix="eval",
):
    registration_model.eval()

    def _build_metric_dict(names):
        list_of_all_test = []
        for m in list_of_eval_metrics:
            for a in list_of_eval_augs:
                for k in list_of_eval_aligns:
                    for n in names:
                        n1, n2 = n
                        list_of_all_test.append(f"{m}:{n1}:{n2}:{a}:{k}")
        _metrics = {}
        _metrics.update({key: [] for key in list_of_all_test})
        return _metrics

    print(
        list_of_eval_metrics,
        list_of_eval_augs,
        list_of_eval_aligns,
        args.save_dir,
    )

    test_metrics = _build_metric_dict(list_of_eval_names)
    for i, (fixed, moving) in enumerate(loader): # TODO CHANGED [0] for whole eval inference

        if args.early_stop_eval_subjects and i == args.early_stop_eval_subjects:
            break
        for aug in list_of_eval_augs:
            param = parse_test_aug(aug)
            mod1 = fixed["modality"][0]
            mod2 = moving["modality"][0]
            print(
                f"\n\n\nRunning test: subject pair {i}, mod {mod1}->{mod2}, aug {aug}"
            )

            # Create directory to save images, segs, points, metrics
            mod1_str = "-".join(mod1.split("/")[-2:])
            mod2_str = "-".join(mod2.split("/")[-2:])
            save_dir = (
                args.model_eval_dir / save_dir_prefix / f"{i}_{mod1_str}_{mod2_str}"
            )
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            # Load metrics (for all alignment types) if they exist, else run registration
            all_metrics_paths = {
                align_type_str: save_dir / f"metrics-{aug}-{align_type_str}.json"
                for align_type_str in list_of_eval_aligns
            }
            if (
                all([os.path.exists(p) for p in all_metrics_paths.values()])
                and args.skip_if_completed
            ):
                print(
                    f"Found metrics for all alignments, skipping running registration..."
                )
                all_metrics = {
                    k: load_dict_from_json(v) for k, v in all_metrics_paths.items()
                }

            else:
                img_f, img_m = (
                    fixed["img"][tio.DATA],
                    moving["img"][tio.DATA],
                )
                aff_f, aff_m = (
                    fixed["img"]["affine"],
                    moving["img"]["affine"],
                )
                print("img_f prod:" + str(np.prod(img_f.shape)))
                if np.prod(img_f.shape) >= 77594624: # TO DELETE
                    print("Skipping large image")
                    print("img_f path:"+ str(fixed['img']['path']))
                    continue


                print("img_m prod:" + str(np.prod(img_m.shape)))
                if np.prod(img_m.shape) >= 77594624:
                    print("Skipping large image")
                    print("img_m path:"+ str(moving['img']['path']))
                    continue

                if args.seg_available:
                    seg_f, seg_m = (
                        fixed["seg"][tio.DATA],
                        moving["seg"][tio.DATA],
                    )
                    
                    print("\nBefore processing:")
                    print(f"seg_f stats: min={seg_f.min().item()}, max={seg_f.max().item()}, dtype={seg_f.dtype}")
                    print(f"seg_m stats: min={seg_m.min().item()}, max={seg_m.max().item()}, dtype={seg_m.dtype}")
                    # check if nan in seg_m or seg_f
                    if torch.isnan(seg_m).any() or torch.isnan(seg_f).any():
                        print(f"WARNING: Found NaN values in moving segmentation for sample {i}")
                        # Replace NaN with 0 or another appropriate value
                        seg_m = torch.nan_to_num(seg_m, nan=0.0)
                        print("After NaN correction:")
                        print(f"seg_m stats: min={seg_m.min().item()}, max={seg_m.max().item()}")
                        continue
                    


                    # One-hot encode segmentations
                    # seg_f = one_hot_eval_abdominal(seg_f)
                    # seg_m = one_hot_eval_abdominal(seg_m)
                    if len(np.unique(seg_f)) == 5 and len(np.unique(seg_m)) == 5:

                        seg_f = one_hot(seg_f)
                        seg_m = one_hot(seg_m)
                    else:
                        print("Seg doesn't have 4 organs..")
                        print("number of classes: " + str(len(np.unique(seg_f))))
                        print("Shape seg fixed " + str(seg_f.size()))
                        print("Shape seg moving " + str(seg_m.size()))
                        print("Shape image fixed " + str(img_f.size()))
                        print("Shape image moving " + str(img_m.size()))

                        seg_f = one_hot(seg_f)
                        seg_m = one_hot(seg_m)
                        print("Shape seg fixed " + str(seg_f.size()))
                        print("Shape seg moving " + str(seg_m.size()))
                        # continue

                # Move to device
                img_f = img_f.float().to(args.device)
                img_m = img_m.float().to(args.device)
                if args.seg_available:
                    seg_f = seg_f.float().to(args.device)
                    seg_m = seg_m.float().to(args.device)
                #
                # # Explicitly augment moving image
                # if args.seg_available:
                #     img_m, seg_m = affine_augment(img_m, param, seg=seg_m)
                # else:
                #     img_m = affine_augment(img_m, param)

                # if args.seg_available:
                #     img_m, seg_m, aug_affine = random_affine_augment(
                #         img_m,
                #         seg=seg_m,
                #         max_random_params=param,  # param from parse_test_aug
                #         scale_params=1.0,  # Fixed scale for evaluation
                #         return_affine_matrix=True
                #     )
                # else:
                #     img_m, aug_affine = random_affine_augment(
                #         img_m,
                #         max_random_params=param,
                #         scale_params=1.0,
                #         return_affine_matrix=True
                #     )

                with torch.set_grad_enabled(False):
                    registration_results = registration_model(
                        img_f,
                        img_m,
                        seg_f=seg_f if args.seg_available else None,
                        seg_m=seg_m if args.seg_available else None,
                        transform_type=list_of_eval_aligns,
                        return_aligned_points=True,
                        save_dir=save_dir,
                        num_resolutions_for_itkelastix=args.num_resolutions_for_itkelastix,
                        aff_f=aff_f.to(img_f),
                        aff_m=aff_m.to(img_m),
                    )

                # Dictionary to save metrics dictionary for all alignment types
                all_metrics = {}
                for align_type_str, res_dict in registration_results.items():
                    if "img_m" in res_dict:
                        img_m = res_dict["img_m"]
                    if "img_f" in res_dict:
                        img_f = res_dict["img_f"]
                    if "img_a" in res_dict:
                        img_a = res_dict["img_a"]
                    elif "grid" in res_dict:
                        grid = res_dict["grid"]
                        img_a = align_img(grid, img_m)
                    else:
                        raise ValueError("No way to get aligned image")
                    if "grid" in res_dict:
                        grid = res_dict["grid"]
                    else:
                        assert (
                            "jdstd" in res_dict and "jdlessthan0" in res_dict
                        )  # If no grid, then must have jdstd and jdlessthan0
                        grid = None
                    if args.seg_available:
                        if "seg_m" in res_dict:
                            seg_m = res_dict["seg_m"]
                        if "seg_f" in res_dict:
                            seg_f = res_dict["seg_f"]
                        if "seg_a" in res_dict:
                            seg_a = res_dict["seg_a"]
                        elif "grid" in res_dict:
                            grid = res_dict["grid"]
                            seg_a = align_img(grid, seg_m).cpu()# Added CPU
                        else:
                            raise ValueError("No way to get aligned segmentation")

                    points_m = res_dict["points_m"] if "points_m" in res_dict else None
                    points_f = res_dict["points_f"] if "points_f" in res_dict else None
                    points_a = res_dict["points_a"] if "points_a" in res_dict else None
                    print(res_dict["points_weights"])
                    points_weights = (
                        res_dict["points_weights"]
                        if "points_weights" in res_dict
                        else None
                    )

                    if args.visualize:
                        if args.dim == 2:
                            imshow_registration_2d(
                                img_m[0, 0].cpu().detach().numpy(),
                                img_f[0, 0].cpu().detach().numpy(),
                                img_a[0, 0].cpu().detach().numpy(),
                                (
                                    points_m[0].cpu().detach().numpy()
                                    if points_m is not None
                                    else None
                                ),
                                (
                                    points_f[0].cpu().detach().numpy()
                                    if points_f is not None
                                    else None
                                ),
                                (
                                    points_a[0].cpu().detach().numpy()
                                    if points_a is not None
                                    else None
                                ),
                                weights=points_weights,
                            )
                            if args.seg_available:
                                imshow_registration_2d(
                                    seg_m[0, 0].cpu().detach().numpy(),
                                    seg_f[0, 0].cpu().detach().numpy(),
                                    seg_a[0, 0].cpu().detach().numpy(),
                                    (
                                        points_m[0].cpu().detach().numpy()
                                        if points_m is not None
                                        else None
                                    ),
                                    (
                                        points_f[0].cpu().detach().numpy()
                                        if points_f is not None
                                        else None
                                    ),
                                    (
                                        points_a[0].cpu().detach().numpy()
                                        if points_a is not None
                                        else None
                                    ),
                                    weights=points_weights,
                                )
                        else:
                            imshow_registration_3d(
                                img_m[0, 0].cpu().detach().numpy(),
                                img_f[0, 0].cpu().detach().numpy(),
                                img_a[0, 0].cpu().detach().numpy(),
                                (
                                    points_m[0].cpu().detach().numpy()
                                    if points_m is not None
                                    else None
                                ),
                                (
                                    points_f[0].cpu().detach().numpy()
                                    if points_f is not None
                                    else None
                                ),
                                (
                                    points_a[0].cpu().detach().numpy()
                                    if points_a is not None
                                    else None
                                ),
                                weights=(
                                    points_weights[0].cpu().detach().numpy()
                                    if points_weights is not None
                                    else None
                                ),
                                # projection=True, # OMER CHANGED TO VISUALIZED Custom Thickness
                                projection=False,
                                slab_thickness=3,  

                                save_path=save_dir / f"reg_{aug}_{align_type_str}.png",
                                resize=(256, 256, 256),
                            )
                            # imshow_registration_3d(
                            #     img_m[0, 0].cpu().detach().numpy(),
                            #     img_f[0, 0].cpu().detach().numpy(),
                            #     img_a[0, 0].cpu().detach().numpy(),
                            #     (
                            #         points_m[0].cpu().detach().numpy()
                            #         if points_m is not None
                            #         else None
                            #     ),
                            #     (
                            #         points_f[0].cpu().detach().numpy()
                            #         if points_f is not None
                            #         else None
                            #     ),
                            #     (
                            #         points_a[0].cpu().detach().numpy()
                            #         if points_a is not None
                            #         else None
                            #     ),
                            #     weights=(
                            #         points_weights[0].cpu().detach().numpy()
                            #         if points_weights is not None
                            #         else None
                            #     ),
                            #     resize=(256, 256, 256),
                            #     projection=True,
                            #     save_path=None,
                            # )
                            if args.seg_available:
                                imshow_registration_3d(
                                    seg_m.argmax(1)[0].cpu().detach().numpy(),
                                    seg_f.argmax(1)[0].cpu().detach().numpy(),
                                    seg_a.argmax(1)[0].cpu().detach().numpy(),
                                    (
                                        points_m[0].cpu().detach().numpy()
                                        if points_m is not None
                                        else None
                                    ),
                                    (
                                        points_f[0].cpu().detach().numpy()
                                        if points_f is not None
                                        else None
                                    ),
                                    (
                                        points_a[0].cpu().detach().numpy()
                                        if points_a is not None
                                        else None
                                    ),
                                    weights=(
                                        points_weights[0].cpu().detach().numpy()
                                        if points_weights is not None
                                        else None
                                    ),
                                    save_path=save_dir / f"seg_{aug}_{align_type_str}.png",
                                )


                    # # save the affine and moving transformations # TODO OMER Wrote it
                    # aff_path = save_dir / f"affine_f_{i}-{mod1_str}.npy"
                    # print("Saving:", aff_path)
                    # np.save(aff_path, aff_f.cpu().detach().numpy())

                    # aff_path = save_dir / f"affine_m_{i}-{mod2_str}.npy"
                    # print("Saving:", aff_path)
                    # np.save(aff_path, aff_m.cpu().detach().numpy())

                    # save moving image path as a txt file
                    img_m_original_path = moving["img"]["path"]
                    # save the path of the moving image
                    img_m_path = save_dir / f"img_m_path_{i}-{mod2_str}.txt"
                    print("Saving:", img_m_path)
                    with open(img_m_path, 'w') as f:
                        f.write(img_m_original_path[0])

                    # save the path of the moving seg
                    if args.seg_available:
                        seg_m_original_path = moving["seg"]["path"]
                        seg_m_path = save_dir / f"seg_m_path_{i}-{mod2_str}.txt"
                        print("Saving:", seg_m_path)
                        with open(seg_m_path, 'w') as f:
                            f.write(seg_m_original_path[0])

                    # Compute metrics
                    metrics = {}
                    if args.seg_available:
                        # Always compute hard dice once ahead of time
                        seg_a = seg_a.to(args.device)
                        seg_f = seg_f.to(args.device)
                        dice_total = loss_ops.DiceLoss(hard=True)(
                            seg_a, seg_f, ign_first_ch=True
                        )
                        dice_roi = loss_ops.DiceLoss(hard=True, return_regions=True)(
                            seg_a, seg_f, ign_first_ch=True
                        )
                        dice_total = 1 - dice_total.item()
                        dice_roi = (1 - dice_roi.cpu().detach().numpy()).tolist()
                    for m in list_of_eval_metrics:
                        if m == "mse":
                            metrics["mse"] = loss_ops.MSELoss()(img_f, img_a).item()
                        elif m == "softdice":
                            assert args.seg_available
                            metrics["softdiceloss"] = loss_ops.DiceLoss()(
                                seg_a, seg_f
                            ).item()
                            metrics["softdice"] = 1 - metrics["softdiceloss"]
                            print(metrics["softdice"])
                        elif m == "harddice":
                            assert args.seg_available
                            metrics["harddice"] = dice_total
                        elif m == "harddiceroi":
                            assert args.seg_available
                            metrics["harddiceroi"] = dice_roi
                        elif m == "ssim":
                            metrics['ssim'] = SSIM3D(window_size=5, size_average=True)(img_f, img_a).item()
                        elif m == "hausd":
                            assert args.seg_available and args.dim == 3
                            metrics["hausd"] = loss_ops.hausdorff_distance(seg_a, seg_f)
                        elif m == "jdstd":
                            assert args.dim == 3
                            if grid is None:
                                metrics["jdstd"] = res_dict["jdstd"]
                            else:
                                grid_permute = grid.permute(0, 4, 1, 2, 3)
                                metrics["jdstd"] = loss_ops.jdstd(grid_permute)
                        elif m == "jdlessthan0":
                            assert args.dim == 3
                            if grid is None:
                                metrics["jdlessthan0"] = res_dict["jdlessthan0"]
                            else:
                                grid_permute = grid.permute(0, 4, 1, 2, 3)
                                metrics["jdlessthan0"] = loss_ops.jdstd(grid_permute)

                        elif m == "RightKidney":
                                metrics['RightKidney'] = DiceOrgan(seg_a,seg_f,1)
                        elif m == "LeftKidney":
                                metrics['LeftKidney'] = DiceOrgan(seg_a,seg_f,2)
                        elif m == "Spleen":
                                metrics['Spleen'] = DiceOrgan(seg_a,seg_f,3)
                        elif m == "Liver":
                                metrics['Liver'] = DiceOrgan(seg_a,seg_f,4)
                        
                        else:
                            raise ValueError('Invalid metric "{}"'.format(m))
                    all_metrics[align_type_str] = metrics

                    # Print some stats
                    print("\nDebugging info:")
                    print(f'-> Time: {res_dict["time"]}')
                    print(f"-> Alignment: {align_type_str} ")
                    print(f"-> Max random params: {param} ")
                    print(f"-> Img shapes: {img_f.shape}, {img_m.shape}")
                    if points_f is not None:
                        print(f"-> Point shapes: {points_f.shape}, {points_m.shape}")
                        print(f"-> Point weights: {points_weights}")
                    print(f"-> Float16: {args.use_amp}")
                    if args.seg_available:
                        print(f"-> Seg shapes: {seg_f.shape}, {seg_m.shape}")
                    # print(f"-> Full Results: {res_dict}")

                    print("\nMetrics:")
                    for metric_name, metric in metrics.items():
                        print(f"-> {metric_name}: {metric}")

                    # Save all outputs to disk
                    assert args.batch_size == 1  # TODO: fix this

                    # Save metrics
                    metrics_path = save_dir / f"metrics-{aug}-{align_type_str}.json"
                    print("Saving:", metrics_path)
                    save_dict_as_json(metrics, metrics_path)

                    # Save images and grid
                    img_f_path = save_dir / f"img_f_{i}-{mod1_str}.npy"
                    
                    img_m_path = save_dir / f"img_m_{i}-{mod2_str}-{aug}.npy"
                    img_a_path = (
                        save_dir
                        / f"img_a_{i}-{mod1_str}-{mod2_str}-{aug}-{align_type_str}.npy"
                    )
                    grid_path = (
                        save_dir
                        / f"grid_{i}-{mod1_str}-{mod2_str}-{aug}-{align_type_str}.npy"
                    )

                    img_f_path_gz = save_dir / f"img_f_{i}-{mod1_str}.nii.gz"
                    img_m_path_gz = save_dir / f"img_m_{i}-{mod2_str}-{aug}.nii.gz"
                    img_a_path_gz = save_dir / f"img_a_{i}-{mod1_str}-{mod2_str}-{aug}-{align_type_str}.nii.gz"
                    if not os.path.exists(img_f_path):
                        print("Saving:", img_f_path)
                        np.save(img_f_path, img_f[0].cpu().detach().numpy())
                        nifti_img = nib.Nifti1Image(img_f[0][0].cpu().detach().numpy(), affine=aff_f[0]) #TODO: Omer Changed this
                        nib.save(nifti_img, img_f_path_gz)

                    if not os.path.exists(img_m_path):
                        print("Saving:", img_m_path)
                        np.save(img_m_path, img_m[0].cpu().detach().numpy())
                        nifti_img = nib.Nifti1Image(img_m[0][0].cpu().detach().numpy(), affine=aff_m[0]) #TODO: Omer Changed this
                        nib.save(nifti_img, img_m_path_gz)

                    print("Saving:", img_a_path)
                    np.save(img_a_path, img_a[0].cpu().detach().numpy())
                    nifti_img = nib.Nifti1Image(img_a[0][0].cpu().detach().numpy(), affine=aff_f[0]) #TODO: Omer Changed this
                    nib.save(nifti_img, img_a_path_gz)
                    if grid is not None:
                        print("Saving:", grid_path)
                        np.save(grid_path, grid[0].cpu().detach().numpy())
                    else:
                        print("Grid is None, not saving!")

                    # Save segmentations
                    if args.seg_available:
                        seg_f_path = save_dir / f"seg_f_{i}-{mod1_str}.npy"
                        seg_m_path = save_dir / f"seg_m_{i}-{mod2_str}-{aug}.npy"
                        seg_a_path = (
                            save_dir
                            / f"seg_a_{i}-{mod1_str}-{mod2_str}-{aug}-{align_type_str}.npy"
                        )

                        seg_f_path_gz = save_dir / f"seg_f_{i}-{mod1_str}.nii.gz"
                        seg_m_path_gz = save_dir / f"seg_m_{i}-{mod2_str}.nii.gz"
                        seg_a_path_gz = save_dir / f"seg_a_{i}-{mod1_str}-{mod2_str}-{aug}-{align_type_str}.nii.gz"

                        if not os.path.exists(seg_f_path):
                            print("Saving:", seg_f_path)
                            seg_f_np = np.argmax(seg_f.cpu().detach().numpy(), axis=1).astype(np.int32)  # Convert to int32
                            np.save(seg_f_path, seg_f_np)
                            nifti_seg_f = nib.Nifti1Image(seg_f_np[0], affine=aff_f[0].numpy())  # Ensure affine is also a NumPy array
                            nib.save(nifti_seg_f, seg_f_path_gz)

                        if not os.path.exists(seg_m_path):
                            print("Saving:", seg_m_path)
                            seg_m_np = np.argmax(seg_m.cpu().detach().numpy(), axis=1).astype(np.int32)  # Convert to int32
                            np.save(seg_m_path, seg_m_np)
                            nifti_seg_m = nib.Nifti1Image(seg_m_np[0], affine=aff_m[0].numpy())  # Ensure affine is also a NumPy array
                            nib.save(nifti_seg_m, seg_m_path_gz)

                        print("Saving:", seg_a_path)
                        seg_a_np = np.argmax(seg_a.cpu().detach().numpy(), axis=1).astype(np.int32)  # Convert to int32
                        np.save(seg_a_path, seg_a_np)
                        nifti_seg_a = nib.Nifti1Image(seg_a_np[0], affine=aff_f[0].numpy())  # Ensure affine is also a NumPy array
                        nib.save(nifti_seg_a, seg_a_path_gz)

                    # Save points
                    if points_f is not None:
                        points_f_path = save_dir / f"points_f_{i}-{mod1_str}.npy"
                        points_m_path = save_dir / f"points_m_{i}-{mod2_str}-{aug}.npy"
                        points_a_path = (
                            save_dir
                            / f"points_a_{i}-{mod1_str}-{mod2_str}-{aug}-{align_type_str}.npy"
                        )
                        if not os.path.exists(points_f_path):
                            print("Saving:", points_f_path)
                            np.save(
                                points_f_path,
                                points_f[0].cpu().detach().numpy(),
                            )
                        if not os.path.exists(points_m_path):
                            print("Saving:", points_m_path)
                            np.save(
                                points_m_path,
                                points_m[0].cpu().detach().numpy(),
                            )
                        print("Saving:", points_a_path)
                        np.save(
                            points_a_path,
                            points_a[0].cpu().detach().numpy(),
                        )
                        if points_weights is not None:
                            points_weights_path = (
                                save_dir
                                / f"points_weights_{i}-{mod1_str}-{mod2_str}-{aug}-{align_type_str}.npy"
                            )
                            print("Saving:", points_weights_path)
                            np.save(
                                points_weights_path,
                                points_weights[0].cpu().detach().numpy(),
                            )

            # Save metrics in global test_metrics dictionary
            for m in list_of_eval_metrics:
                for align_type_str in list_of_eval_aligns:
                    metrics = all_metrics[align_type_str]
                    test_metrics[f"{m}:{mod1}:{mod2}:{aug}:{align_type_str}"].append(
                        metrics[m]
                    )

    return test_metrics





##############################################################3

