import torch
import numpy as np
import torchio as tio
import time

from keymorph.utils import align_img, one_hot, one_hot_subsampled_pair

from scripts.script_utils import aggregate_dicts, aggregate_dicts, compute_metrics_and_loss

def print_dimension_info(fixed, moving):
    """Print detailed dimension information for debugging"""
    print("\n=== DIMENSION DEBUGGING INFO ===")
    
    # Print image paths
    print(f"Fixed image path: {fixed['img']['path']}")
    print(f"Moving image path: {moving['img']['path']}")
    
    # Print segmentation paths if available
    if 'seg' in fixed:
        print(f"Fixed segmentation path: {fixed['seg']['path']}")
    if 'seg' in moving:
        print(f"Moving segmentation path: {moving['seg']['path']}")
    
    # Print dimensions
    img_f, img_m = fixed["img"][tio.DATA], moving["img"][tio.DATA]
    print(f"Fixed image dimensions: {img_f.shape}")
    print(f"Moving image dimensions: {img_m.shape}")
    
    if 'seg' in fixed and 'seg' in moving:
        seg_f, seg_m = fixed["seg"][tio.DATA], moving["seg"][tio.DATA]
        print(f"Fixed segmentation dimensions: {seg_f.shape}")
        print(f"Moving segmentation dimensions: {seg_m.shape}")


@torch.no_grad()
def run_val(val_loader, registration_model, args):
    """Validate for one epoch.

    Args:
        val_loader: Dataloader which returns pair of TorchIO subjects per iteration
        registration_model: Registration model
        args: Other script arguments
    """
    start_time = time.time()
    
    registration_model.eval()

    res = []

    transform_type = args.transform_type
    loss_fn = args.loss_fn

    for step_idx, subjects in enumerate(val_loader):
        fixed, moving = subjects
        print_dimension_info(fixed, moving)
        
        # Get images and segmentations from TorchIO subject
        img_f, img_m = fixed["img"][tio.DATA], moving["img"][tio.DATA]
        aff_f, aff_m = fixed["img"]["affine"], moving["img"]["affine"]
        
        print("img_f prod:" + str(np.prod(img_f.shape)))
        if np.prod(img_f.shape) >= 24000000:
            print("LARGE RESOLUTION IMAGE")
            print("img_f path:"+ str(fixed['img']['path']))
            continue

        print("img_m prod:" + str(np.prod(img_m.shape)))
        if np.prod(img_m.shape) >= 24000000:
            print("LARGE RESOLUTION IMAGE")
            print("img_m path:"+ str(moving['img']['path']))
            continue
            
        print("step: " + str(step_idx))

        if args.seg_available:
            seg_f, seg_m = fixed["seg"][tio.DATA], moving["seg"][tio.DATA]
            
            if np.prod(seg_f.shape) >= 23000000:
                print("LARGE RESOLUTION IMAGE")
                print("seg_f path:"+ str(fixed['seg']['path']))
                continue
                
            if np.prod(seg_m.shape) >= 23000000:
                print("LARGE RESOLUTION IMAGE")
                print("seg_m path:"+ str(moving['seg']['path']))
                continue
                
            print("seg_f shape before one-hot: " + str(seg_f.shape))
            print("seg_m shape before one-hot: " + str(seg_m.shape))

            if args.max_train_seg_channels is not None:
                seg_f, seg_m = one_hot_subsampled_pair(
                    seg_f.long(), seg_m.long(), args.max_train_seg_channels
                )
            else:
                if len(np.unique(seg_f)) == 13 or len(np.unique(seg_m)) == 13:
                    print("something is wrong...")
                    print("number of classes: " + str(len(np.unique(seg_f))))
                    print("number of classes: " + str(len(np.unique(seg_m))))
                    seg_f = one_hot(seg_f)
                    seg_m = one_hot(seg_m)
                    continue

        assert (
            img_f.shape[1] == 1
        ), f"Fixed image must have 1 channel:\n --> {fixed['img']['path']}: {img_f.shape}"
        assert (
            img_m.shape[1] == 1
        ), f"Moving image must have 1 channel:\n--> {moving['img']['path']}: {img_m.shape}"

        # Move to device
        img_f = img_f.float().to(args.device)
        img_m = img_m.float().to(args.device)
        aff_f = aff_f.float().to(args.device)
        aff_m = aff_m.float().to(args.device)
        if args.seg_available:
            seg_f = seg_f.float().to(args.device)
            seg_m = seg_m.float().to(args.device)

        # Forward pass (No profiler needed for validation loop)
        registration_results = registration_model(
            img_f,
            img_m,
            transform_type=transform_type,
            return_aligned_points=args.visualize or "dispersion" in loss_fn,
            aff_f=aff_f,
            aff_m=aff_m,
        )[transform_type]
        
        grid = registration_results["grid"]
        align_type = transform_type
        tps_lmbda = registration_results.get("tps_lmbda", None)
        points_m = registration_results.get("points_m", None)
        points_f = registration_results.get("points_f", None)
        points_a = registration_results.get("points_a", None)
        points_weights = registration_results.get("points_weights", None)

        img_a = align_img(grid, img_m)
        if args.seg_available:
            seg_a = align_img(
                grid, seg_m
            )  # Note we use bilinear interpolation here so that backprop works

        # Compute unified metrics and loss
        metrics, loss = compute_metrics_and_loss(
            args, img_f, img_a, 
            seg_f=seg_f if args.seg_available else None, 
            seg_a=seg_a if args.seg_available else None, 
            points_f=points_f, 
            points_m=points_m
        )

        print("[VAL] LOSS :" + str(loss.item()))

        end_time = time.time()
        metrics["epoch_time"] = end_time - start_time

        # Convert metrics to numpy
        metrics = {
            k: torch.as_tensor(v).detach().cpu().numpy().item()
            for k, v in metrics.items()
        }
        res.append(metrics)

        if args.debug_mode:
            print("\nDebugging info [VAL]:")
            print(f"-> Alignment: {align_type} ")
            print(f"-> TPS lambda: {tps_lmbda} ")
            print(f"-> Loss: {loss_fn}")
            print(f"-> Img shapes: {img_f.shape}, {img_m.shape}")
            if points_f is not None:
                print(f"-> Point shapes: {points_f.shape}, {points_m.shape}")
            print(f"-> Float16: {args.use_amp}")
            if args.seg_available:
                print(f"-> Seg shapes: {seg_f.shape}, {seg_m.shape}")

    return aggregate_dicts(res)
