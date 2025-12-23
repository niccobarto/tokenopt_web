import os
from pathlib import Path
from typing import List, cast
from PIL import Image,ImageDraw
from torch import cuda
from tokenopt_generator.inpaiting_utils.inpainting import add_conf, ObjectiveType, image_to_tensor, \
    tensor_to_image, mask_to_tensor, ComposedCLIP, ReconstructionObjective,TokenResetter
from tokenopt_generator.token_opt.tto.test_time_opt import TestTimeOpt, CLIPObjective, MultiObjective
import tempfile
import shutil


device = "cuda" if cuda.is_available() else "cpu"
# region Configurazioni TTO
configs_implemented = [
    add_conf(
        #name="C1_RESET",
        name="config1",
        tto_params=dict(
            num_iter=351,
            ema_decay=0.97,
            lr=1e-1,
            enable_amp=True,
            reg_weight=2.5e-2,
            token_noise=2e-4,
            reg_type="seed",

        ),
        cfg_scale=1.5,
        num_aug=8,
        weights=[1.0, 1.0],
        enable_token_reset=True,
        reset_period=15,
        objective_types=[ObjectiveType.ReconstructionObjective,
                         ObjectiveType.ComposedCLIP]
    ),
    add_conf(
        #name="C3_RECON_STRONG_RESET",
        name="config2",
        tto_params=dict(
            num_iter=351,
            ema_decay=0.95,
            lr=1e-1,
            enable_amp=True,
            reg_weight=2.0e-2,
            token_noise=2e-4,
            reg_type="seed",

        ),
        cfg_scale=1.2,
        num_aug=8,
        weights=[1.5, 0.8],
        enable_token_reset=True,
        reset_period=15,
        objective_types=[ObjectiveType.ReconstructionObjective,
                         ObjectiveType.ComposedCLIP]
    ),
    add_conf(
        #name="C4_CLIPONLY_RESET",
        name="config3",
        tto_params=dict(
            num_iter=351,
            ema_decay=0.95,
            lr=1e-1,
            enable_amp=True,
            reg_weight=2.0e-2,
            token_noise=2e-4,
            reg_type="seed",

        ),
        cfg_scale=1.5,
        num_aug=8,
        weights=[1],
        enable_token_reset=True,
        reset_period=15,
        objective_types=[ObjectiveType.ComposedCLIP]
    ),
    add_conf(
        #name="BALANCED_RESET",
        name="config4",
        tto_params=dict(
            num_iter=351,
            ema_decay=0.98,
            lr=1e-1,
            enable_amp=True,
            reg_weight=0.02,
            token_noise=0.005,
            reg_type="seed",

        ),
        cfg_scale=1.5,
        num_aug=8,
        weights=[1],
        enable_token_reset=True,
        reset_period=15,
        objective_types=[ObjectiveType.ComposedCLIP]
    )
]


# endregion

def generate_inpainting(
        input_image_path: Path,
        mask_path: Path,
        prompt: str,
        num_generations: int,
        configs:dict[str,bool],
        output_dir: Path,
):
    out_path_images = []  # lista dei path delle immagini generate
    input_tns = image_to_tensor(image_path=input_image_path, device=device)  # carico immagine come tensore
    mask_tns = mask_to_tensor(mask_path=mask_path, device=device)  # carico maschera come tensore
    input_masked = input_tns * mask_tns  # immagine mascherata
    for name, config, objective_types in configs_implemented:
        if not configs.get(name, False):
            continue
        objectives = []
        for obj_type, weight in zip(objective_types, config.objective_weights):
            if obj_type == ObjectiveType.ReconstructionObjective:
                recon_obj = ReconstructionObjective(input_masked, mask_tns)
                objectives.append(recon_obj)
            elif obj_type == ObjectiveType.ComposedCLIP:  # se e' un objective di ComposedCLIP e siamo in inpainting
                base_clip_obj = CLIPObjective(
                    prompt=prompt,
                    cfg_scale=config.cfg_scale,
                    num_augmentations=config.num_augmentations
                ) # creo la CLIPObjective di base
                composed_clip_obj = ComposedCLIP(
                    base_clip_obj=base_clip_obj,
                    orig_img=input_tns,
                    mask_bin=mask_tns,
                    outside_grad=0.0
                )  # creo la ComposedCLIP
                objectives.append(composed_clip_obj)
            else:
                raise ValueError(f"Objective type {obj_type} not recognized")
        multi_objective = MultiObjective(objectives, config.objective_weights)
        print("Start creation of TTO with config:", name)
        tto = TestTimeOpt(config.tto_config, multi_objective)
        print("Starting generation with config:", name)
        token_reset = TokenResetter(
            titok=tto.titok,
            masked_img=input_masked,
            mask=mask_tns,
            reset_period=config.reset_period
        )
        tto.to(device)
        result_tns = tto(seed=input_masked,token_reset_callback=token_reset)
        print("Generation completed.")
        output_tns=input_tns * mask_tns + result_tns * (1-mask_tns)
        result_img = tensor_to_image(result_tns)
        output_img= tensor_to_image(output_tns)
        result_path=output_dir / f"{name}_raw_result.png"
        out_path = output_dir / f"{name}_result.png"
        result_img.save(result_path)
        output_img.save(out_path)
        out_path_images.append(out_path)
    return out_path_images



def generate_inpainting_bytes(
        input_image_bytes: bytes,
        input_mask_bytes: bytes,
        prompt:str,
        num_generations:int,
        configs:dict[str,bool],
)->List[dict]:
    """WRAPPER"""

    workdir=Path(tempfile.mkdtemp(prefix="tto_job"))
    results: List[dict] = []
    try:
        inputs_dir=workdir / "inputs"
        outputs_dir=workdir / "outputs"
        inputs_dir.mkdir(parents=True, exist_ok=True)
        outputs_dir.mkdir(parents=True, exist_ok=True)

        input_path=inputs_dir / "original.png"
        mask_path=inputs_dir / "mask.png"

        input_path.write_bytes(input_image_bytes)
        mask_path.write_bytes(input_mask_bytes)

        generated_paths=generate_inpainting(
            input_image_path=input_path,
            mask_path=mask_path,
            prompt=prompt,
            num_generations=num_generations,
            configs=configs,
            output_dir=outputs_dir,
        )
        for p in generated_paths:
            result_bytes=p.read_bytes()
            results.append({
                "filename":Path(p).name,
                "content_type":"image/png",
                "data":result_bytes,
            })
        return results
    except Exception as e:
        import traceback
        print("TTO ERROR:", repr(e))
        print(traceback.format_exc())
        raise
    finally:
        shutil.rmtree(workdir,ignore_errors=True)

