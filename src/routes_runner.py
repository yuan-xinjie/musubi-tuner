"""
训练 & 缓存执行路由：命令组装与进程启动
"""
import sys
from fastapi import APIRouter, HTTPException

from config_utils import CONFIG_PATH, read_toml, is_wan_task
from process_manager import ProcessManager

router = APIRouter()

# 由主模块注入的 manager 实例引用
_manager: ProcessManager = None

def init_manager(manager: ProcessManager):
    global _manager
    _manager = manager

# ─── 参数映射表（训练命令共用） ──────────────────────────────

PARAM_MAP = [
    # 核心模型/任务参数 (来自 task_item)
    ("dit", "--dit", False),
    ("vae", "--vae", False),
    ("text_encoder", "--text_encoder", False),
    ("dataset_config", "--dataset_config", False),
    ("seed", "--seed", False),
    ("learning_rate", "--learning_rate", False),
    ("timestep_sampling", "--timestep_sampling", False),
    ("network_dim", "--network_dim", False),
    ("network_alpha", "--network_alpha", False),
    ("blocks_to_swap", "--blocks_to_swap", False),
    ("output_name", "--output_name", False),
    ("output_dir", "--output_dir", False),
    ("model_version", "--model_version", False),
    ("max_train_epochs", "--max_train_epochs", False),
    ("save_every_n_epochs", "--save_every_n_epochs", False),
    ("sample_every_n_epochs", "--sample_every_n_epochs", False),
    ("sample_prompts", "--sample_prompts", False),
    ("t5", "--t5", False),
    ("task", "--task", False),
    ("clip", "--clip", False),
    ("min_timestep", "--min_timestep", False),
    ("max_timestep", "--max_timestep", False),
    ("max_data_loader_n_workers", "--max_data_loader_n_workers", False),
    ("discrete_flow_shift", "--discrete_flow_shift", False),

    # 全局/训练优化参数 (来自 qwen_config 或 wan_config)
    ("logging_dir", "--logging_dir", False),
    ("force_v2_1_time_embedding", "--force_v2_1_time_embedding", True),
    ("vae_cache_cpu", "--vae_cache_cpu", True),
    ("preserve_distribution_shape", "--preserve_distribution_shape", True),
    ("flash_attn", "--flash_attn", True),
    ("sdpa", "--sdpa", True),
    ("split_attn", "--split_attn", True),
    ("guidance_scale", "--guidance_scale", False),
    ("gradient_checkpointing", "--gradient_checkpointing", True),
    ("gradient_checkpointing_cpu_offload", "--gradient_checkpointing_cpu_offload", True),
    ("gradient_accumulation_steps", "--gradient_accumulation_steps", False),
    ("network_module", "--network_module", False),
    ("lr_scheduler", "--lr_scheduler", False),
    ("lr_scheduler_num_cycles", "--lr_scheduler_num_cycles", False),
    ("lr_decay_steps", "--lr_decay_steps", False),
    ("lr_scheduler_min_lr_ratio", "--lr_scheduler_min_lr_ratio", False),
    ("cuda_allow_tf32", "--cuda_allow_tf32", True),
    ("cuda_cudnn_benchmark", "--cuda_cudnn_benchmark", True),
    ("mixed_precision", "--mixed_precision", False),
    ("fp8_base", "--fp8_base", True),
    ("fp8_scaled", "--fp8_scaled", True),
    ("persistent_data_loader_workers", "--persistent_data_loader_workers", True),
    ("use_pinned_memory_for_block_swap", "--use_pinned_memory_for_block_swap", True),
    ("compile", "--compile", True),
    ("compile_backend", "--compile_backend", False),
    ("compile_mode", "--compile_mode", False),
    ("compile_dynamic", "--compile_dynamic", False),
    ("compile_cache_size_limit", "--compile_cache_size_limit", False),
    ("img_in_txt_in_offloading", "--img_in_txt_in_offloading", True),
    ("optimizer_type", "--optimizer_type", False),
    ("optimizer_args", "--optimizer_args", False),
    ("wandb_api_key", "--wandb_api_key", False),
    ("log_with", "--log_with", False),
]

# ─── 辅助：非空判断 ──────────────────────────────────────────

def _has_value(val) -> bool:
    return val is not None and str(val).strip() != ""

# ─── 缓存路由 ────────────────────────────────────────────────

def _build_qwen_cache_cmds(config, task_item):
    """组装 Qwen 图像缓存命令"""
    python_exe = sys.executable
    cache_config = config.get("qwen_cache", {})
    dataset_config = task_item.get("dataset_config", "")
    model_version = task_item.get("model_version", "edit-2511")
    text_encoder = task_item.get("text_encoder", "")
    vae_path = task_item.get("vae", "")

    # 1. Latents
    latent_cmd = [
        python_exe, "-m", "accelerate.commands.launch",
        "./qwen_image_cache_latents.py",
        f"--dataset_config={dataset_config}",
        f"--model_version={model_version}"
    ]
    if vae_path:
        latent_cmd.append(f"--vae={vae_path}")
    if cache_config.get("vae_tiling"):
        latent_cmd.append("--vae_tiling")
    if cache_config.get("vae_chunk_size"):
        latent_cmd.append(f"--vae_chunk_size={cache_config['vae_chunk_size']}")
    if cache_config.get("vae_spatial_tile_sample_min_size"):
        latent_cmd.append(f"--vae_spatial_tile_sample_min_size={cache_config['vae_spatial_tile_sample_min_size']}")

    # 2. Text Encoder
    te_cmd = [
        python_exe, "-m", "accelerate.commands.launch",
        "./qwen_image_cache_text_encoder_outputs.py",
        f"--dataset_config={dataset_config}",
        f"--text_encoder={text_encoder}",
        f"--model_version={model_version}",
        f"--batch_size={cache_config.get('batch_size', 16)}"
    ]
    if cache_config.get("fp8_vl"):
        te_cmd.append("--fp8_vl")

    return [latent_cmd, te_cmd]


def _build_wan_cache_cmds(config, task_item):
    """组装 Wan2.2 视频缓存命令"""
    python_exe = sys.executable
    cache_config = config.get("wan_cache", {})
    dataset_config = task_item.get("dataset_config", "")
    vae_path = task_item.get("vae", "")
    is_i2v = task_item.get("i2v", False)

    # 1. Latents
    latent_cmd = [
        python_exe, "-m", "accelerate.commands.launch",
        "./wan_cache_latents.py",
        f"--dataset_config={dataset_config}"
    ]
    if vae_path:
        latent_cmd.append(f"--vae={vae_path}")
    clip_path = task_item.get("clip", "")
    if clip_path:
        latent_cmd.append(f"--clip={clip_path}")
    if is_i2v:
        latent_cmd.append("--i2v")

    # 2. Text Encoder
    te_cmd = [
        python_exe, "-m", "accelerate.commands.launch",
        "./wan_cache_text_encoder_outputs.py",
        f"--dataset_config={dataset_config}",
        f"--batch_size={cache_config.get('batch_size', 16)}"
    ]
    t5_path = task_item.get("t5", "")
    if t5_path:
        te_cmd.append(f"--t5={t5_path}")

    return [latent_cmd, te_cmd]


@router.post("/api/cache/{name}")
async def run_cache(name: str):
    config = read_toml(CONFIG_PATH)
    tasks = config.get("task", [])
    task_item = next((t for t in tasks if t.get("output_name") == name), None)

    if not task_item:
        raise HTTPException(status_code=404, detail="Task not found")

    is_wan = is_wan_task(task_item)
    cmds = _build_wan_cache_cmds(config, task_item) if is_wan else _build_qwen_cache_cmds(config, task_item)

    print(f"[DEBUG] Cache Commands: {cmds}")
    try:
        _manager.start(cmds)
        return {"status": "success", "message": "Caching started"}
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=str(e))

# ─── 训练路由 ────────────────────────────────────────────────

def _build_train_cmd(config, task_item):
    """组装训练命令行参数列表"""
    python_exe = sys.executable
    is_wan = is_wan_task(task_item)

    global_config = config.get("wan_config", {}) if is_wan else config.get("qwen_config", {})
    train_script = "./wan_train_network.py" if is_wan else "./qwen_image_train_network.py"

    # 动态获取 mixed_precision
    mixed_prec = task_item.get("mixed_precision")
    if not mixed_prec:
        mixed_prec = global_config.get("mixed_precision", "fp16" if is_wan else "bf16")

    # 基础命令架构
    if is_wan:
        assembled = [
            python_exe, "-m", "accelerate.commands.launch",
            "--num_cpu_threads_per_process=8",
            f"--mixed_precision={mixed_prec}",
            train_script
        ]
    else:
        assembled = [
            python_exe, "-m", "accelerate.commands.launch",
            "--num_cpu_threads_per_process=8",
            f"--mixed_precision={mixed_prec}"
        ]
        if mixed_prec == "bf16":
            assembled.append("--downcast_bf16")
        assembled.append(train_script)

    # 按映射表填充参数
    for key, arg, is_flag in PARAM_MAP:
        val = task_item.get(key)
        if val is None or val == "":
            val = global_config.get(key)
        if val is None or val == "":
            continue
        if is_flag:
            if val is True:
                assembled.append(arg)
        else:
            assembled.append(f"{arg}={val}")

    # --network_args loraplus_lr_ratio=X
    lora_ratio = task_item.get("loraplus_lr_ratio")
    if _has_value(lora_ratio):
        assembled.append("--network_args")
        assembled.append(f"loraplus_lr_ratio={lora_ratio}")

    # --optimizer_args 汇总
    opt_args = []
    for opt_key, opt_name in [
        ("rank", "rank"), ("weight_decay", "weight_decay"),
        ("update_proj_gap", "update_proj_gap"), ("scale", "scale"),
    ]:
        v = task_item.get(opt_key)
        if _has_value(v):
            opt_args.append(f"{opt_name}={v}")
    proj_type = task_item.get("projection_type")
    if _has_value(proj_type):
        opt_args.append(f"projection_type='{proj_type}'")
    if opt_args:
        assembled.append("--optimizer_args")
        assembled.extend(opt_args)

    # log_tracker_name
    log_tracker = task_item.get("log_tracker_name")
    if _has_value(log_tracker):
        assembled.append(f"--log_tracker_name={log_tracker}")

    return assembled


@router.post("/api/train/{name}")
async def run_train(name: str):
    config = read_toml(CONFIG_PATH)
    tasks = config.get("task", [])
    task_item = next((t for t in tasks if t.get("output_name") == name), None)

    if not task_item:
        raise HTTPException(status_code=404, detail="Task not found")

    assembled_args = _build_train_cmd(config, task_item)
    print(f"[DEBUG] Train Command: {assembled_args}")

    try:
        _manager.start([assembled_args])
        return {"status": "success", "message": "Training started"}
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=str(e))
