import torch
import weakref
COMPILED_FORWARD_CACHE = weakref.WeakKeyDictionary()


class TorchCompileSpeedSettings:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "backend": (["inductor","cudagraphs"], {"default": "inductor"}),
                "fullgraph": ("BOOLEAN", {"default": False, "tooltip": "Enable full graph mode"}),
                "mode": (["default", "max-autotune", "max-autotune-no-cudagraphs", "reduce-overhead", "speed"], {"default": "speed"}),
                "dynamic": ("BOOLEAN", {"default": False, "tooltip": "Enable dynamic mode"}),
                "dynamo_cache_size_limit": ("INT", {"default": 64, "min": 0, "max": 1024, "step": 1, "tooltip": "torch._dynamo.config.cache_size_limit"}),
                "compile_transformer_blocks_only": ("BOOLEAN", {"default": True, "tooltip": "Compile only transformer blocks"}),
                "reuse_if_similar": ("BOOLEAN", {"default": True, "tooltip": "Reuse compiled artifacts when similar"}),
                "experimental_ptx": ("BOOLEAN", {"default": False, "tooltip": "Enable experimental PTX acceleration"}),
                "ptx_fast_math": ("BOOLEAN", {"default": True, "tooltip": "Enable fast math if available"}),
                "warmup_runs": ("INT", {"default": 1, "min": 0, "max": 5, "step": 1}),
            },
            "optional": {
                "ptx_cache_dir": ("STRING", {"default": ""}),
                "dynamo_recompile_limit": ("INT", {"default": 128, "min": 0, "max": 1024, "step": 1, "tooltip": "torch._dynamo.config.recompile_limit"}),
                "force_parameter_static_shapes": ("BOOLEAN", {"default": False, "tooltip": "torch._dynamo.config.force_parameter_static_shapes"}),
            },
        }
    RETURN_TYPES = ("WANCOMPILEARGS",)
    RETURN_NAMES = ("torch_compile_args",)
    FUNCTION = "set_args"
    CATEGORY = "optimization"
    DESCRIPTION = """torch.compile settings for maximum speed optimization.

Speed Mode Features:
- Uses inductor backend with max-autotune-no-cudagraphs
- Enables dynamic compilation for better cache reuse
- Disables CUDA graphs for flexibility
- Enables all Triton autotune optimizations
- First run: comprehensive autotune (slower)
- Second run: cached execution (extremely fast)

Author: eddy
"""

    def set_args(self, backend, fullgraph, mode, dynamic, dynamo_cache_size_limit, compile_transformer_blocks_only, reuse_if_similar, experimental_ptx, ptx_fast_math, warmup_runs, ptx_cache_dir="", dynamo_recompile_limit=128, force_parameter_static_shapes=True):

        if mode == "speed":
            backend = "inductor"
            fullgraph = False
            dynamic = True
            effective_mode = "max-autotune-no-cudagraphs"
            speed_preset = True
        else:
            effective_mode = mode
            speed_preset = False

        compile_args = {
            "backend": backend,
            "fullgraph": fullgraph,
            "mode": effective_mode,
            "dynamic": dynamic,
            "dynamo_cache_size_limit": dynamo_cache_size_limit,
            "dynamo_recompile_limit": dynamo_recompile_limit,
            "compile_transformer_blocks_only": compile_transformer_blocks_only,
            "force_parameter_static_shapes": force_parameter_static_shapes,
            "reuse_if_similar": reuse_if_similar,
            "experimental_ptx": experimental_ptx,
            "ptx_fast_math": ptx_fast_math,
            "warmup_runs": warmup_runs,
            "ptx_cache_dir": ptx_cache_dir,
            "speed_preset": speed_preset,
        }

        return (compile_args, )
        
class AnyType(str):
    def __eq__(self, _) -> bool: return True
    def __ne__(self, _) -> bool: return False

any_type = AnyType("*")

class ApplyTorchCompile:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": (any_type,{"forceInput": True}), 
                "compile_args": ("WANCOMPILEARGS",),
            },
        }
        
    @classmethod
    def VALIDATE_INPUTS(s, input_types):
        if "model" in input_types:
            provided_type = input_types["model"]
            
            # Allow standard models or the specific WanVideoWrapper type
            allowed = ["MODEL", "WANVIDEOMODEL"]
            
            if provided_type not in allowed:
                return f"Input must be MODEL or WANVIDEOMODEL. Received: {provided_type}"
        
        return True

    RETURN_TYPES = (any_type,)
    RETURN_NAMES = ("model",)
    FUNCTION = "apply_compile"
    CATEGORY = "optimization"
    DESCRIPTION = """Apply torch.compile to model with specified settings.

This node wraps the model's forward pass with torch.compile for acceleration.
Use with TorchCompileSpeedSettings node for optimal configuration.

Author: eddy
"""

    def apply_compile(self, model, compile_args):
        backend   = compile_args["backend"]
        mode      = compile_args["mode"]
        dynamic   = compile_args["dynamic"]
        fullgraph = compile_args["fullgraph"]
        
        base_model = model.model
        if hasattr(base_model, "diffusion_model"):
            target_module = base_model.diffusion_model
        else:
            target_module = base_model

        if compile_args.get("speed_preset", False) or compile_args.get("experimental_ptx", False):
            try:
                from torch._inductor import config as inductor_config
                
                inductor_config.triton.cudagraphs      = False
                inductor_config.max_autotune           = True
                inductor_config.max_autotune_pointwise = True
                inductor_config.max_autotune_gemm      = True
                
                if hasattr(inductor_config, 'max_autotune_conv'):
                    inductor_config.max_autotune_conv = True
                if compile_args.get("experimental_ptx", False):
                    if hasattr(inductor_config, 'coordinate_descent_tuning'):
                        inductor_config.coordinate_descent_tuning = True
                    if hasattr(inductor_config, 'triton') and hasattr(inductor_config.triton, 'use_fast_math'):
                        inductor_config.triton.use_fast_math = compile_args.get("ptx_fast_math", True)
                        
                print("[TorchCompileSpeed] Applied inductor config")
            except Exception as e:
                print(f"[TorchCompileSpeed] Warning: Could not apply inductor config: {e}")

        try:
            torch._dynamo.config.cache_size_limit = compile_args["dynamo_cache_size_limit"]
            torch._dynamo.config.recompile_limit  = compile_args.get("dynamo_recompile_limit", 128)
        except Exception as e:
            print(f"[TorchCompileSpeed] Warning: Could not set dynamo config: {e}")

        if compile_args.get("experimental_ptx", False):
            try:
                import os
                
                if compile_args.get("ptx_cache_dir"):
                    os.environ["TRITON_CACHE_DIR"] = str(compile_args.get("ptx_cache_dir"))
                    
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                
                device = next(iter(target_module.parameters())).device if any(True for _ in target_module.parameters()) else torch.device("cuda" if torch.cuda.is_available() else "cpu")
                dtype  = torch.bfloat16 if hasattr(torch, 'bfloat16') else torch.float16
                warmed = False
                
                try:
                    import triton
                    import triton.ops as tops
                    
                    if torch.cuda.is_available():
                        a = torch.randn(256, 256, device=device, dtype=dtype)
                        b = torch.randn(256, 256, device=device, dtype=dtype)
                        for _ in range(int(compile_args.get("warmup_runs", 1))):
                            _ = tops.matmul(a, b)
                        torch.cuda.synchronize()
                        warmed = True
                        print("[TorchCompileSpeed] PTX warmup via triton.ops.matmul")
                except Exception:
                    pass
                    
                if not warmed and torch.cuda.is_available():
                    class _Matmul(torch.nn.Module):
                        def forward(self, x, y):
                            return torch.matmul(x, y)
                            
                    mod  = _Matmul().to(device=device, dtype=dtype).eval()
                    cmod = torch.compile(mod, backend=backend, fullgraph=fullgraph, mode=mode, dynamic=dynamic)
                    a    = torch.randn(512, 512, device=device, dtype=dtype)
                    b    = torch.randn(512, 512, device=device, dtype=dtype)
                    
                    for _ in range(int(compile_args.get("warmup_runs", 1))):
                        with torch.no_grad():
                            _ = cmod(a, b)
                    torch.cuda.synchronize()
                    print("[TorchCompileSpeed] PTX warmup via torch.compile(matmul)")
            except Exception as e:
                print(f"[TorchCompileSpeed] Warning: PTX warmup failed: {e}")

        model_clone  = model.clone()
        clone_base   = model_clone.model
        clone_target = clone_base.diffusion_model if hasattr(clone_base, "diffusion_model") else clone_base

        try:
            cache_enabled    = compile_args.get("reuse_if_similar", True)
            sig              = (id(target_module), backend, mode, dynamic, fullgraph)
            compiled_forward = None
            
            if cache_enabled:
                cached_map = COMPILED_FORWARD_CACHE.get(target_module)
                if cached_map is None:
                    cached_map = {}
                    COMPILED_FORWARD_CACHE[target_module] = cached_map
                if sig in cached_map:
                    compiled_forward = cached_map[sig]
                    print("[TorchCompileSpeed] Reused compiled forward from cache")
                else:
                    original_forward = clone_target.forward
                    compiled_forward = torch.compile(
                        original_forward,
                        backend=backend,
                        fullgraph=fullgraph,
                        mode=mode,
                        dynamic=dynamic
                    )
                    cached_map[sig] = compiled_forward
                    print(f"[TorchCompileSpeed] Compiled and cached forward backend={backend}, mode={mode}, dynamic={dynamic}")
            else:
                original_forward = clone_target.forward
                compiled_forward = torch.compile(
                    original_forward,
                    backend=backend,
                    fullgraph=fullgraph,
                    mode=mode,
                    dynamic=dynamic
                )
                
            clone_target.forward = compiled_forward
        except Exception as e:
            print(f"[TorchCompileSpeed] ERROR: Compilation failed: {e}")
            return (model,)

        return (model_clone,)

NODE_CLASS_MAPPINGS = {
    "TorchCompileSpeedSettings": TorchCompileSpeedSettings,
    "ApplyTorchCompile": ApplyTorchCompile,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TorchCompileSpeedSettings": "Torch Compile Speed Settings",
    "ApplyTorchCompile": "Apply Torch Compile",
}

