import json
import os

# Source directories
BASE_DIR = "d:/APPLICATION"
MODELS_SRC = {
    "base_rtdetr": "Normal RT-DETR", 
    "p2_p3_fusion": "Improved versions/1-P2-P3 fusion",
    "query_imp": "Improved versions/2-Query IMP",
    "aware_loss": "Improved versions/3- Aware Loss Reweighting",
    "gnconv": "Improved versions/5 - gnconv",
    "p2_layer": "Improved versions/6 - p2 layer",
    "slim_p2": "Improved versions/6_1 - SLIM p2 layer",
    "gnconv_p2": "Improved versions/7 - gnconv + p2 layer",
    "gnconv_p2_repvgg": "Improved versions/7_1 - gnconv + p2 layer + Repvgg",
    "gnconv_slim_p2_repvgg": "Improved versions/7_2 - gnconv + SLIM p2 layer + Repvgg"
}

OUTPUT_FILE = "d:/APPLICATION/backend/models_config.json"

def find_json(directory, filename_pattern):
    """Finds a file matching the pattern (ignoring (1) suffix differences)"""
    import glob
    # Try exact match first
    exact_path = os.path.join(directory, filename_pattern)
    if os.path.exists(exact_path):
        return exact_path
    
    # Try with (1) suffix or wildcard
    base_name, ext = os.path.splitext(filename_pattern)
    # Search for files that start with the basename
    candidates = glob.glob(os.path.join(directory, f"{base_name}*{ext}"))
    if candidates:
        return candidates[0] # Return the first match
    return None

def load_json(path_or_dir, filename=None):
    path = path_or_dir
    if filename:
        path = find_json(path_or_dir, filename)
    
    if not path or not os.path.exists(path):
        return {}
    with open(path, 'r') as f:
        return json.load(f)

def generate_config():
    config_list = []
    
    # Define metadata
    meta = {
        "base_rtdetr": {
            "name": "Normal RT-DETR", 
            "description": "Baseline RT-DETR-R18 model.",
            "path": "backend/models/base_rtdetr.onnx"
        },
        "p2_p3_fusion": {
            "name": "Improved: P2-P3 Fusion", 
            "description": "Enhanced with P2-P3 feature fusion for small objects.",
            "path": "backend/models/p2_p3_fusion.onnx"
        },
        "query_imp": {
            "name": "Improved: Query IMP", 
            "description": "Scale-aware query initialization strategy.",
            "path": "backend/models/query_imp.onnx"
        },
        "aware_loss": {
            "name": "Improved: Aware Loss", 
            "description": "Diffculty-aware loss reweighting function.",
            "path": "backend/models/aware_loss.onnx"
        },
        "gnconv": {
            "name": "Improved: GNConv", 
            "description": "Group Normalization Convolution integration.",
            "path": "backend/models/gnconv.onnx"
        },
        "p2_layer": {
            "name": "Improved: P2 Layer", 
            "description": "Dedicated P2 layer for high-resolution feature processing.",
            "path": "backend/models/p2_layer.onnx"
        },
        "slim_p2": {
            "name": "Improved: Slim P2", 
            "description": "Optimized lightweight P2 layer implementation.",
            "path": "backend/models/slim_p2.onnx"
        },
        "gnconv_p2": {
            "name": "Improved: GNConv + P2", 
            "description": "Combination of GNConv and P2 layer.",
            "path": "backend/models/gnconv_p2.onnx"
        },
        "gnconv_p2_repvgg": {
            "name": "Improved: GNConv + P2 + RepVGG", 
            "description": "Advanced architecture with GNConv, P2, and RepVGG blocks.",
            "path": "backend/models/gnconv_p2_repvgg.onnx"
        },
        "gnconv_slim_p2_repvgg": {
            "name": "Improved: GNConv + Slim P2 + RepVGG", 
            "description": "SOTA: GNConv + Slim P2 + RepVGG fusion.",
            "path": "backend/models/gnconv_slim_p2_repvgg.onnx"
        }
    }

    for model_id, src_dir in MODELS_SRC.items():
        full_src_path = os.path.join(BASE_DIR, src_dir)
        
        # Load all requested JSONs
        profiling = load_json(full_src_path, "profiling_results.json")
        model_eval = load_json(full_src_path, "model_evaluation.json")
        fps_res = load_json(full_src_path, "fps_results.json")
        eval_res = load_json(full_src_path, "evaluation_results.json")
        detailed = load_json(full_src_path, "detailed_evaluation_results.json")

        # Aggregate metrics
        fps_val = fps_res.get("fps_benchmark", {}).get("640x640", {}).get("fps_mean", 0) if fps_res else 0
        
        # Get total params from profiling
        total_params = profiling.get("layer_analysis", {}).get("total_params", 0) if profiling else 0
        params_str = f"{total_params / 1e6:.1f} M" if total_params > 0 else "N/A"

        summary_metrics = {
            "mAP_50": f"{detailed.get('overall_metrics', {}).get('AP@0.5', 0)*100:.2f}%" if detailed else "N/A",
            "AP_Small": f"{detailed.get('overall_metrics', {}).get('AP_small', 0)*100:.2f}%" if detailed else "N/A",
            "FPS": f"{fps_val:.1f}" if fps_val else "N/A",
            "Params": params_str
        }

        entry = {
            "id": model_id,
            "name": meta[model_id]["name"],
            "description": meta[model_id]["description"],
            "path": meta[model_id]["path"],
            "full_metrics": {
                "profiling": profiling,
                "model_evaluation": model_eval,
                "fps": fps_res,
                "evaluation": eval_res,
                "detailed": detailed
            },
            "summary_metrics": summary_metrics
        }
        config_list.append(entry)

    with open(OUTPUT_FILE, 'w') as f:
        json.dump(config_list, f, indent=2)
    
    print("Successfully generated models_config.json")

if __name__ == "__main__":
    generate_config()
