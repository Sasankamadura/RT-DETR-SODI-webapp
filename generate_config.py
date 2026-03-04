import json
import os
import glob
import shutil
import re

# Paths
BASE_DIR = "d:/APPLICATION"
MODELS_DIR = os.path.join(BASE_DIR, "backend/models")
OUTPUT_FILE = os.path.join(BASE_DIR, "backend/models_config.json")

# Sources
SOURCES = {
    "Final": os.path.join(BASE_DIR, "Final Models"),
    "Experimental": [
        os.path.join(BASE_DIR, "experimented models"),
        os.path.join(BASE_DIR, "Normal RT-DETR")
    ]
}

def slugify(text):
    return re.sub(r'[^a-z0-0_]+', '_', text.lower()).strip('_')

def find_file(directory, pattern):
    """Finds a file matching the pattern using glob."""
    candidates = glob.glob(os.path.join(directory, pattern))
    if candidates:
        return candidates[0]
    return None

def load_json(path):
    if not path or not os.path.exists(path):
        return {}
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except:
        return {}

def get_decoder_layers(folder_name, all_folders):
    """
    Folders ending in CRR = 3 decoder layers.
    Folders with a matching CRR counterpart = 6 layers.
    Others = 3 layers.
    """
    if folder_name.endswith("CRR"):
        return 3
    
    crr_counterpart = folder_name + " CRR"
    if crr_counterpart in all_folders:
        return 6
    
    return 3

def generate_config():
    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR)

    config_list = []
    
    # Get all folders in sources to check for counterparts
    all_folders = {}
    for group, root_dirs in SOURCES.items():
        if isinstance(root_dirs, str):
             root_dirs = [root_dirs]
             
        for root_dir in root_dirs:
            if not os.path.exists(root_dir):
                continue
                
            if "Normal RT-DETR" in root_dir:
                # Baseline is a single model folder itself
                all_folders[os.path.basename(root_dir)] = root_dir
            else:
                for d in os.listdir(root_dir):
                    if os.path.isdir(os.path.join(root_dir, d)):
                        all_folders[d] = os.path.join(root_dir, d)

    for folder_name, full_path in all_folders.items():
        # Determine Group
        group = "Experimental"
        if "Final Models" in full_path:
            group = "Final"

        # Determine Decoder Layers
        decoder_layers = get_decoder_layers(folder_name, all_folders)
        
        # Discover Files
        onnx_path = find_file(full_path, "*.onnx")
        if not onnx_path:
            # Try deeper search or specific names
            onnx_path = find_file(full_path, "model*.onnx")
        
        if not onnx_path:
            print(f"Skipping {folder_name}: No ONNX file found.")
            continue

        # Metadata from JSONs
        profiling = load_json(find_file(full_path, "profiling_results*.json"))
        model_eval = load_json(find_file(full_path, "model_evaluation*.json"))
        fps_res = load_json(find_file(full_path, "fps_results*.json"))
        eval_res = load_json(find_file(full_path, "evaluation_results*.json"))
        detailed = load_json(find_file(full_path, "detailed_evaluation_results*.json"))

        # Model ID and Path
        base_id = slugify(folder_name)
        model_id = f"{base_id}_{group.lower()}"
        dest_filename = f"{model_id}.onnx"
        dest_path = os.path.join(MODELS_DIR, dest_filename)

        # Copy ONNX if different or missing
        print(f"Processing {folder_name} ({group})...")
        shutil.copy2(onnx_path, dest_path)

        # Aggregate metrics
        fps_val = fps_res.get("fps_benchmark", {}).get("640x640", {}).get("fps_mean", 0) if fps_res else 0
        if not fps_val and "fps_mean" in fps_res: # fallback for different formats
            fps_val = fps_res["fps_mean"]
            
        total_params = profiling.get("layer_analysis", {}).get("total_params", 0) if profiling else 0
        if not total_params and "total_params" in model_eval:
             total_params = model_eval["total_params"]
             
        params_str = f"{total_params / 1e6:.1f} M" if total_params > 0 else "N/A"

        mAP = detailed.get('overall_metrics', {}).get('AP@0.5', 0)
        if not mAP and "mAP_0.5" in eval_res:
            mAP = eval_res["mAP_0.5"]

        summary_metrics = {
            "mAP_50": f"{mAP*100:.2f}%" if mAP else "N/A",
            "FPS": f"{fps_val:.1f}" if fps_val else "N/A",
            "Params": params_str,
            "Decoders": str(decoder_layers)
        }

        # Clean Name for Display
        display_name = folder_name
        # Remove version markers if any (fallback)
        display_name = re.sub(r'\s*\((Final|Experimental|Baseline)\)', '', display_name, flags=re.IGNORECASE)
        # Remove leading numbers and dashes (e.g., "6_1 - SLIM p2 layer" -> "SLIM p2 layer")
        display_name = re.sub(r'^[\d\._\s-]+', '', display_name)
        
        entry = {
            "id": model_id,
            "name": display_name,
            "description": f"{group} version of {display_name} with {decoder_layers} decoder layers.",
            "path": f"backend/models/{dest_filename}",
            "full_metrics": {
                "profiling": profiling,
                "model_evaluation": model_eval,
                "fps": fps_res,
                "evaluation": eval_res,
                "detailed": detailed
            },
            "summary_metrics": summary_metrics,
            "version": group,
            "decoder_layers": decoder_layers
        }
        config_list.append(entry)

    # Sort by group (Final first, then Experimental)
    group_order = {"Final": 0, "Experimental": 1}
    config_list.sort(key=lambda x: (group_order.get(x["version"], 2), x["name"]))

    with open(OUTPUT_FILE, 'w') as f:
        json.dump(config_list, f, indent=2)
    
    print(f"Successfully generated {OUTPUT_FILE} with {len(config_list)} models.")

if __name__ == "__main__":
    generate_config()
