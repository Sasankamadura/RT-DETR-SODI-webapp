import json
import os

# Source directories
BASE_DIR = "d:/APPLICATION"
MODELS_SRC = {
    "base_rtdetr": "Normal RT-DETR", 
    "p2_p3_fusion": "Improved versions/1-P2-P3 fusion",
    "query_imp": "Improved versions/2-Query IMP",
    "aware_loss": "Improved versions/3- Aware Loss Reweighting"
}

OUTPUT_FILE = "d:/APPLICATION/backend/models_config.json"

def load_json(path):
    if not os.path.exists(path):
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
        }
    }

    for model_id, src_dir in MODELS_SRC.items():
        full_src_path = os.path.join(BASE_DIR, src_dir)
        
        # Load all requested JSONs
        profiling = load_json(os.path.join(full_src_path, "profiling_results.json"))
        model_eval = load_json(os.path.join(full_src_path, "model_evaluation.json"))
        fps_res = load_json(os.path.join(full_src_path, "fps_results.json"))
        eval_res = load_json(os.path.join(full_src_path, "evaluation_results.json"))
        detailed = load_json(os.path.join(full_src_path, "detailed_evaluation_results.json"))

        # Aggregate metrics
        fps_val = fps_res.get("fps_benchmark", {}).get("640x640", {}).get("fps_mean", 0) if fps_res else 0
        
        summary_metrics = {
            "mAP_50": f"{detailed.get('overall_metrics', {}).get('AP@0.5', 0)*100:.2f}%" if detailed else "N/A",
            "AP_Small": f"{detailed.get('overall_metrics', {}).get('AP_small', 0)*100:.2f}%" if detailed else "N/A",
            "FPS": f"{fps_val:.1f}" if fps_val else "N/A",
            "Params": f"{eval_res.get('model_size_mb', 0):.1f} MB" if eval_res else "N/A"
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
