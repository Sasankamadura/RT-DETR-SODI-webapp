import json
import os

CONFIG_PATH = 'backend/models_config.json'

def main():
    if not os.path.exists(CONFIG_PATH):
        print(f"Error: {CONFIG_PATH} not found.")
        return

    with open(CONFIG_PATH, 'r') as f:
        data = json.load(f)

    # Header
    print("| Model Name | mAP 50 | mAP Small | FPS (Mean) | Latency (ms) | Params (M) |")
    print("| :--- | :--- | :--- | :--- | :--- | :--- |")

    for model in data:
        name = model.get('name', 'Unknown')
        
        # Metrics
        summary = model.get('summary_metrics', {})
        map50 = summary.get('mAP_50', 'N/A')
        map_Small = summary.get('AP_Small', 'N/A')
        
        # FPS
        full = model.get('full_metrics', {})
        fps_data = full.get('fps', {}).get('fps_benchmark', {}).get('640x640', {})
        fps = fps_data.get('fps_mean', 'N/A')
        if fps != 'N/A':
            fps = f"{fps:.2f}"
            
        latency = fps_data.get('latency_mean_ms', 'N/A')
        if latency != 'N/A':
            latency = f"{latency:.2f}"

        # Params
        profiling = full.get('profiling', {})
        layer_analysis = profiling.get('layer_analysis', {})
        total_params = layer_analysis.get('total_params', 0)
        
        params_m = "N/A"
        if total_params > 0:
            params_m = f"{total_params / 1e6:.2f} M"

        print(f"| {name} | {map50} | {map_Small} | {fps} | {latency} | {params_m} |")

if __name__ == "__main__":
    main()
