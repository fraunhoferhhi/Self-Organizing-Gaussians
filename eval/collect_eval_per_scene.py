import os
import json
import click
import pandas as pd
from pathlib import Path

@click.command()
@click.option('--output-dir', required=True, type=click.Path(), help="The output directory where results will be stored.")
@click.option('--dataset', required=True, type=str, help="The dataset name.")
@click.option('--scene', required=True, type=str, help="The scene name.")
@click.option('--model-path', required=True, type=click.Path(exists=True), help="The path to the model directory.")
@click.option('--submethod', default="", type=str, help="Submethod name, if applicable.")
def process_data(output_dir, dataset, scene, model_path, submethod):
    results_dir = Path(output_dir) / 'results'
    results_dir.mkdir(parents=True, exist_ok=True)

    dataset_dir = results_dir / dataset
    dataset_dir.mkdir(parents=True, exist_ok=True)

    # stats.json created from decompress.py, with #Gaussians and Size [Bytes]
    stats_path = Path(model_path) / 'stats.json'
    with open(stats_path, 'r') as f:
        stats = json.load(f)

    # results.json created in decompressed model from running render.py and metrics.py
    results_path = Path(model_path) / 'decompressed_model/results.json'
    with open(results_path, 'r') as f:
        results = json.load(f)

    metrics = {**stats, **results.get("ours_1", {})}

    # Check if required values are present in the metrics dictionary
    required_keys = ['PSNR', 'SSIM', 'LPIPS', 'Size [Bytes]', '#Gaussians']
    for key in required_keys:
        if key not in metrics:
            raise ValueError(f"Missing required metric: {key}")

    # Scene CSV path within the results directory
    scene_csv_path = dataset_dir / f'{scene}.csv'

    # Define table structure
    columns = ['Submethod', 'PSNR', 'SSIM', 'LPIPS', 'Size [Bytes]', '#Gaussians']
    
    # Extract row values from metrics
    row_values = [
        metrics['PSNR'],
        metrics['SSIM'],
        metrics['LPIPS'],
        metrics['Size [Bytes]'],
        metrics['#Gaussians']
    ]

    # If CSV exists, load it; otherwise, create a new DataFrame
    if scene_csv_path.exists():
        df = pd.read_csv(scene_csv_path)
        updated = False

        # Check if submethod exists, and update if necessary
        if submethod in df['Submethod'].values:
            df.loc[df['Submethod'] == submethod, columns[1:]] = row_values
            updated = True
        else:
            # Append new row if submethod does not exist
            new_row = [submethod] + row_values
            df = pd.concat([df, pd.DataFrame([new_row], columns=columns)], ignore_index=True)
        
        if updated:
            action = "updated"
        else:
            action = "added"
    else:
        # Create a new DataFrame with the metrics
        initial_data = [[submethod] + row_values]
        df = pd.DataFrame(initial_data, columns=columns)
        
        action = "created"

    # Write the updated DataFrame back to the CSV file
    df.to_csv(scene_csv_path, index=False)
    
    print(f"The CSV file at {scene_csv_path} was {action}.")

if __name__ == '__main__':
    process_data()
