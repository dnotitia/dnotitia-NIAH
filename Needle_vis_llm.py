# Batch visualization and file storage for LLM needle-in-haystack experiments

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import json
import os
import glob
from typing import List, Dict, Optional
from tqdm import tqdm


class NeedleVisualizer:
    def __init__(self, base_result_dir: str, exclude_dirs: List[str] = None, force_reprocess: bool = False):
        """
        Initialize the visualizer
        @param base_result_dir: Root directory for results
        @param exclude_dirs: List of directories to exclude
        @param force_reprocess: Whether to force reprocessing all experiments, even if visualization results already exist
        """
        self.base_result_dir = base_result_dir
        self.visualization_dir = os.path.join(base_result_dir, "visualizations")
        self.exclude_dirs = exclude_dirs or []
        self.force_reprocess = force_reprocess
        self.processed_experiments = set()  # Track processed experiments
        os.makedirs(self.visualization_dir, exist_ok=True)

        # Load existing processing records
        self.record_file = os.path.join(self.visualization_dir, "processed_experiments.json")
        self._load_processed_records()

    def _load_processed_records(self):
        """Load records of processed experiments"""
        try:
            if os.path.exists(self.record_file):
                with open(self.record_file, 'r') as f:
                    self.processed_experiments = set(json.load(f))
        except Exception as e:
            print(f"Error loading processing records: {str(e)}")
            self.processed_experiments = set()

    def _save_processed_records(self):
        """Save records of processed experiments"""
        try:
            with open(self.record_file, 'w') as f:
                json.dump(list(self.processed_experiments), f)
        except Exception as e:
            print(f"Error saving processing records: {str(e)}")

    def _should_process_experiment(self, experiment_path: str) -> bool:
        """Determine whether this experiment needs to be processed"""
        dataset = self._extract_dataset_name(experiment_path)
        model_name = self._extract_model_name(experiment_path)

        if self.force_reprocess:
            return True

        if experiment_path in self.processed_experiments:
            return False

        # Check if heatmap files exist
        filename = f'{dataset}_{model_name}_heatmap.png'
        vis_path = os.path.join(self.visualization_dir, filename)
        original_path = os.path.join(experiment_path, filename)

        return not (os.path.exists(vis_path) and os.path.exists(original_path))

    def load_experiment_data(self, experiment_path: str) -> pd.DataFrame:
        """Load data for a single experiment"""
        data = []

        json_files = glob.glob(f"{experiment_path}/*.json") + glob.glob(f"{experiment_path}/*.jsonl")

        if not json_files:
            tqdm.write(f"Warning: No JSON files found in {os.path.basename(experiment_path)}")
            return pd.DataFrame()

        # Process found JSON files
        for file in tqdm(json_files, desc="Processing JSON files", unit="file", leave=False):
            try:
                with open(file, 'r') as f:
                    json_data = json.load(f)
                    # Verify required fields exist
                    required_fields = ["depth_percent", "context_length", "score"]
                    if not all(field in json_data for field in required_fields):
                        tqdm.write(f"Warning: {os.path.basename(file)} missing required fields")
                        continue

                    data.append({
                        "Document Depth": json_data["depth_percent"],
                        "Context Length": json_data["context_length"],
                        "Score": json_data["score"],
                        "Dataset": self._extract_dataset_name(experiment_path),
                        "Model": self._extract_model_name(experiment_path),  # Use folder name as model name
                    })
            except json.JSONDecodeError:
                tqdm.write(f"Error: Cannot parse JSON file {os.path.basename(file)}")
            except Exception as e:
                tqdm.write(f"Error processing file {os.path.basename(file)}: {str(e)}")

        df = pd.DataFrame(data)
        if df.empty:
            tqdm.write(f"Warning: Data loaded from {os.path.basename(experiment_path)} is empty")
        return df

    def create_heatmap(self, df: pd.DataFrame, save_path: str, title: str):
        """Generate heatmap"""
        if df.empty:
            tqdm.write(f"Warning: Skipping heatmap generation for empty dataset {os.path.basename(save_path)}")
            return

        try:
            pivot_table = pd.pivot_table(
                df,
                values='Score',
                index=['Document Depth', 'Context Length'],
                aggfunc='mean'
            ).reset_index()

            # Reorganize data into format required for heatmap
            pivot_table = pivot_table.pivot(
                index="Document Depth",
                columns="Context Length",
                values="Score"
            )

            plt.figure(figsize=(17.5, 8))
            cmap = LinearSegmentedColormap.from_list("custom_cmap", ["#F0496E", "#EBB839", "#0CD79F"])

            sns.heatmap(
                pivot_table,
                fmt="",  # Remove numeric annotations
                cmap=cmap,
                cbar_kws={'label': 'Score'},
                vmin=0,
                vmax=10,
                annot=False  # Don't show specific values
            )

            plt.title(title)
            plt.xlabel('Token Limit')
            plt.ylabel('Depth Percent')
            plt.xticks(rotation=45)
            plt.yticks(rotation=0)
            plt.tight_layout()

            # Ensure save path directory exists
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            tqdm.write(f"Successfully generated heatmap: {os.path.basename(save_path)}")

        except Exception as e:
            tqdm.write(f"Error generating heatmap {os.path.basename(save_path)}: {str(e)}")
            plt.close()  # Ensure figure is closed even if error occurs

    def process_all_experiments(self):
        """Process all experiment data"""
        all_data = pd.DataFrame()
        experiment_paths = self._get_experiment_paths()
        heatmap_count = 0  # Add heatmap counter

        if not experiment_paths:
            print("Warning: No experiment paths found")
            return all_data

        # Clean invalid records from processed records
        self._clean_processed_records()

        # Filter experiments that need processing
        experiments_to_process = [
            path for path in experiment_paths
            if self._should_process_experiment(path)
        ]

        if not experiments_to_process:
            print("All experiments have been processed, no reprocessing needed")
            return self._load_existing_data()

        print(f"Found {len(experiments_to_process)} experiment paths that need processing")

        for experiment_path in tqdm(experiments_to_process, desc="Processing experiments", unit="exp"):
            try:
                df = self.load_experiment_data(experiment_path)
                if not df.empty:
                    all_data = pd.concat([all_data, df], ignore_index=True)

                    dataset = self._extract_dataset_name(experiment_path)
                    model_name = self._extract_model_name(experiment_path)

                    title = f'{dataset} - {model_name}'
                    filename = f'{dataset}_{model_name}_heatmap.png'

                    original_save_path = os.path.join(experiment_path, filename)
                    vis_save_path = os.path.join(self.visualization_dir, filename)

                    self.create_heatmap(df, original_save_path, title)
                    self.create_heatmap(df, vis_save_path, title)
                    heatmap_count += 2

                    self.processed_experiments.add(experiment_path)

            except Exception as e:
                tqdm.write(f"Error processing experiment {experiment_path}: {str(e)}")

        # Save processing records
        self._save_processed_records()

        # Print heatmap generation statistics
        print(f"\nGenerated {heatmap_count} heatmaps")

        # Merge existing data with new data
        if not all_data.empty:
            existing_data = self._load_existing_data()
            if not existing_data.empty:
                all_data = pd.concat([existing_data, all_data], ignore_index=True)

            summary_path = os.path.join(self.visualization_dir, 'llm_all_experiments_data.csv')
            try:
                all_data.to_csv(summary_path, index=False)
                print(f"Summary data saved to: {summary_path}")
            except Exception as e:
                print(f"Error saving summary data: {str(e)}")

        return all_data

    def _load_existing_data(self) -> pd.DataFrame:
        """Load existing summary data"""
        summary_path = os.path.join(self.visualization_dir, 'llm_all_experiments_data.csv')
        if os.path.exists(summary_path):
            try:
                return pd.read_csv(summary_path)
            except Exception as e:
                print(f"Error loading existing summary data: {str(e)}")
        return pd.DataFrame()

    def _get_experiment_paths(self) -> List[str]:
        """Get all experiment paths"""
        # First get all case_dirs (dataset directories)
        case_dirs = [d for d in glob.glob(os.path.join(self.base_result_dir, "*"))
                     if os.path.isdir(d) and not any(exclude_dir in d for exclude_dir in self.exclude_dirs)]
        print(f"Found {len(case_dirs)} dataset directories")

        # For each dataset directory, get model directories under it
        experiment_paths = []
        for case_dir in tqdm(case_dirs, desc="Scanning datasets", unit="dir"):
            model_dirs = [d for d in glob.glob(os.path.join(case_dir, "*"))
                          if os.path.isdir(d)]
            experiment_paths.extend(model_dirs)

        print(f"Total found {len(experiment_paths)} experiment paths")
        return experiment_paths

    @staticmethod
    def _extract_dataset_name(path: str) -> str:
        """Extract dataset name (case_name) from path"""
        # Return second-to-last directory name
        parts = path.split(os.sep)
        return parts[-2]

    @staticmethod
    def _extract_model_name(path: str) -> str:
        """Extract model name from path"""
        return os.path.basename(path)

    def _clean_processed_records(self):
        """Clean invalid records from processed records"""
        invalid_records = set()

        for experiment_path in tqdm(self.processed_experiments, desc="Validating experiment records", unit="exp"):
            dataset = self._extract_dataset_name(experiment_path)
            model_name = self._extract_model_name(experiment_path)

            # Check if heatmap files exist
            filename = f'{dataset}_{model_name}_heatmap.png'
            vis_path = os.path.join(self.visualization_dir, filename)
            original_path = os.path.join(experiment_path, filename)

            # If heatmap is incomplete, mark as invalid record
            if not (os.path.exists(vis_path) and os.path.exists(original_path)):
                tqdm.write(f"Found invalid record: {dataset}/{model_name}")
                invalid_records.add(experiment_path)

        # Remove invalid records from processed records
        self.processed_experiments -= invalid_records
        if invalid_records:
            print(f"Cleaned {len(invalid_records)} invalid records")
            # Save updated records
            self._save_processed_records()


# Usage example
if __name__ == "__main__":
    base_dir = "./llm_multi_needle/results"
    # If you need to exclude certain folders, add them to the exclude_dirs list
    exclude_dirs = ["visualizations"]

    # If you need to force reprocessing of all experiments, set force_reprocess=True
    visualizer = NeedleVisualizer(base_dir, exclude_dirs, force_reprocess=True)
    all_experiments_data = visualizer.process_all_experiments()
