"""Load and process human behavioral data from OIID dataset."""
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
import json
from typing import Dict, List, Tuple
from collections import defaultdict


class HumanDataLoader:
    """Load human behavioral data from OIID dataset."""

    def __init__(self, dataset_root: str):
        """
        Initialize human data loader.

        Args:
            dataset_root: Path to ds005226 dataset
        """
        self.dataset_root = Path(dataset_root)
        self.behavioral_data_dir = self.dataset_root / 'derivatives' / 'Behavioral_data'

        # Check if behavioral data exists
        if not self.behavioral_data_dir.exists():
            raise FileNotFoundError(
                f"Behavioral data directory not found: {self.behavioral_data_dir}\n"
                f"Expected structure: {dataset_root}/derivatives/Behavioral_data/"
            )

    def load_all_subjects(self, subject_ids: List[int] = None) -> pd.DataFrame:
        """
        Load behavioral data for all or specified subjects.

        Args:
            subject_ids: List of subject IDs (1-65). If None, load all.

        Returns:
            DataFrame with columns:
            - subject_id: Subject ID (01-65)
            - stimulus_file: e.g., 'Aircraft1_10%_01.jpg'
            - correct: Whether subject answered correctly (bool)
            - response_time: RT in seconds (float)
            - occlusion_level: 0.1, 0.75, or 0.9
            - aircraft_type: 'Aircraft1' or 'Aircraft2'
        """
        all_data = []

        # Find all behavioral CSV files
        csv_files = sorted(self.behavioral_data_dir.glob('sub-*.csv'))

        if len(csv_files) == 0:
            # Try alternative structure: reading from func/*.tsv
            return self._load_from_events_files(subject_ids)

        for csv_file in csv_files:
            # Extract subject ID from filename
            sub_id_str = csv_file.stem.split('-')[1]  # e.g., 'sub-01.csv' -> '01'
            sub_id = int(sub_id_str)

            # Skip if not in requested subjects
            if subject_ids is not None and sub_id not in subject_ids:
                continue

            # Load CSV
            try:
                df = pd.read_csv(csv_file)
                df['subject_id'] = sub_id_str
                all_data.append(df)
            except Exception as e:
                print(f"Warning: Failed to load {csv_file}: {e}")
                continue

        if len(all_data) == 0:
            raise ValueError("No behavioral data found!")

        # Concatenate all subjects
        combined_df = pd.concat(all_data, ignore_index=True)

        # Standardize column names and extract info
        combined_df = self._standardize_dataframe(combined_df)

        return combined_df

    def _load_from_events_files(self, subject_ids: List[int] = None) -> pd.DataFrame:
        """
        Load behavioral data from events.tsv files.

        This is a fallback method when derivatives/Behavioral_data/ doesn't exist.
        """
        all_data = []

        for sub_id in range(1, 66):
            if subject_ids is not None and sub_id not in subject_ids:
                continue

            sub_id_str = f"{sub_id:02d}"
            sub_dir = self.dataset_root / f"sub-{sub_id_str}" / "ses-01" / "func"

            if not sub_dir.exists():
                continue

            # Find event files
            event_files = sorted(sub_dir.glob("*_task-image_run-*_events.tsv"))

            for event_file in event_files:
                try:
                    df = pd.read_csv(event_file, sep='\t')
                    df = df[df['stim_file'] != 'rest.jpg'].copy()  # Remove rest trials
                    df['subject_id'] = sub_id_str
                    all_data.append(df)
                except Exception as e:
                    print(f"Warning: Failed to load {event_file}: {e}")

        if len(all_data) == 0:
            raise ValueError("No event files found!")

        combined_df = pd.concat(all_data, ignore_index=True)
        combined_df = self._standardize_dataframe(combined_df)

        return combined_df

    def _standardize_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize column names and extract useful info."""

        # Rename columns if needed
        column_mapping = {
            'stim_file': 'stimulus_file',
            'stimuli': 'stimulus_file',  # OIID dataset uses 'stimuli'
            'stim_lable': 'label',  # Note: dataset has typo
            'levelOfOcclusion': 'occlusion_level',
            'response.corr': 'correct',
            'response.rt': 'response_time',
            'Keypress_Score': 'correct',  # OIID: 1=correct, 0=incorrect
            'reaction_time (ms)': 'response_time_ms',
        }

        df = df.rename(columns=column_mapping)

        # Convert response time from ms to seconds if needed
        if 'response_time_ms' in df.columns:
            df['response_time'] = df['response_time_ms'] / 1000.0

        # Extract aircraft type from stimulus filename
        if 'stimulus_file' in df.columns:
            df['aircraft_type'] = df['stimulus_file'].str.extract(r'(Aircraft\d)')[0]

        # Ensure occlusion_level is float (convert from percentage if needed)
        if 'occlusion_level' in df.columns:
            # Check if values are strings like "10%", "70%", "90%"
            if df['occlusion_level'].dtype == 'object':
                # Remove '%' and convert to decimal
                df['occlusion_level'] = df['occlusion_level'].str.replace('%', '').astype(float) / 100.0
            else:
                # Check if values are like 10, 75, 90 (percentage) or 0.1, 0.75, 0.9 (decimal)
                if df['occlusion_level'].max() > 1:
                    df['occlusion_level'] = df['occlusion_level'] / 100.0
                else:
                    df['occlusion_level'] = df['occlusion_level'].astype(float)

        # Ensure correct is boolean
        if 'correct' in df.columns:
            df['correct'] = df['correct'].astype(bool)

        return df

    def compute_performance_by_image(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute human performance aggregated by image.

        Args:
            df: DataFrame from load_all_subjects()

        Returns:
            DataFrame with columns:
            - stimulus_file
            - occlusion_level
            - aircraft_type
            - human_accuracy: Proportion of subjects who got it correct
            - correct_count: Number of subjects who got it correct
            - total_count: Total number of subjects who saw this image
            - mean_rt: Mean response time (if available)
        """
        grouped = df.groupby('stimulus_file').agg({
            'correct': ['mean', 'sum', 'count'],
            'occlusion_level': 'first',
            'aircraft_type': 'first',
            'response_time': 'mean' if 'response_time' in df.columns else lambda x: np.nan
        }).reset_index()

        # Flatten column names
        grouped.columns = [
            'stimulus_file',
            'human_accuracy',
            'correct_count',
            'total_count',
            'occlusion_level',
            'aircraft_type',
            'mean_rt'
        ]

        return grouped

    def compute_performance_by_occlusion(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute human performance by occlusion level.

        Returns:
            DataFrame with columns:
            - occlusion_level
            - mean_accuracy
            - std_accuracy
            - num_images
            - num_trials
        """
        # First get per-image performance
        img_perf = self.compute_performance_by_image(df)

        # Then aggregate by occlusion level
        grouped = img_perf.groupby('occlusion_level').agg({
            'human_accuracy': ['mean', 'std'],
            'stimulus_file': 'count',
        }).reset_index()

        grouped.columns = [
            'occlusion_level',
            'mean_accuracy',
            'std_accuracy',
            'num_images'
        ]

        # Also get total trial count
        trial_counts = df.groupby('occlusion_level').size()
        grouped['num_trials'] = grouped['occlusion_level'].map(trial_counts)

        return grouped

    def save_to_json(self, df: pd.DataFrame, output_path: str):
        """Save human performance data to JSON."""
        # Convert to dictionary format
        data = {}
        for _, row in df.iterrows():
            data[row['stimulus_file']] = {
                'human_accuracy': float(row['human_accuracy']),
                'correct_count': int(row['correct_count']),
                'total_count': int(row['total_count']),
                'occlusion_level': float(row['occlusion_level']),
                'aircraft_type': str(row['aircraft_type']),
            }

        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"Saved human performance data to {output_path}")


def main():
    """Load human data and save to files."""
    import argparse

    parser = argparse.ArgumentParser(description="Load human behavioral data from OIID")
    parser.add_argument(
        '--dataset-root',
        type=str,
        default='E:/Dataset/ds005226',
        help='Path to OIID dataset'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/human_performance',
        help='Output directory for processed data'
    )
    args = parser.parse_args()

    # Create output directory
    output_dir = project_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*60)
    print("Loading Human Behavioral Data")
    print("="*60)

    # Load data
    loader = HumanDataLoader(args.dataset_root)

    print("\n1. Loading all subjects...")
    all_data = loader.load_all_subjects()
    print(f"   Loaded {len(all_data)} trials from {all_data['subject_id'].nunique()} subjects")

    # Save raw data
    all_data.to_csv(output_dir / 'human_all_trials.csv', index=False)
    print(f"   Saved to {output_dir / 'human_all_trials.csv'}")

    # Compute per-image performance
    print("\n2. Computing per-image performance...")
    img_perf = loader.compute_performance_by_image(all_data)
    print(f"   {len(img_perf)} unique images")
    img_perf.to_csv(output_dir / 'human_performance_by_image.csv', index=False)
    print(f"   Saved to {output_dir / 'human_performance_by_image.csv'}")

    # Save as JSON too
    loader.save_to_json(img_perf, output_dir / 'human_performance_by_image.json')

    # Compute by occlusion level
    print("\n3. Computing performance by occlusion level...")
    occ_perf = loader.compute_performance_by_occlusion(all_data)
    print("\n" + str(occ_perf))
    occ_perf.to_csv(output_dir / 'human_performance_by_occlusion.csv', index=False)
    print(f"\n   Saved to {output_dir / 'human_performance_by_occlusion.csv'}")

    print("\n" + "="*60)
    print("Human data loading completed!")
    print("="*60)


if __name__ == "__main__":
    main()
