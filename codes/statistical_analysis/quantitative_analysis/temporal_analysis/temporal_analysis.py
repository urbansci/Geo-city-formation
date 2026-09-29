# -*- coding: utf-8 -*-
# Author: Weiyu Zhang
# Date: 2024
# Description: Distributed time window analysis for geographical city formation models.
#              Uses Ray for parallel processing to compare model performance with and without
#              2nd nature geography features across different historical time periods.

# Python standard library imports
import os
import random
import copy
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import sys
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

import ray
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import shap
from xgboost import XGBClassifier
from sklearn.metrics import (
   average_precision_score,
   f1_score,
   matthews_corrcoef
)
from utils import PreprocessPipeline
from models import DataSampler, XGBoostTrainer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TimeWindowFilter:
    def __init__(self, second_nature_path: str, window_size: int = 30, step: int = 1):
        """
        Initialize TimeWindowFilter for temporal analysis.
        
        Args:
            second_nature_path: Path to 2nd Nature Geography data file
            window_size: Time window size in years
            step: Step size for sliding window
        """
        self.window_size = window_size
        self.second_nature_path = second_nature_path
        self.second_nature_data = None
        self.birth_years = None
        self.step = step
        self._load_second_nature_data()
    
    def _load_second_nature_data(self):
        """Load 2nd Nature Geography data from file."""
        logger.info(f"Loading 2nd Nature Geography data from: {self.second_nature_path}")
        try:
            self.second_nature_data = pd.read_parquet(self.second_nature_path)
            logger.info(f"Successfully loaded 2nd Nature Geography data, shape: {self.second_nature_data.shape}")
        except Exception as e:
            logger.error(f"Failed to load 2nd Nature Geography data: {str(e)}")
            raise

    def get_2nd_nature_columns(self, year: int, distances: List[int] = [20, 50, 100]) -> List[str]:
        """Get 2nd Nature Geography column names for specified year."""
        birth_year = -1000
        self.birth_years = sorted(self.birth_years)
        for year_ in self.birth_years:
            if year_ >= year:
                birth_year = year_
                break
        return [f'Y{int(birth_year)}_city_count_{distance}km' for distance in distances]

    def merge_2nd_nature_data(self, df: pd.DataFrame, start_year: int) -> pd.DataFrame:
        """Merge original data with 2nd Nature Geography data."""
        nature_columns = [col for col in self.second_nature_data.columns if col.startswith('existing_cities_distance_')]
        nature_data_subset = self.second_nature_data.loc[
            (self.second_nature_data['start_year'] == start_year) &
            (self.second_nature_data['end_year'] == start_year + self.window_size),
            ['id'] + nature_columns]
        return df.merge(nature_data_subset, on='id', how='left')

    def get_time_windows_generator(self, df: pd.DataFrame):
        """Generate fixed-length time windows using generator pattern."""
        city_data = df[df['is_city'] == 1][['birth', 'id']].copy(deep=True)
        city_data['random'] = np.random.rand(len(city_data))
        city_data = city_data.sort_values(['birth', 'random'])
        city_data = city_data.drop('random', axis=1)
        
        unique_years = sorted(city_data['birth'].unique())
        min_year = unique_years[0]
        max_year = unique_years[-1]
        self.birth_years = unique_years
        
        # Generate time windows
        for start_year in range(int(min_year), int(max_year - self.window_size + 1), self.step):
            end_year = start_year + self.window_size
            window_cities = city_data[
                (city_data['birth'] >= start_year) & 
                (city_data['birth'] < end_year)
            ]['id'].tolist()
            
            if len(window_cities) > 0:  # Ensure at least one sample in window
                yield start_year, end_year, window_cities.copy()

    def filter_data(self, df: pd.DataFrame, start_year: int, end_year: int, city_ids: List[int]) -> pd.DataFrame:
        """Filter and process data for specific time window."""
        df = df.copy()
        df = self.merge_2nd_nature_data(df, start_year)
        df = df[~((df['birth'] < start_year) & (df['is_city'] == 1))]
        df.loc[(df['birth'] > end_year) & (df['is_city'] == 1), 'is_city'] = False
        return df


@ray.remote(num_gpus=0.3)
class ModelTrainer:
    def __init__(self, config):
        self.config = config
        
    def train_and_evaluate(self, x_train, y_train, x_test, y_test, with_2nd_nature=True):
        """Train XGBoost model and evaluate performance."""
        if not with_2nd_nature:
            # Remove 2nd nature geography columns
            nature_cols = [col for col in x_train.columns if col.startswith('existing_cities_distance_')]
            print("Removing 2nd nature columns: ", nature_cols)
            print("Features before removal: ", len(x_train.columns))
            x_train = x_train.drop(columns=nature_cols)
            x_test = x_test.drop(columns=nature_cols)
            print("Features after removal: ", len(x_train.columns))
            
        model = XGBClassifier(**self.config)
        model.fit(x_train, y_train)
        
        y_pred_proba = model.predict_proba(x_test)[:, 1]
        y_pred = model.predict(x_test)
        
        metrics = self._evaluate_model(model, x_test, y_test, y_pred, y_pred_proba)
        
        # Calculate SHAP values for feature importance
        explainer = shap.TreeExplainer(model)
        shap_values = explainer(x_test)
        
        # Calculate SHAP values for different feature groups
        feature_groups = {
            'clim': [col for col in x_test.columns if col.startswith('C')],
            'dem': [col for col in x_test.columns if col.startswith('D')],
        }
        
        for group_name, features in feature_groups.items():
            feature_idx = [i for i, col in enumerate(x_test.columns) if col in features]
            if feature_idx:
                explanation = shap_values[:, feature_idx]
                shap_sum = explanation.values.sum(1)
                metrics[f'shap_{group_name}'] = np.abs(shap_sum).mean()
            else:
                metrics[f'shap_{group_name}'] = 0.0
        
        metrics['positive_count'] = y_test.sum() + y_train.sum()
        return metrics

    def _evaluate_model(self, model, x_test, y_test, y_pred, y_pred_proba):
        """Evaluate model performance using multiple metrics."""
        
        metrics = {
            'f1_score': f1_score(y_test, y_pred),
            'mcc': matthews_corrcoef(y_test, y_pred),
            'pr_auc': average_precision_score(y_test, y_pred_proba)
        }
        return metrics


class RayTimeWindowAnalysis:
    def __init__(
        self,
        base_path: str,
        model_name: str,
        data_file_name: str,
        second_nature_path: str,
        save_dir: str,
        data_type: str = "pref",
        window_size: int = 300,
        min_samples: int = 300,
        n_repeats: int = 20,
        step: int = 50,
        n_gpus: int = 8,
    ):
        """Initialize time window analysis framework."""
        self.base_settings = {
            'base_path': base_path,
            'model_name': model_name,
            'data_file_name': data_file_name,
            'second_nature_path': second_nature_path,
            'data_type': data_type
        }
        
        self.current_climate_year = None
        self.data_type = data_type
        self.model_name = model_name
        self.target_test_ratio = 0.1

        self.window_size = window_size
        self.min_samples = min_samples
        self.n_repeats = n_repeats
        self.step = step
        self.n_gpus = n_gpus
        
        # Create save directory
        self.save_dir = Path(save_dir)
        
        # XGBoost configuration
        configs = json.load(open("./configs/configs.json"))
        self.model_config = configs["cn-pref"]["All_embedding"]
        
        # Initialize Ray
        if not ray.is_initialized():
            ray.init()
            
        # Create GPU trainer pool
        self.trainers = [
            ModelTrainer.remote(self.model_config)
            for _ in range(self.n_gpus * 3)
        ]

        self.pipeline = PreprocessPipeline(
            base_path=base_path,
            model_name=model_name,
            random_state=random.randint(0, 100000),
            test_ratio=0.2,
            data_file_name=data_file_name
        )

        self.sampler = DataSampler(random_state=random.randint(0, 100000), sampling_strategy=0.1)

        self.window_filter = TimeWindowFilter(
            second_nature_path=second_nature_path,
            window_size=window_size,
            step=step
        )
        
    def prepare_data(self, window_data):
        """Prepare training and test data for model training."""
        processed_data = self.pipeline.run_with_loaded_data(window_data)
        
        # Resample test set to maintain target ratio
        x_test = processed_data['x_test'].copy()
        y_test = processed_data['y_test'].copy()
        positive_count = y_test.sum()
        target_negative_count = int(positive_count / self.target_test_ratio)
        
        test_negative_mask = ~y_test
        test_negative_indices = y_test[test_negative_mask].index
        sampled_negative_indices = np.random.choice(
            test_negative_indices,
            size=target_negative_count,
            replace=False
        )
        
        test_indices = np.concatenate([
            y_test[y_test].index,
            sampled_negative_indices
        ])
        
        # Update test data
        processed_data['x_test'] = x_test.loc[test_indices]
        processed_data['y_test'] = y_test.loc[test_indices]
        used_nature_cols = [col for col in processed_data['x_test'].columns if col.startswith('existing_cities_distance_')]
        print("Used 2nd nature columns: ", used_nature_cols)
        
        # Prepare balanced training data
        balanced_data, y_train_balanced, y_test = self.sampler.prepare_balanced_data(
            processed_data,
            sampling_strategy='under',
            second_nature=used_nature_cols,
        )

        data = {
            'x_train': balanced_data["Terrain-Water&Clim"][0],
            'y_train': y_train_balanced,
            'x_test': balanced_data["Terrain-Water&Clim"][1],
            'y_test': y_test
        }
        
        return data

    def run_window_analysis(self, window_data, start_year, end_year):
        """Run analysis for single time window with multiple repetitions."""
        results = []
        logger.info(f"Preparing data for {self.n_repeats} repetitions")
        all_data = [
            self.prepare_data(window_data) 
            for _ in range(self.n_repeats)
        ]
        
        # Submit tasks for with and without 2nd nature separately
        with_tasks = []
        without_tasks = []
        
        logger.info("Submitting training tasks to Ray for parallel processing")

        n_trainers = len(self.trainers)
        n_trainers_per_group = n_trainers // 2
        
        # Submit tasks with 2nd nature features
        for i, data in enumerate(all_data): 
            trainer = self.trainers[i % n_trainers_per_group]
            
            with_task = trainer.train_and_evaluate.remote(
                data['x_train'], data['y_train'],
                data['x_test'], data['y_test'],
                True
            )
            with_tasks.append((i, with_task))
        
        # Prepare new data for without 2nd nature experiments
        all_data = []
        all_data = [
            self.prepare_data(window_data)
            for _ in range(self.n_repeats)
        ]

        # Submit tasks without 2nd nature features
        for i, data in enumerate(all_data):
            trainer_without = self.trainers[i % n_trainers_per_group + n_trainers_per_group]
            without_task = trainer_without.train_and_evaluate.remote(
                data['x_train'], data['y_train'],
                data['x_test'], data['y_test'],
                False
            )
            without_tasks.append((i, without_task))
        
        # Collect results from with 2nd nature experiments
        logger.info("Waiting for 'with 2nd nature' tasks to complete")
        for i, task in with_tasks:
            metrics = ray.get(task)
            result = {
                'start_year': start_year,
                'end_year': end_year,
                'window_size': self.window_size,
                'iteration': i + 1,
                'has_2nd_nature': True,
                **metrics
            }
            results.append(result)
            logger.info(f"Completed iteration {i+1} with 2nd nature")
        
        # Collect results from without 2nd nature experiments
        logger.info("Waiting for 'without 2nd nature' tasks to complete")
        for i, task in without_tasks:
            metrics = ray.get(task)
            result = {
                'start_year': start_year,
                'end_year': end_year,
                'window_size': self.window_size,
                'iteration': i + 1,
                'has_2nd_nature': False,
                **metrics
            }
            results.append(result)
            logger.info(f"Completed iteration {i+1} without 2nd nature")
        
        return results

   def process_window_data(self, filtered_df: pd.DataFrame, start_year: int, end_year: int) -> pd.DataFrame:
       """Process data for specific time window by updating climate data."""
       df = filtered_df
       mid_year = int((start_year + end_year) / 2)
       mid_year = mid_year - mid_year % 100
       
       logger.info(f"Updating climate data to year {mid_year}")
       df = self.pipeline.data_loader.load_agri_data(df, data_type=self.data_type, year=mid_year)
       df = self.pipeline.data_loader.load_climate_embeddings(df, data_type=self.data_type, year=mid_year)
       self.current_climate_year = mid_year
       return df

   def save_results(self, all_results):
       """Save experimental results to files."""
       timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
       
       # Separate results with and without 2nd nature
       results_with = pd.DataFrame([r for r in all_results if r['has_2nd_nature']])
       results_without = pd.DataFrame([r for r in all_results if not r['has_2nd_nature']])
       
       os.makedirs(self.save_dir / timestamp, exist_ok=True)

       # Save results
       results_with.to_csv(
           self.save_dir / timestamp / "time_window_analysis_with_2nd_nature.csv",
           index=False
       )
       results_without.to_csv(
           self.save_dir / timestamp / "time_window_analysis_without_2nd_nature.csv",
           index=False
       )
       
       # Save experiment configuration
       config = {
           'timestamp': timestamp,
           'window_size': self.window_size,
           'min_samples': self.min_samples,
           'n_repeats': self.n_repeats,
           'step': self.step,
           'model_config': self.model_config,
           **self.base_settings
       }
       
       with open(self.save_dir / timestamp / "experiment_config.json", 'w') as f:
           json.dump(config, f, indent=2)

    def run_analysis(self):
        """Run complete time window analysis across all time periods."""
        combined_df = pd.read_parquet(
            "/path/to/data/combined_df_in_polygon.parquet"
        )
        all_results = []
        
        # Remove existing climate columns to use temporal climate data
        combined_df = combined_df.drop(columns=[col for col in combined_df.columns if col.startswith('C')])
        
        for start_year, end_year, city_ids in self.window_filter.get_time_windows_generator(combined_df):
            logger.info(f"================= Processing time window: {start_year}-{end_year} ===================")
            
            # Check if minimum sample requirement is met
            if len(city_ids) < self.min_samples:
                logger.warning(f"Insufficient positive samples ({len(city_ids)} < {self.min_samples}), skipping window")
                continue

            try:
                filtered_df = self.window_filter.filter_data(
                    combined_df, start_year, end_year, city_ids
                )
                used_nature_cols = [col for col in filtered_df.columns if col.startswith('existing_cities_distance_')]
                processed_df = self.process_window_data(filtered_df, start_year, end_year)
                
                processed_df['is_city'] = processed_df['is_city'].astype(bool)
                
                window_results = self.run_window_analysis(processed_df, start_year, end_year)
                all_results.extend(window_results)

            except Exception as e:
                logger.error(f"Error processing window: {str(e)}", exc_info=True)
                continue
                
        logger.info(f"Analysis complete. Processed {len(all_results)} total iterations across all windows")
        self.save_results(all_results)


def main():
    """Main function to run time window analysis."""
    # Set GPU visibility
    os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3,4"
    data_name = "pref"
    
    analysis = RayTimeWindowAnalysis(
        base_path="/path/to/ssl/results",
        model_name=f"model_name", # Not used in this context
        data_file_name="", # Not used in this context
        second_nature_path="/path/to/existing_cities_samples.parquet",
        save_dir="/path/to/results/time_window_analysis/pref/",
        window_size=300,
        n_repeats=80,
        data_type=data_name,
        step=50,
        n_gpus=4
    )
    
    analysis.run_analysis()


if __name__ == "__main__":
   main()
