"""
Automatic Visualization Generation for EluSight
Generates professional plots for relationship discovery
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

class EluSightVisualizer:
    """
    Automatic visualization generator for input-output relationships
    """
    
    def __init__(self, output_dir: str = "elusight_plots"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")
    
    def plot_correlation_heatmap(self, df: pd.DataFrame, save: bool = True) -> plt.Figure:
        """Generate correlation heatmap showing input-output relationships"""
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Calculate correlations
        corr = df.select_dtypes(include=[np.number]).corr()
        
        # Create mask for upper triangle
        mask = np.triu(np.ones_like(corr, dtype=bool))
        
        # Plot heatmap
        sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', 
                   cmap='RdBu_r', center=0, 
                   square=True, linewidths=0.5,
                   cbar_kws={"shrink": 0.8}, ax=ax)
        
        ax.set_title('Input-Output Correlation Matrix', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save:
            plt.savefig(self.output_dir / 'correlation_heatmap.png', dpi=150, bbox_inches='tight')
            plt.savefig(self.output_dir / 'correlation_heatmap.svg', bbox_inches='tight')
        
        return fig
    
    def plot_feature_importance(self, feature_names: List[str], 
                                 importance_scores: List[float],
                                 title: str = "Feature Importance",
                                 save: bool = True) -> plt.Figure:
        """Generate feature importance bar plot"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Sort by importance
        sorted_idx = np.argsort(importance_scores)
        names_sorted = [feature_names[i] for i in sorted_idx]
        scores_sorted = [importance_scores[i] for i in sorted_idx]
        
        # Create horizontal bar plot
        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(names_sorted)))
        bars = ax.barh(names_sorted, scores_sorted, color=colors)
        
        # Add value labels
        for bar, score in zip(bars, scores_sorted):
            ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                   f'{score:.3f}', va='center', fontsize=10)
        
        ax.set_xlabel('Importance Score', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlim(0, max(scores_sorted) * 1.1)
        plt.tight_layout()
        
        if save:
            plt.savefig(self.output_dir / 'feature_importance.png', dpi=150, bbox_inches='tight')
            plt.savefig(self.output_dir / 'feature_importance.svg', bbox_inches='tight')
        
        return fig
    
    def plot_partial_dependence(self, X: np.ndarray, y: np.ndarray,
                                feature_names: List[str],
                                model, save: bool = True) -> plt.Figure:
        """Generate partial dependence plots for top features"""
        n_features = min(4, len(feature_names))
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
        
        for idx, ax in enumerate(axes[:n_features]):
            feature_idx = idx
            feature_name = feature_names[feature_idx]
            
            # Get unique values for the feature
            unique_vals = np.percentile(X[:, feature_idx], np.linspace(0, 100, 20))
            
            # Calculate partial dependence
            predictions = []
            for val in unique_vals:
                X_temp = X.copy()
                X_temp[:, feature_idx] = val
                pred = model.predict(X_temp).mean()
                predictions.append(pred)
            
            ax.plot(unique_vals, predictions, 'o-', linewidth=2, markersize=8, color='#2E86AB')
            ax.fill_between(unique_vals, 
                           np.array(predictions) - np.std(predictions),
                           np.array(predictions) + np.std(predictions),
                           alpha=0.2, color='#2E86AB')
            ax.set_xlabel(feature_name, fontsize=10)
            ax.set_ylabel('Predicted Response', fontsize=10)
            ax.set_title(f'Effect of {feature_name}', fontsize=11, fontweight='bold')
            ax.grid(True, alpha=0.3)
        
        # Hide empty subplots
        for idx in range(n_features, 4):
            axes[idx].set_visible(False)
        
        plt.suptitle('Partial Dependence Plots', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save:
            plt.savefig(self.output_dir / 'partial_dependence.png', dpi=150, bbox_inches='tight')
        
        return fig
    
    def plot_shap_summary(self, shap_values, X: np.ndarray,
                          feature_names: List[str],
                          save: bool = True) -> plt.Figure:
        """Generate SHAP summary plot"""
        try:
            import shap
            fig, ax = plt.subplots(figsize=(10, 8))
            shap.summary_plot(shap_values, X, feature_names=feature_names,
                            show=False, max_display=10)
            plt.tight_layout()
            
            if save:
                plt.savefig(self.output_dir / 'shap_summary.png', dpi=150, bbox_inches='tight')
            
            return fig
        except Exception as e:
            print(f"SHAP plot not generated: {e}")
            return None
    
    def plot_parallel_coordinates(self, df: pd.DataFrame,
                                   target_col: str,
                                   top_n: int = 20,
                                   save: bool = True) -> plt.Figure:
        """Generate parallel coordinates plot for top methods"""
        from pandas.plotting import parallel_coordinates
        
        # Select top N methods by target
        df_sorted = df.nlargest(top_n, target_col).copy()
        df_sorted['Method_Rank'] = range(1, len(df_sorted) + 1)
        
        # Normalize features for better visualization
        feature_cols = df_sorted.select_dtypes(include=[np.number]).columns
        feature_cols = [c for c in feature_cols if c != target_col and c != 'Method_Rank']
        
        df_norm = df_sorted[feature_cols].copy()
        df_norm = (df_norm - df_norm.min()) / (df_norm.max() - df_norm.min())
        df_norm['Method_Rank'] = df_sorted['Method_Rank']
        df_norm[target_col] = df_sorted[target_col]
        
        fig, ax = plt.subplots(figsize=(14, 6))
        parallel_coordinates(df_norm, 'Method_Rank', ax=ax, colormap='viridis', alpha=0.7)
        ax.set_title(f'Top {top_n} Methods - Parameter Comparison', fontsize=14, fontweight='bold')
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.tight_layout()
        
        if save:
            plt.savefig(self.output_dir / 'parallel_coordinates.png', dpi=150, bbox_inches='tight')
        
        return fig
    
    def plot_residuals(self, y_true: np.ndarray, y_pred: np.ndarray,
                       save: bool = True) -> plt.Figure:
        """Generate residual plot for model validation"""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Actual vs Predicted
        axes[0].scatter(y_true, y_pred, alpha=0.5, color='#2E86AB')
        axes[0].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()],
                    'r--', linewidth=2, label='Perfect Prediction')
        axes[0].set_xlabel('Actual Values', fontsize=12)
        axes[0].set_ylabel('Predicted Values', fontsize=12)
        axes[0].set_title('Actual vs Predicted', fontsize=12, fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Residuals
        residuals = y_true - y_pred
        axes[1].scatter(y_pred, residuals, alpha=0.5, color='#A23B72')
        axes[1].axhline(y=0, color='r', linestyle='--', linewidth=2)
        axes[1].set_xlabel('Predicted Values', fontsize=12)
        axes[1].set_ylabel('Residuals', fontsize=12)
        axes[1].set_title('Residual Plot', fontsize=12, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        
        plt.suptitle('Model Diagnostics', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save:
            plt.savefig(self.output_dir / 'residual_plot.png', dpi=150, bbox_inches='tight')
        
        return fig
    
    def plot_radar_chart(self, method_data: Dict[str, float],
                         method_name: str,
                         target_ranges: Dict[str, Tuple[float, float]],
                         save: bool = True) -> plt.Figure:
        """Generate radar chart comparing method against targets"""
        categories = list(method_data.keys())
        values = list(method_data.values())
        
        # Normalize to 0-1 scale based on targets
        normalized = []
        for cat, val in zip(categories, values):
            if cat in target_ranges:
                min_t, max_t = target_ranges[cat]
                norm = (val - min_t) / (max_t - min_t)
                normalized.append(max(0, min(1, norm)))
            else:
                normalized.append(val / max(values))
        
        # Close the loop
        categories_loop = categories + [categories[0]]
        values_loop = normalized + [normalized[0]]
        
        # Create radar chart
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        angles_loop = angles + [angles[0]]
        
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
        ax.plot(angles_loop, values_loop, 'o-', linewidth=2, color='#2E86AB')
        ax.fill(angles_loop, values_loop, alpha=0.25, color='#2E86AB')
        ax.set_xticks(angles)
        ax.set_xticklabels(categories, size=10)
        ax.set_ylim(0, 1)
        ax.set_yticks([0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels(['25%', '50%', '75%', '100%'], size=8)
        ax.set_title(f'Method Performance: {method_name}', size=14, fontweight='bold', pad=20)
        ax.grid(True)
        
        if save:
            plt.savefig(self.output_dir / f'radar_chart_{method_name}.png', dpi=150, bbox_inches='tight')
        
        return fig
    
    def generate_full_report(self, df: pd.DataFrame, 
                            feature_names: List[str],
                            importance_scores: List[float],
                            output_prefix: str = "elusight_analysis") -> Dict[str, str]:
        """Generate all visualizations automatically"""
        print("\n📊 Generating automatic visualizations...")
        
        generated_files = {}
        
        # 1. Correlation heatmap
        print("   ✓ Correlation heatmap")
        self.plot_correlation_heatmap(df)
        generated_files['correlation'] = str(self.output_dir / 'correlation_heatmap.png')
        
        # 2. Feature importance
        print("   ✓ Feature importance plot")
        self.plot_feature_importance(feature_names, importance_scores)
        generated_files['importance'] = str(self.output_dir / 'feature_importance.png')
        
        # 3. Radar chart for top method (if applicable)
        if 'trust_score' in df.columns:
            print("   ✓ Radar chart")
            top_method = df.nlargest(1, 'trust_score').iloc[0]
            method_data = {col: top_method[col] for col in feature_names if col in df.columns}
            target_ranges = {col: (df[col].min(), df[col].max()) for col in feature_names if col in df.columns}
            self.plot_radar_chart(method_data, top_method.get('method_id', 'Best'), target_ranges)
            generated_files['radar'] = str(self.output_dir / f'radar_chart_Best.png')
        
        print(f"\n✅ All visualizations saved to: {self.output_dir}/")
        return generated_files