#!/usr/bin/env python3
"""
Comprehensive experiment evaluation and results aggregation.

This script:
1. Discovers all trained models in checkpoints/ and checkpoints_server/
2. Evaluates each model using the evaluation pipeline
3. Aggregates results into structured CSV and markdown reports
4. Generates comprehensive summary for thesis writing

Usage:
    python scripts/evaluate_all_experiments.py --config config.yaml
    python scripts/evaluate_all_experiments.py --config config.yaml --models bert_text lstm_text
    python scripts/evaluate_all_experiments.py --help
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Evaluate all experiments and aggregate results',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to configuration file'
    )

    parser.add_argument(
        '--checkpoint-dirs',
        type=str,
        nargs='+',
        default=['checkpoints', 'checkpoints_server'],
        help='Directories containing model checkpoints'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default='evaluation_final',
        help='Directory to save aggregated evaluation results'
    )

    parser.add_argument(
        '--skip-evaluation',
        action='store_true',
        help='Skip evaluation and only aggregate existing results'
    )

    parser.add_argument(
        '--split',
        type=str,
        default='test',
        choices=['val', 'test'],
        help='Dataset split to evaluate on'
    )

    parser.add_argument(
        '--models',
        type=str,
        nargs='+',
        default=None,
        help='Specific models to evaluate (e.g., bert_text lstm_text). If not specified, all models are evaluated.'
    )

    parser.add_argument(
        '--use-best',
        action='store_true',
        default=True,
        help='Use best.pth checkpoints (default: True). If False, uses last.pth'
    )

    return parser.parse_args()


def discover_checkpoints(checkpoint_dirs: List[str], models: List[str] = None, use_best: bool = True) -> Dict[str, Dict]:
    """
    Discover all model checkpoints.

    Args:
        checkpoint_dirs: List of directories to search
        models: Optional list of specific model names to evaluate
        use_best: Whether to use best.pth (True) or last.pth (False)

    Returns:
        Dictionary mapping model_type -> checkpoint info
    """
    checkpoints = {}

    for checkpoint_dir in checkpoint_dirs:
        checkpoint_path = Path(checkpoint_dir)
        if not checkpoint_path.exists():
            print(f"Warning: Checkpoint directory {checkpoint_dir} does not exist")
            continue

        # Find all model directories
        for model_dir in checkpoint_path.iterdir():
            if not model_dir.is_dir():
                continue

            model_type = model_dir.name

            # Skip if specific models requested and this isn't one of them
            if models is not None and model_type not in models:
                continue

            # Look for best.pth (preferred) or last.pth
            preferred_checkpoint = model_dir / ('best.pth' if use_best else 'last.pth')
            fallback_checkpoint = model_dir / ('last.pth' if use_best else 'best.pth')

            if preferred_checkpoint.exists():
                checkpoint_file = preferred_checkpoint
            elif fallback_checkpoint.exists():
                checkpoint_file = fallback_checkpoint
                print(f"Note: Using {fallback_checkpoint.name} for {model_type} (preferred not found)")
            else:
                print(f"Warning: No checkpoint found in {model_dir}")
                continue

            # Only store one checkpoint per model type (prefer checkpoints over checkpoints_server)
            if model_type not in checkpoints or checkpoint_dir == 'checkpoints':
                checkpoints[model_type] = {
                    'path': checkpoint_file,
                    'source': checkpoint_dir,
                    'model_type': model_type
                }

    return checkpoints


def evaluate_model(checkpoint_path: Path, config: str, split: str, output_dir: Path) -> Tuple[bool, Path]:
    """
    Run evaluation script for a single model.

    Args:
        checkpoint_path: Path to model checkpoint
        config: Path to config file
        split: Dataset split (val/test)
        output_dir: Base output directory

    Returns:
        (success: bool, results_dir: Path)
    """
    cmd = [
        sys.executable,
        'scripts/evaluate.py',
        '--checkpoint', str(checkpoint_path),
        '--config', config,
        '--split', split,
        '--output-dir', str(output_dir)
    ]

    print(f"\n{'='*80}")
    print(f"Evaluating: {checkpoint_path}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*80}\n")

    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(result.stdout)

        # Find the results directory from output
        # The evaluate script creates: output_dir/model_type/split_timestamp/
        model_type = checkpoint_path.parent.name
        results_dirs = list((Path(output_dir) / model_type).glob(f"{split}_*"))
        if results_dirs:
            latest_results = max(results_dirs, key=lambda p: p.name)
            return True, latest_results
        else:
            print(f"Warning: Could not find results directory for {checkpoint_path}")
            return False, None

    except subprocess.CalledProcessError as e:
        print(f"Error evaluating {checkpoint_path}:")
        print(e.stderr)
        return False, None


def load_evaluation_results(results_dir: Path) -> Dict:
    """
    Load evaluation results from a results directory.

    Args:
        results_dir: Path to evaluation results directory

    Returns:
        Dictionary containing metrics and metadata
    """
    metrics_file = results_dir / 'metrics.json'

    if not metrics_file.exists():
        return None

    with open(metrics_file, 'r') as f:
        metrics = json.load(f)

    # Add results directory path
    metrics['results_dir'] = str(results_dir)

    return metrics


def aggregate_results(checkpoints: Dict[str, Dict], output_dir: Path, split: str) -> pd.DataFrame:
    """
    Aggregate all evaluation results into a summary table.

    Args:
        checkpoints: Dictionary of checkpoint info
        output_dir: Base output directory for evaluations
        split: Dataset split used for evaluation

    Returns:
        DataFrame with aggregated results
    """
    results = []

    # For each model type, find the most recent evaluation results
    for model_type, checkpoint_info in checkpoints.items():
        model_results_dir = Path(output_dir) / model_type

        if not model_results_dir.exists():
            print(f"Warning: No results found for {model_type}")
            continue

        # Find all result directories for this split
        result_dirs = list(model_results_dir.glob(f"{split}_*"))

        if not result_dirs:
            print(f"Warning: No {split} results found for {model_type}")
            continue

        # Load the most recent results
        latest_results = max(result_dirs, key=lambda p: p.name)
        metrics = load_evaluation_results(latest_results)

        if metrics is None:
            print(f"Warning: Could not load metrics for {model_type}")
            continue

        # Extract key metrics
        result_row = {
            'Model Type': model_type,
            'Source': checkpoint_info['source'],
            'Checkpoint': str(checkpoint_info['path']),
            'Split': split,
            'Threshold': metrics.get('threshold', 0.5),
            'F1-Macro': metrics.get('f1_macro', 0.0),
            'F1-Micro': metrics.get('f1_micro', 0.0),
            'F1-Weighted': metrics.get('f1_weighted', 0.0),
            'Precision-Macro': metrics.get('precision_macro', 0.0),
            'Recall-Macro': metrics.get('recall_macro', 0.0),
            'ROC-AUC-Macro': metrics.get('roc_auc_macro', 0.0),
            'Hamming Loss': metrics.get('hamming_loss', 0.0),
            'Subset Accuracy': metrics.get('subset_accuracy', 0.0),
            'Results Dir': metrics.get('results_dir', '')
        }

        results.append(result_row)

    # Create DataFrame
    df = pd.DataFrame(results)

    # Sort by F1-Macro descending
    if not df.empty:
        df = df.sort_values('F1-Macro', ascending=False)

    return df


def generate_summary_report(df: pd.DataFrame, output_path: Path):
    """
    Generate comprehensive markdown summary report.

    Args:
        df: DataFrame with aggregated results
        output_path: Path to save markdown file
    """
    md_lines = [
        "# Complete Experiment Results Summary",
        "",
        f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Total Models Evaluated**: {len(df)}",
        "",
        "---",
        "",
        "## Overall Performance Ranking",
        "",
        "Ranked by F1-Macro score on test set:",
        ""
    ]

    # Overall ranking table
    if not df.empty:
        ranking_df = df[['Model Type', 'F1-Macro', 'ROC-AUC-Macro', 'Precision-Macro', 'Recall-Macro', 'Threshold']].copy()
        ranking_df['Rank'] = range(1, len(ranking_df) + 1)
        ranking_df = ranking_df[['Rank', 'Model Type', 'F1-Macro', 'ROC-AUC-Macro', 'Precision-Macro', 'Recall-Macro', 'Threshold']]

        # Format percentages
        for col in ['F1-Macro', 'ROC-AUC-Macro', 'Precision-Macro', 'Recall-Macro']:
            ranking_df[col] = ranking_df[col].apply(lambda x: f"{x*100:.2f}%")

        md_lines.append(ranking_df.to_markdown(index=False))
        md_lines.append("")

    # Category comparisons
    md_lines.extend([
        "---",
        "",
        "## Model Category Comparison",
        ""
    ])

    # Group by model category
    if not df.empty:
        categories = {
            'Text-Only': df[df['Model Type'].str.contains('text', case=False)],
            'Vision-Only': df[df['Model Type'].str.contains('vision|cnn', case=False)],
            'Multimodal': df[df['Model Type'].str.contains('fusion', case=False)]
        }

        for category, cat_df in categories.items():
            if cat_df.empty:
                continue

            md_lines.append(f"### {category} Models")
            md_lines.append("")

            best_model = cat_df.iloc[0]
            md_lines.extend([
                f"**Best Model**: {best_model['Model Type']}",
                f"- F1-Macro: {best_model['F1-Macro']*100:.2f}%",
                f"- ROC-AUC: {best_model['ROC-AUC-Macro']*100:.2f}%",
                f"- Precision: {best_model['Precision-Macro']*100:.2f}%",
                f"- Recall: {best_model['Recall-Macro']*100:.2f}%",
                ""
            ])

            # All models in category
            if len(cat_df) > 1:
                cat_summary = cat_df[['Model Type', 'F1-Macro', 'ROC-AUC-Macro']].copy()
                cat_summary['F1-Macro'] = cat_summary['F1-Macro'].apply(lambda x: f"{x*100:.2f}%")
                cat_summary['ROC-AUC-Macro'] = cat_summary['ROC-AUC-Macro'].apply(lambda x: f"{x*100:.2f}%")
                md_lines.append(cat_summary.to_markdown(index=False))
            md_lines.append("")

    # Detailed per-model results
    md_lines.extend([
        "---",
        "",
        "## Detailed Model Results",
        ""
    ])

    for idx, row in df.iterrows():
        md_lines.extend([
            f"### {row['Model Type'].upper()}",
            "",
            f"**Source**: `{row['Source']}`",
            f"**Checkpoint**: `{row['Checkpoint']}`",
            "",
            "#### Metrics",
            "",
            f"- **F1-Macro**: {row['F1-Macro']*100:.2f}%",
            f"- **F1-Micro**: {row['F1-Micro']*100:.2f}%",
            f"- **F1-Weighted**: {row['F1-Weighted']*100:.2f}%",
            f"- **Precision-Macro**: {row['Precision-Macro']*100:.2f}%",
            f"- **Recall-Macro**: {row['Recall-Macro']*100:.2f}%",
            f"- **ROC-AUC-Macro**: {row['ROC-AUC-Macro']*100:.2f}%",
            f"- **Hamming Loss**: {row['Hamming Loss']:.4f}",
            f"- **Subset Accuracy**: {row['Subset Accuracy']*100:.2f}%",
            f"- **Threshold**: {row['Threshold']:.2f}",
            "",
            f"**Detailed Results**: `{row['Results Dir']}`",
            "",
            "---",
            ""
        ])

    # Key comparisons
    if not df.empty:
        md_lines.extend([
            "---",
            "",
            "## Key Comparisons for Thesis",
            "",
            "### Text-Only Models Comparison",
            ""
        ])

        text_models = df[df['Model Type'].str.contains('text', case=False)]
        if not text_models.empty:
            if len(text_models) > 1:
                best_text = text_models.iloc[0]
                worst_text = text_models.iloc[-1]
                improvement = (best_text['F1-Macro'] - worst_text['F1-Macro']) * 100
                md_lines.extend([
                    f"- **Best**: {best_text['Model Type']} ({best_text['F1-Macro']*100:.2f}%)",
                    f"- **Baseline**: {worst_text['Model Type']} ({worst_text['F1-Macro']*100:.2f}%)",
                    f"- **Improvement**: +{improvement:.2f} percentage points",
                    ""
                ])
            else:
                md_lines.append(f"- Only one text model: {text_models.iloc[0]['Model Type']} ({text_models.iloc[0]['F1-Macro']*100:.2f}%)")
                md_lines.append("")

        md_lines.extend([
            "### Vision-Only Models Comparison",
            ""
        ])

        vision_models = df[df['Model Type'].str.contains('vision|cnn', case=False)]
        if not vision_models.empty:
            if len(vision_models) > 1:
                best_vision = vision_models.iloc[0]
                worst_vision = vision_models.iloc[-1]
                improvement = (best_vision['F1-Macro'] - worst_vision['F1-Macro']) * 100
                md_lines.extend([
                    f"- **Best**: {best_vision['Model Type']} ({best_vision['F1-Macro']*100:.2f}%)",
                    f"- **Baseline**: {worst_vision['Model Type']} ({worst_vision['F1-Macro']*100:.2f}%)",
                    f"- **Improvement**: +{improvement:.2f} percentage points",
                    ""
                ])
            else:
                md_lines.append(f"- Only one vision model: {vision_models.iloc[0]['Model Type']} ({vision_models.iloc[0]['F1-Macro']*100:.2f}%)")
                md_lines.append("")

        md_lines.extend([
            "### Multimodal vs. Unimodal Comparison",
            ""
        ])

        fusion_models = df[df['Model Type'].str.contains('fusion', case=False)]
        if not fusion_models.empty and not text_models.empty:
            best_fusion = fusion_models.iloc[0]
            best_text = text_models.iloc[0]
            best_vision = vision_models.iloc[0] if not vision_models.empty else None

            best_unimodal_f1 = max(
                best_text['F1-Macro'],
                best_vision['F1-Macro'] if best_vision is not None else 0
            )
            best_unimodal_name = best_text['Model Type'] if best_text['F1-Macro'] >= (best_vision['F1-Macro'] if best_vision is not None else 0) else (best_vision['Model Type'] if best_vision is not None else 'N/A')

            improvement = (best_fusion['F1-Macro'] - best_unimodal_f1) * 100

            md_lines.extend([
                f"- **Best Multimodal**: {best_fusion['Model Type']} ({best_fusion['F1-Macro']*100:.2f}%)",
                f"- **Best Unimodal**: {best_unimodal_name} ({best_unimodal_f1*100:.2f}%)",
                f"- **Improvement**: {improvement:+.2f} percentage points",
                ""
            ])

            md_lines.extend([
                "### Fusion Strategy Comparison",
                ""
            ])

            if len(fusion_models) > 1:
                fusion_comparison = fusion_models[['Model Type', 'F1-Macro', 'ROC-AUC-Macro']].copy()
                fusion_comparison['F1-Macro'] = fusion_comparison['F1-Macro'].apply(lambda x: f"{x*100:.2f}%")
                fusion_comparison['ROC-AUC-Macro'] = fusion_comparison['ROC-AUC-Macro'].apply(lambda x: f"{x*100:.2f}%")
                md_lines.append(fusion_comparison.to_markdown(index=False))
                md_lines.append("")

    # Summary statistics
    if not df.empty:
        md_lines.extend([
            "---",
            "",
            "## Summary Statistics",
            "",
            f"- **Mean F1-Macro**: {df['F1-Macro'].mean()*100:.2f}%",
            f"- **Std F1-Macro**: {df['F1-Macro'].std()*100:.2f}%",
            f"- **Best F1-Macro**: {df['F1-Macro'].max()*100:.2f}% ({df.iloc[0]['Model Type']})",
            f"- **Mean ROC-AUC**: {df['ROC-AUC-Macro'].mean()*100:.2f}%",
            ""
        ])

    # Write to file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(md_lines))

    print(f"\nSummary report saved to: {output_path}")


def main():
    """Main execution function."""
    args = parse_args()

    print("\n" + "="*80)
    print("COMPREHENSIVE EXPERIMENT EVALUATION")
    print("="*80)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Discover checkpoints
    print("\n" + "="*80)
    print("STEP 1: DISCOVERING CHECKPOINTS")
    print("="*80)

    checkpoints = discover_checkpoints(args.checkpoint_dirs, args.models, args.use_best)

    print(f"\nFound {len(checkpoints)} model type(s):")
    for model_type, checkpoint_info in checkpoints.items():
        print(f"  - {model_type}:")
        print(f"    Path: {checkpoint_info['path']}")
        print(f"    Source: {checkpoint_info['source']}")

    if not checkpoints:
        print("No checkpoints found. Exiting.")
        return

    # Step 2: Evaluate models (unless skipped)
    if not args.skip_evaluation:
        print("\n" + "="*80)
        print("STEP 2: EVALUATING MODELS")
        print("="*80)

        evaluation_results = {}

        for model_type, checkpoint_info in checkpoints.items():
            checkpoint_path = checkpoint_info['path']

            success, results_dir = evaluate_model(
                checkpoint_path,
                args.config,
                args.split,
                output_dir
            )

            if success:
                evaluation_results[model_type] = results_dir
                print(f"[SUCCESS] Successfully evaluated {model_type}")
            else:
                print(f"[FAILED] Failed to evaluate {model_type}")

        print(f"\nSuccessfully evaluated {len(evaluation_results)}/{len(checkpoints)} models")
    else:
        print("\n" + "="*80)
        print("STEP 2: SKIPPING EVALUATION (--skip-evaluation flag set)")
        print("="*80)

    # Step 3: Aggregate results
    print("\n" + "="*80)
    print("STEP 3: AGGREGATING RESULTS")
    print("="*80)

    results_df = aggregate_results(checkpoints, output_dir, args.split)

    if results_df.empty:
        print("No results to aggregate. Exiting.")
        return

    print(f"\nAggregated results from {len(results_df)} models")
    print("\n" + "="*80)
    print("RESULTS PREVIEW")
    print("="*80)
    print("\nTop Models by F1-Macro:")
    preview_df = results_df[['Model Type', 'F1-Macro', 'Precision-Macro', 'Recall-Macro', 'ROC-AUC-Macro']].copy()
    preview_df['F1-Macro'] = preview_df['F1-Macro'].apply(lambda x: f"{x*100:.2f}%")
    preview_df['Precision-Macro'] = preview_df['Precision-Macro'].apply(lambda x: f"{x*100:.2f}%")
    preview_df['Recall-Macro'] = preview_df['Recall-Macro'].apply(lambda x: f"{x*100:.2f}%")
    preview_df['ROC-AUC-Macro'] = preview_df['ROC-AUC-Macro'].apply(lambda x: f"{x*100:.2f}%")
    print(preview_df.to_string(index=False))

    # Save aggregated results
    results_csv = output_dir / 'all_results.csv'
    results_df.to_csv(results_csv, index=False)
    print(f"\n[SAVED] Results CSV saved to: {results_csv}")

    # Generate summary report
    summary_md = output_dir / 'RESULTS_SUMMARY.md'
    generate_summary_report(results_df, summary_md)
    print(f"[SAVED] Summary report saved to: {summary_md}")

    print("\n" + "="*80)
    print("EVALUATION COMPLETE")
    print("="*80)
    print(f"\nAll results saved to: {output_dir}/")
    print(f"  - CSV data: {results_csv.name}")
    print(f"  - Summary report: {summary_md.name}")
    print(f"  - Individual evaluations: evaluation_final/<model_type>/{args.split}_*/")
    print("\nNext steps:")
    print("  1. Review the summary report and CSV data")
    print("  2. Manually update .docs/ files for thesis documentation")
    print("  3. Include visualizations from individual evaluation directories")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
