#!/usr/bin/env python3
"""
RF-Diffusion Dataset Condition Analysis Script

This script analyzes the condition parameters (cond) from WiFi and FMCW
inference datasets and generates a comprehensive Markdown report.
"""

import os
import scipy.io as scio
import numpy as np
from collections import defaultdict
from pathlib import Path


def load_conditions(dataset_dir):
    """
    Load all condition parameters from .mat files in a directory.
    
    Args:
        dataset_dir: Path to directory containing .mat files
        
    Returns:
        dict: {filename: condition_array}
    """
    conditions = {}
    
    if not os.path.exists(dataset_dir):
        print(f"Warning: Directory {dataset_dir} does not exist")
        return conditions
    
    mat_files = sorted([f for f in os.listdir(dataset_dir) if f.endswith('.mat')])
    
    for filename in mat_files:
        filepath = os.path.join(dataset_dir, filename)
        try:
            data = scio.loadmat(filepath)
            if 'cond' in data:
                conditions[filename] = data['cond'][0]  # cond is (1, 6), extract to 1D array
            else:
                print(f"Warning: 'cond' not found in {filename}")
        except Exception as e:
            print(f"Error loading {filename}: {e}")
    
    return conditions


def analyze_conditions(conditions):
    """
    Analyze condition parameters to find unique conditions and statistics.
    
    Args:
        conditions: dict of {filename: condition_array}
        
    Returns:
        dict: Analysis results including unique conditions and counts
    """
    if not conditions:
        return {
            'total_files': 0,
            'unique_conditions': [],
            'condition_counts': {},
            'condition_to_files': {}
        }
    
    # Convert condition arrays to tuples for hashability
    condition_tuples = [(filename, tuple(cond)) for filename, cond in conditions.items()]
    
    # Count occurrences of each unique condition
    condition_counts = defaultdict(int)
    condition_to_files = defaultdict(list)
    
    for filename, cond_tuple in condition_tuples:
        condition_counts[cond_tuple] += 1
        condition_to_files[cond_tuple].append(filename)
    
    # Get unique conditions sorted by first occurrence
    unique_conditions = list(dict.fromkeys([cond for _, cond in condition_tuples]))
    
    return {
        'total_files': len(conditions),
        'unique_conditions': unique_conditions,
        'condition_counts': dict(condition_counts),
        'condition_to_files': dict(condition_to_files)
    }


def generate_markdown_report(wifi_conditions, fmcw_conditions, output_file='dataset_conditions_report.md'):
    """
    Generate a Markdown report of condition analysis.
    
    Args:
        wifi_conditions: dict of WiFi conditions
        fmcw_conditions: dict of FMCW conditions
        output_file: Output markdown file path
    """
    # Analyze both datasets
    wifi_analysis = analyze_conditions(wifi_conditions)
    fmcw_analysis = analyze_conditions(fmcw_conditions)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("# RF-Diffusion Dataset Condition Analysis Report\n\n")
        f.write("This report analyzes the condition parameters from WiFi and FMCW inference datasets.\n\n")
        f.write("---\n\n")
        
        # WiFi Section
        f.write("## 1. WiFi Dataset\n\n")
        f.write(f"**Total Files**: {wifi_analysis['total_files']}\n\n")
        f.write(f"**Unique Condition Types**: {len(wifi_analysis['unique_conditions'])}\n\n")
        
        if wifi_analysis['total_files'] > 0:
            f.write("### 1.1 File List with Conditions\n\n")
            f.write("| # | Filename | Condition Parameters |\n")
            f.write("|---|----------|---------------------|\n")
            
            for idx, (filename, cond) in enumerate(sorted(wifi_conditions.items()), 1):
                cond_str = f"[{', '.join(map(str, cond))}]"
                f.write(f"| {idx} | `{filename}` | {cond_str} |\n")
            
            f.write("\n### 1.2 Unique Conditions Summary\n\n")
            f.write(f"Total unique condition patterns: **{len(wifi_analysis['unique_conditions'])}**\n\n")
            f.write("| Condition Pattern | Count | Example Files |\n")
            f.write("|-------------------|-------|---------------|\n")
            
            for cond_tuple in wifi_analysis['unique_conditions']:
                count = wifi_analysis['condition_counts'][cond_tuple]
                files = wifi_analysis['condition_to_files'][cond_tuple]
                cond_str = f"[{', '.join(map(str, cond_tuple))}]"
                example_files = ', '.join([f"`{f}`" for f in files[:2]])
                if len(files) > 2:
                    example_files += f", ... (+{len(files)-2} more)"
                f.write(f"| {cond_str} | {count} | {example_files} |\n")
        
        f.write("\n---\n\n")
        
        # FMCW Section
        f.write("## 2. FMCW Radar Dataset\n\n")
        f.write(f"**Total Files**: {fmcw_analysis['total_files']}\n\n")
        f.write(f"**Unique Condition Types**: {len(fmcw_analysis['unique_conditions'])}\n\n")
        
        if fmcw_analysis['total_files'] > 0:
            f.write("### 2.1 File List with Conditions\n\n")
            f.write("| # | Filename | Condition Parameters |\n")
            f.write("|---|----------|---------------------|\n")
            
            for idx, (filename, cond) in enumerate(sorted(fmcw_conditions.items()), 1):
                cond_str = f"[{', '.join(map(str, cond))}]"
                f.write(f"| {idx} | `{filename}` | {cond_str} |\n")
            
            f.write("\n### 2.2 Unique Conditions Summary\n\n")
            f.write(f"Total unique condition patterns: **{len(fmcw_analysis['unique_conditions'])}**\n\n")
            f.write("| Condition Pattern | Count | Example Files |\n")
            f.write("|-------------------|-------|---------------|\n")
            
            for cond_tuple in fmcw_analysis['unique_conditions']:
                count = fmcw_analysis['condition_counts'][cond_tuple]
                files = fmcw_analysis['condition_to_files'][cond_tuple]
                cond_str = f"[{', '.join(map(str, cond_tuple))}]"
                example_files = ', '.join([f"`{f}`" for f in files[:2]])
                if len(files) > 2:
                    example_files += f", ... (+{len(files)-2} more)"
                f.write(f"| {cond_str} | {count} | {example_files} |\n")
        
        f.write("\n---\n\n")
        
        # Overall Summary
        f.write("## 3. Overall Summary\n\n")
        f.write("| Dataset | Total Files | Unique Conditions |\n")
        f.write("|---------|-------------|------------------|\n")
        f.write(f"| WiFi | {wifi_analysis['total_files']} | {len(wifi_analysis['unique_conditions'])} |\n")
        f.write(f"| FMCW | {fmcw_analysis['total_files']} | {len(fmcw_analysis['unique_conditions'])} |\n")
        f.write(f"| **Total** | **{wifi_analysis['total_files'] + fmcw_analysis['total_files']}** | **{len(wifi_analysis['unique_conditions']) + len(fmcw_analysis['unique_conditions'])}** |\n")
        
        f.write("\n## 4. Condition Parameter Notes\n\n")
        f.write("The condition parameters (`cond`) are 6-element arrays that appear to encode:\n\n")
        f.write("Based on the filename patterns, the parameters likely represent:\n\n")
        f.write("### WiFi (filename format: `user{X}-1-1-5-{Y}-r5.mat`)\n")
        f.write("- May encode: user ID, environment, gesture type, location, orientation, etc.\n\n")
        f.write("### FMCW (filename format: `user-{X}-{Y}-{Z}-{W}.mat`)\n")
        f.write("- May encode: user ID, scenario, position, gesture, direction, etc.\n\n")
        f.write("**Note**: The exact meaning of each parameter would need to be confirmed by examining the dataset documentation or source code.\n")
    
    print(f"\n✓ Report generated: {output_file}")


def main():
    """Main execution function."""
    print("=" * 60)
    print("RF-Diffusion Dataset Condition Analysis")
    print("=" * 60)
    
    # Define dataset directories
    wifi_dir = "./dataset/wifi/cond"
    fmcw_dir = "./dataset/fmcw/cond"
    
    # Load conditions
    print("\n[1/3] Loading WiFi conditions...")
    wifi_conditions = load_conditions(wifi_dir)
    print(f"      Loaded {len(wifi_conditions)} WiFi files")
    
    print("\n[2/3] Loading FMCW conditions...")
    fmcw_conditions = load_conditions(fmcw_dir)
    print(f"      Loaded {len(fmcw_conditions)} FMCW files")
    
    # Generate report
    print("\n[3/3] Generating Markdown report...")
    generate_markdown_report(wifi_conditions, fmcw_conditions)
    
    print("\n" + "=" * 60)
    print("Analysis complete!")
    print("=" * 60)
    print(f"\nTotal files analyzed: {len(wifi_conditions) + len(fmcw_conditions)}")
    print(f"  - WiFi: {len(wifi_conditions)} files")
    print(f"  - FMCW: {len(fmcw_conditions)} files")
    print("\nReport saved to: dataset_conditions_report.md")


if __name__ == "__main__":
    main()
