"""
Compare accuracy results between OLMo-2-0425-1B and Qwen2.5-0.5B models on MMLU benchmark
"""

import matplotlib.pyplot as plt
import numpy as np

# Data from timelog.txt
subjects = [
    "abstract_algebra",
    "anatomy", 
    "astronomy",
    "business_ethics",
    "clinical_knowledge",
    "college_biology",
    "college_chemistry",
    "college_computer_science",
    "college_mathematics",
    "college_medicine"
]

# OLMo-2-0425-1B results (from timelog Step 4)
olmo_results = {
    "anatomy": 44.44,
    "college_biology": 38.89,
    "astronomy": 38.82,
    "business_ethics": 38.00,
    "clinical_knowledge": 33.21,
    "college_chemistry": 33.00,
    "abstract_algebra": 32.00,
    "college_medicine": 30.64,
    "college_mathematics": 29.00,
    "college_computer_science": 28.00
}
olmo_overall = 34.77

# Qwen2.5-0.5B results (from timelog Step 6)
qwen_results = {
    "clinical_knowledge": 52.08,
    "astronomy": 50.00,
    "business_ethics": 48.00,
    "anatomy": 45.19,
    "college_medicine": 45.09,
    "college_computer_science": 38.00,
    "college_biology": 37.50,
    "abstract_algebra": 36.00,
    "college_chemistry": 32.00,
    "college_mathematics": 25.00
}
qwen_overall = 42.80

# Sort subjects alphabetically for consistent display
subjects_sorted = sorted(subjects)

# Get accuracy values in sorted order
olmo_acc = [olmo_results.get(s, 0) for s in subjects_sorted]
qwen_acc = [qwen_results.get(s, 0) for s in subjects_sorted]

# Create figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

# Color scheme
olmo_color = '#E74C3C'  # Red
qwen_color = '#3498DB'  # Blue

# ==================== Plot 1: Grouped Bar Chart ====================
x = np.arange(len(subjects_sorted))
width = 0.35

bars1 = ax1.bar(x - width/2, olmo_acc, width, label=f'OLMo-2-0425-1B ({olmo_overall:.2f}%)', 
                color=olmo_color, edgecolor='white', linewidth=0.7)
bars2 = ax1.bar(x + width/2, qwen_acc, width, label=f'Qwen2.5-0.5B ({qwen_overall:.2f}%)', 
                color=qwen_color, edgecolor='white', linewidth=0.7)

# Customize plot 1
ax1.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
ax1.set_xlabel('MMLU Subject', fontsize=12, fontweight='bold')
ax1.set_title('MMLU Accuracy Comparison by Subject', fontsize=14, fontweight='bold', pad=15)
ax1.set_xticks(x)
ax1.set_xticklabels([s.replace('_', '\n') for s in subjects_sorted], rotation=45, ha='right', fontsize=9)
ax1.legend(loc='upper right', fontsize=10)
ax1.set_ylim(0, 60)
ax1.axhline(y=25, color='gray', linestyle='--', alpha=0.5, label='Random guess (25%)')
ax1.grid(axis='y', alpha=0.3)

# Add value labels on bars
for bar in bars1:
    height = bar.get_height()
    ax1.annotate(f'{height:.1f}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3), textcoords="offset points",
                ha='center', va='bottom', fontsize=7, color=olmo_color)

for bar in bars2:
    height = bar.get_height()
    ax1.annotate(f'{height:.1f}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3), textcoords="offset points",
                ha='center', va='bottom', fontsize=7, color=qwen_color)

# ==================== Plot 2: Overall Comparison ====================
models = ['OLMo-2-0425-1B', 'Qwen2.5-0.5B']
overall_acc = [olmo_overall, qwen_overall]
colors = [olmo_color, qwen_color]

bars = ax2.bar(models, overall_acc, color=colors, edgecolor='white', linewidth=2, width=0.5)

# Customize plot 2
ax2.set_ylabel('Overall Accuracy (%)', fontsize=12, fontweight='bold')
ax2.set_title('Overall MMLU Accuracy', fontsize=14, fontweight='bold', pad=15)
ax2.set_ylim(0, 60)
ax2.axhline(y=25, color='gray', linestyle='--', alpha=0.5)
ax2.grid(axis='y', alpha=0.3)

# Add value labels
for bar, acc in zip(bars, overall_acc):
    height = bar.get_height()
    ax2.annotate(f'{acc:.2f}%',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 5), textcoords="offset points",
                ha='center', va='bottom', fontsize=14, fontweight='bold')

# Add improvement annotation
improvement = qwen_overall - olmo_overall
ax2.annotate(f'+{improvement:.2f}%\nimprovement',
            xy=(1, qwen_overall), xytext=(1.3, qwen_overall - 5),
            fontsize=11, color='green', fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='green', lw=1.5))

# Add text box with summary
textstr = f'Qwen2.5-0.5B outperforms\nOLMo-2-0425-1B by\n{improvement:.2f} percentage points'
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
ax2.text(0.5, 0.15, textstr, transform=ax2.transAxes, fontsize=10,
        verticalalignment='top', horizontalalignment='center', bbox=props)

plt.tight_layout()
plt.savefig('model_comparison.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.savefig('model_comparison.pdf', bbox_inches='tight', facecolor='white')
print("✓ Saved comparison charts to:")
print("  - model_comparison.png")
print("  - model_comparison.pdf")

# Show the plot
plt.show()

# Print summary table
print("\n" + "="*70)
print("ACCURACY COMPARISON SUMMARY")
print("="*70)
print(f"{'Subject':<25} {'OLMo-2-1B':>12} {'Qwen2.5-0.5B':>14} {'Difference':>12}")
print("-"*70)
for subject in subjects_sorted:
    olmo = olmo_results.get(subject, 0)
    qwen = qwen_results.get(subject, 0)
    diff = qwen - olmo
    diff_str = f"+{diff:.2f}" if diff > 0 else f"{diff:.2f}"
    print(f"{subject:<25} {olmo:>11.2f}% {qwen:>13.2f}% {diff_str:>11}%")
print("-"*70)
print(f"{'OVERALL':<25} {olmo_overall:>11.2f}% {qwen_overall:>13.2f}% +{improvement:.2f}%")
print("="*70)
