# Quantitative Analysis Results

## Overview
This analysis examines the human-AI alignment study data across four experimental groups (A, B, BP, C) to understand the impact of different AI confidence presentation methods on decision-making behavior.

## Generated Figures

### Fig1_Utility_Gain_Distribution.png
- **Analysis**: ANOVA comparing utility gain across groups
- **Result**: No significant difference (p=0.0502)
- **Interpretation**: All groups perform similarly in terms of decision accuracy improvement

### Fig2_Switching_Rate_Distribution.png
- **Analysis**: ANOVA comparing switching behavior across groups
- **Result**: Significant difference (p=0.0345)
- **Interpretation**: Groups differ in their tendency to change initial decisions after seeing AI recommendations

### Fig3_Silence_Cost_Effect.png
- **Analysis**: Paired t-test comparing two analysis logics for Group BP
- **Result**: Highly significant (p<0.001)
- **Interpretation**: Including "silence" cases (where AI confidence = 50%) significantly affects the analysis outcome

### Fig4_Confidence_Difference_Analysis.png
- **Analysis**: Relationship between AI-Human confidence difference and switching behavior
- **Focus**: Group BP only
- **Interpretation**: Shows how confidence alignment affects decision switching patterns

### Fig5_Summary_Statistics.png
- **Content**: Comprehensive summary table with group statistics and all statistical test results
- **Purpose**: Quick reference for all key findings

## Key Findings

1. **Utility Gain**: No significant differences between groups (p=0.0502)
2. **Switching Behavior**: Significant group differences exist (p=0.0345)
3. **Silence Cost**: Highly significant methodological effect (p<0.001)
4. **Education Impact**: No significant effect on utility gain (p=0.4546)

## Statistical Methods Used
- One-way ANOVA for group comparisons
- Paired t-test for silence cost analysis
- Independent t-test for education effect
- Tukey HSD for post-hoc comparisons (when applicable)

## Data Quality
- Total participants: 401
- Total trials: 9,624 (after filtering attention tests)
- No missing data in key variables after preprocessing

## Files
- `quantitative_analysis.py`: Complete analysis script
- `Fig1-Fig5`: Individual analysis plots
- Raw data in `study_data/` directory