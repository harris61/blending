# Mining Stockpile Blending Optimizer

A Streamlit web application that optimizes stockpile blending to meet tonnage and chemistry targets at minimum cost or distance.

## Features

- **CSV Upload**: Upload stockpile data with auto-detected chemistry columns
- **Flexible Tonnage Handling**: Continuous or discrete (integer multiples) mode
- **Chemistry Targeting**: Weighted priority system to minimize deviation from targets
- **Display Units**: Choose distance and currency units for display
- **Distance Optimization**: Minimize distance while meeting chemistry targets
- **Interactive UI**: Real-time results with cost breakdown and chemistry analysis

## Quick Start

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the app**:
   ```bash
   streamlit run app/main.py
   ```

3. **Open browser** to http://localhost:8501

## Usage

1. Upload a CSV file with stockpile data (or download the sample from sidebar)
2. Set target tonnage
3. Configure chemistry targets with weights (higher = higher priority)
4. Click "Run Optimization" to get results

## CSV Format

Required columns:
- `name`: Stockpile identifier
- `tonnage_available`: Available tonnage
- `distance`: Distance from destination (unit set in the app)
- `cost_per_ton`: Cost per ton
- `revenue_per_ton`: Revenue per ton

At least one of `distance`, `cost_per_ton`, or `revenue_per_ton` must be filled for each stockpile row.

Any additional columns are treated as chemistry data (element percentages).

Example:
```csv
name,tonnage_available,distance,cost_per_ton,revenue_per_ton,Fe,SiO2,Al2O3,P
Stockpile_A,5000,2.5,15.00,22.50,62.5,4.2,2.1,0.05
```

## Optimization Model

The optimizer minimizes:
```
total_cost = material_cost + M × chemistry_penalty
```

For distance mode, the optimizer minimizes:
```
distance_total = Σ(tonnage × distance) + M × chemistry_penalty
```

Where:
- `chemistry_penalty = Σ(weight[e] × |achieved[e] - target[e]|)`
- `M` is a large multiplier to prioritize chemistry compliance

## Tech Stack

- **Frontend**: Streamlit
- **Optimizer**: PuLP (CBC solver)
- **Data handling**: Pandas
