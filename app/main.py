"""Streamlit app for the Mining Stockpile Blending Optimizer."""

import streamlit as st
import pandas as pd
from dataclasses import dataclass, field

from models import (
    Stockpile,
    OptimizationRequest,
    ChemistryTarget,
)
from optimizer import optimize_blend

# Standard column names and their aliases (case-insensitive matching)
COLUMN_ALIASES = {
    "name": ["name", "stockpile", "stockpile_name", "pile", "pile_name", "id"],
    "tonnage_available": ["tonnage_available", "tonnage", "tons", "available", "qty", "quantity", "weight"],
    "distance_km": ["distance_km", "distance", "dist", "haul_distance"],
    "cost_per_ton": ["cost_per_ton", "cost", "price", "cost_per_t", "unit_cost", "price_per_ton"],
    "revenue_per_ton": ["revenue_per_ton", "revenue", "rev", "selling_price", "sale_price"],
}

STANDARD_COLUMNS = set(COLUMN_ALIASES.keys())
REQUIRED_COLUMNS = {
    "name",
    "tonnage_available",
    "distance_km",
    "cost_per_ton",
    "revenue_per_ton",
}
VALUE_COLUMNS = {"distance_km", "cost_per_ton", "revenue_per_ton"}


@dataclass
class ValidationResult:
    """Result of CSV validation."""
    is_valid: bool = False
    df: pd.DataFrame = None
    column_mapping: dict = field(default_factory=dict)
    chemistry_columns: list = field(default_factory=list)
    errors: list = field(default_factory=list)
    warnings: list = field(default_factory=list)


def normalize_column_name(col: str) -> str | None:
    """Try to match a column name to a standard column."""
    col_lower = col.strip().lower().replace(" ", "_").replace("-", "_")
    for standard, aliases in COLUMN_ALIASES.items():
        if col_lower in [a.lower() for a in aliases]:
            return standard
    return None


def validate_csv(uploaded_file) -> ValidationResult:
    """Validate uploaded CSV and return detailed validation result."""
    result = ValidationResult()

    # Try to read CSV
    try:
        df = pd.read_csv(uploaded_file)
    except Exception as e:
        result.errors.append(f"Failed to read CSV: {str(e)}")
        return result

    if df.empty:
        result.errors.append("CSV file is empty")
        return result

    # Map columns to standard names
    column_mapping = {}
    unmapped_columns = []

    for col in df.columns:
        standard = normalize_column_name(col)
        if standard:
            if standard in column_mapping.values():
                result.warnings.append(f"Multiple columns map to '{standard}': using '{col}'")
            column_mapping[col] = standard
        else:
            unmapped_columns.append(col)

    # Check required columns
    mapped_standards = set(column_mapping.values())

    missing_required = REQUIRED_COLUMNS - mapped_standards
    if missing_required:
        display_missing = [col.replace("distance_km", "distance") for col in missing_required]
        result.errors.append(f"Missing required columns: {', '.join(display_missing)}")
        result.errors.append(
            "  Expected: 'name', 'tonnage_available', 'distance', 'cost_per_ton', 'revenue_per_ton' (or aliases)"
        )

    # Rename columns to standard names
    df_normalized = df.rename(columns=column_mapping)
    result.df = df_normalized
    result.column_mapping = column_mapping

    # Chemistry columns are unmapped columns
    result.chemistry_columns = unmapped_columns

    if result.errors:
        return result

    # Validate data types and values
    row_errors = []

    for idx, row in df_normalized.iterrows():
        row_num = idx + 2  # +2 for header row and 0-index

        # Validate name
        if pd.isna(row.get("name")) or str(row.get("name")).strip() == "":
            row_errors.append(f"Row {row_num}: 'name' is empty")

        # Validate tonnage (must be positive integer)
        tonnage = row.get("tonnage_available")
        if pd.isna(tonnage):
            row_errors.append(f"Row {row_num}: 'tonnage_available' is empty")
        else:
            try:
                t = float(tonnage)
                if t <= 0:
                    row_errors.append(f"Row {row_num}: 'tonnage_available' must be positive (got {t})")
                elif t != int(t):
                    row_errors.append(f"Row {row_num}: 'tonnage_available' must be a whole number (got {t})")
            except (ValueError, TypeError):
                row_errors.append(f"Row {row_num}: 'tonnage_available' is not a valid number")

        # Validate value columns (at least one required per row)
        distance = row.get("distance_km")
        cost = row.get("cost_per_ton")
        revenue = row.get("revenue_per_ton")

        has_distance = "distance_km" in df_normalized.columns and pd.notna(distance)
        has_cost = "cost_per_ton" in df_normalized.columns and pd.notna(cost)
        has_revenue = "revenue_per_ton" in df_normalized.columns and pd.notna(revenue)

        if not has_distance and not has_cost and not has_revenue:
            row_errors.append(f"Row {row_num}: at least one of 'distance', 'cost_per_ton', or 'revenue_per_ton' must be filled")
        else:
            if has_distance:
                try:
                    d = float(distance)
                    if d < 0:
                        row_errors.append(f"Row {row_num}: 'distance' cannot be negative")
                except (ValueError, TypeError):
                    row_errors.append(f"Row {row_num}: 'distance' is not a valid number")

            if has_cost:
                try:
                    c = float(cost)
                    if c < 0:
                        row_errors.append(f"Row {row_num}: 'cost_per_ton' cannot be negative")
                except (ValueError, TypeError):
                    row_errors.append(f"Row {row_num}: 'cost_per_ton' is not a valid number")

            if has_revenue:
                try:
                    r = float(revenue)
                    if r < 0:
                        row_errors.append(f"Row {row_num}: 'revenue_per_ton' cannot be negative")
                except (ValueError, TypeError):
                    row_errors.append(f"Row {row_num}: 'revenue_per_ton' is not a valid number")

        # Validate chemistry columns (should be numeric)
        for chem_col in result.chemistry_columns:
            val = row.get(chem_col)
            if pd.notna(val):
                try:
                    float(val)
                except (ValueError, TypeError):
                    row_errors.append(f"Row {row_num}: '{chem_col}' is not a valid number")

    if row_errors:
        result.errors.extend(row_errors[:10])  # Limit to first 10 errors
        if len(row_errors) > 10:
            result.errors.append(f"... and {len(row_errors) - 10} more errors")
        return result

    # Add info warnings
    if unmapped_columns:
        result.warnings.append(f"Detected chemistry columns: {', '.join(unmapped_columns)}")

    result.is_valid = True
    return result


def parse_validated_csv(validation: ValidationResult) -> tuple[list[Stockpile], list[str], int]:
    """Parse a validated CSV into stockpiles."""
    df = validation.df
    chemistry_columns = validation.chemistry_columns

    stockpiles = []
    for _, row in df.iterrows():
        distance = None
        cost = None
        revenue = None

        if "distance_km" in df.columns and pd.notna(row.get("distance_km")):
            distance = float(row["distance_km"])
        if "cost_per_ton" in df.columns and pd.notna(row.get("cost_per_ton")):
            cost = float(row["cost_per_ton"])
        if "revenue_per_ton" in df.columns and pd.notna(row.get("revenue_per_ton")):
            revenue = float(row["revenue_per_ton"])

        chemistry = {col: float(row[col]) for col in chemistry_columns if pd.notna(row.get(col))}

        stockpile = Stockpile(
            name=str(row["name"]).strip(),
            tonnage_available=int(row["tonnage_available"]),
            distance_km=distance,
            cost_per_ton=cost,
            revenue_per_ton=revenue,
            chemistry=chemistry,
        )
        stockpiles.append(stockpile)

    total_available = sum(s.tonnage_available for s in stockpiles)
    return stockpiles, chemistry_columns, total_available


def main():
    st.set_page_config(
        page_title="Stockpile Blending Optimizer",
        page_icon="⛏️",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # CSS to make sidebar fixed and non-collapsible
    st.markdown("""
        <style>
            [data-testid="collapsedControl"] { display: none; }
            [data-testid="stSidebarCollapseButton"] { display: none; }
            section[data-testid="stSidebar"] { width: 300px !important; }
            .metric-card {
                border: 1px solid #e6e6e6;
                border-radius: 10px;
                padding: 12px 14px;
                background: #ffffff;
            }
            .metric-card.selected {
                border: 2px solid #1f77b4;
                box-shadow: 0 0 0 2px rgba(31, 119, 180, 0.15);
            }
            .metric-label {
                font-size: 0.85rem;
                color: #666666;
                margin-bottom: 4px;
            }
            .metric-value {
                font-size: 1.4rem;
                font-weight: 600;
                color: #111111;
            }
            .metric-delta {
                font-size: 0.85rem;
                color: #666666;
                margin-top: 4px;
            }
        </style>
    """, unsafe_allow_html=True)

    st.title("⛏️ Mining Stockpile Blending Optimizer")
    st.markdown("Optimize stockpile blending to meet tonnage and chemistry targets at the most optimum profit, cost, or distance.")

    # Initialize session state
    if "stockpiles" not in st.session_state:
        st.session_state.stockpiles = []
    if "chemistry_columns" not in st.session_state:
        st.session_state.chemistry_columns = []
    if "total_available" not in st.session_state:
        st.session_state.total_available = 0
    if "validation" not in st.session_state:
        st.session_state.validation = None

    # Sidebar for file upload and configuration
    with st.sidebar:
        st.header("📁 Data Upload")

        # File upload
        uploaded_file = st.file_uploader(
            "Upload Stockpile CSV",
            type=["csv"],
            help="Required columns: name, tonnage_available, distance, cost_per_ton, revenue_per_ton (at least one of the last three must be filled per row)",
        )

        if uploaded_file is not None:
            validation = validate_csv(uploaded_file)
            st.session_state.validation = validation

            if validation.is_valid:
                stockpiles, chemistry_columns, total_available = parse_validated_csv(validation)
                st.session_state.stockpiles = stockpiles
                st.session_state.chemistry_columns = chemistry_columns
                st.session_state.total_available = total_available
                st.success(f"✅ Valid! {len(stockpiles)} stockpiles ({total_available:,.0f}t)")
            else:
                st.session_state.stockpiles = []
                st.session_state.chemistry_columns = []
                st.session_state.total_available = 0
                st.error("❌ Validation failed")

        st.header("🎛️ Display Settings")
        distance_unit = st.text_input(
            "Distance Unit",
            value="",
            placeholder="e.g., mi, m",
            help="Used only for display",
        )
        currency_unit = st.text_input(
            "Currency",
            value="",
            placeholder="e.g., $, USD",
            help="Used only for display",
        )

    # Main content
    if not st.session_state.stockpiles:
        # Check if there's a failed validation to show
        validation = st.session_state.get("validation")

        if validation and not validation.is_valid:
            st.header("❌ CSV Validation Failed")

            # Show column mapping if available
            if validation.column_mapping:
                st.subheader("Column Mapping")
                mapping_data = [{"Your Column": k, "Mapped To": v} for k, v in validation.column_mapping.items()]
                st.dataframe(pd.DataFrame(mapping_data), use_container_width=True, hide_index=True)

            # Show errors
            if validation.errors:
                st.subheader("Errors")
                for error in validation.errors:
                    st.error(error)

            # Show warnings
            if validation.warnings:
                st.subheader("Warnings")
                for warning in validation.warnings:
                    st.warning(warning)

            st.markdown("---")
            st.markdown("**Please fix the errors above and re-upload your CSV.**")

        else:
            st.info("👈 Upload a CSV file to get started")

        # User Guide
        st.header("📖 User Guide")

        st.subheader("Step 1: Prepare Your CSV File")
        st.markdown("""
        Your CSV file must contain the following columns:

        | Column | Required | Description |
        |--------|----------|-------------|
        | `name` | Yes | Stockpile identifier |
        | `tonnage_available` | Yes | Available tonnage (must be whole number) |
        | `distance` | Yes | Distance from destination |
        | `cost_per_ton` | Yes | Cost per ton |
        | `revenue_per_ton` | Yes | Revenue per ton |
        | *Chemistry columns* | Optional | Any additional columns (e.g., Ni, Fe, SiO2) |
        """)

        st.info("📝 **Note:** At least one of `distance`, `cost_per_ton`, or `revenue_per_ton` must be provided for each stockpile.")

        st.markdown("**Example CSV format:**")
        example_df = pd.DataFrame({
            "name": ["Stockpile_A", "Stockpile_B", "Stockpile_C"],
            "tonnage_available": [5000, 8000, 3000],
            "distance": [2.5, 4.0, 1.5],
            "cost_per_ton": [3.50, 4.20, 4.80],
            "revenue_per_ton": [12.00, 10.50, 14.00],
            "Ni": [1.82, 1.65, 1.95],
            "Fe": [18.5, 22.0, 16.8],
            "Co": [0.08, 0.06, 0.09],
            "MgO": [22.5, 18.0, 25.0],
            "SiO2": [38.0, 42.0, 35.0],
            "Basicity": [0.59, 0.43, 0.71],
        })
        st.dataframe(example_df, use_container_width=True, hide_index=True)

        st.subheader("Step 2: Upload and Validate")
        st.markdown("""
        - Use the sidebar to upload your CSV file
        - The system will automatically validate your data
        - If there are errors, they will be displayed with specific row numbers
        - Fix any errors and re-upload
        """)

        st.subheader("Step 3: Configure Optimization")
        st.markdown("""
        After successful upload, configure the optimization parameters:
        - **Target Tonnage**: Total tonnage you want to produce (wmt)
        - **Minimum Blending Unit**: Tonnages will be multiples of this value (default: 100 wmt)
        - **Optimization Mode**:
          - *Minimize Distance*: Optimize based on distance (unit set in display settings)
          - *Minimize Cost*: Optimize based on cost per ton
          - *Maximize Profit*: Maximize revenue minus cost
        """)

        st.subheader("Step 4: Set Chemistry Targets")
        st.markdown("""
        For each chemistry element:
        - **Enable**: Check the box to include this element as a target
        - **Target %**: Set the desired percentage
        - **Mode**:
          - *Exact*: The optimizer will prioritize hitting this target precisely
          - *Approximate*: The optimizer will try its best but may deviate to optimize cost/profit
        """)

        st.subheader("Step 5: Run Optimization")
        st.markdown("""
        Click "Run Optimization" to find the optimal blend that:
        - Meets your target tonnage exactly
        - Optimizes based on selected mode (minimize cost, distance, or maximize profit)
        - Achieves chemistry targets based on their mode (Exact/Approximate)
        """)

        st.subheader("Understanding Results")
        st.markdown("""
        The results show:
        - **Summary**: Total tonnage, costs, revenue, profit, and distance
        - **Selected Stockpiles**: Which stockpiles to use and how much from each
        - **Achieved Chemistry**: Target vs. achieved with status (✓ = exact match, ~ = approximate)
        - **Recommendations**: Suggestions if exact targets couldn't be met precisely
        """)
        return

    def format_currency(value: float, include_unit: bool = True) -> str:
        formatted = f"{value:,.0f}"
        if include_unit and currency_unit:
            if currency_unit.isalpha():
                return f"{currency_unit} {formatted}"
            return f"{currency_unit}{formatted}"
        return formatted

    def render_metric_card(label: str, value: str, delta: str | None = None, selected: bool = False) -> None:
        delta_html = f"<div class=\"metric-delta\">{delta}</div>" if delta else ""
        class_name = "metric-card selected" if selected else "metric-card"
        st.markdown(
            f"""
            <div class="{class_name}">
                <div class="metric-label">{label}</div>
                <div class="metric-value">{value}</div>
                {delta_html}
            </div>
            """,
            unsafe_allow_html=True,
        )

    distance_label = "Distance"
    if distance_unit:
        distance_label = f"Distance ({distance_unit})"
    cost_per_ton_label = "Cost (/t)"
    revenue_per_ton_label = "Revenue (/t)"
    if currency_unit:
        cost_per_ton_label = f"Cost ({currency_unit}/t)"
        revenue_per_ton_label = f"Revenue ({currency_unit}/t)"

    # Display loaded stockpiles
    st.header("📊 Loaded Stockpiles")

    # Convert to DataFrame for display
    stockpile_data = []
    for sp in st.session_state.stockpiles:
        revenue = getattr(sp, 'revenue_per_ton', None)
        row = {
            "Name": sp.name,
            "Tonnage": f"{sp.tonnage_available:,}",
            distance_label: sp.distance_km if sp.distance_km is not None else "-",
            cost_per_ton_label: format_currency(sp.cost_per_ton, include_unit=False) if sp.cost_per_ton is not None else "-",
            revenue_per_ton_label: format_currency(revenue, include_unit=False) if revenue is not None else "-",
        }
        for col in st.session_state.chemistry_columns:
            row[f"{col}"] = f"{sp.chemistry.get(col, 0):.2f}"
        stockpile_data.append(row)

    st.dataframe(pd.DataFrame(stockpile_data), use_container_width=True, hide_index=True)

    # Configuration section
    st.header("⚙️ Optimization Settings")

    col1, col2 = st.columns(2)

    with col1:
        target_tonnage = st.number_input(
            "Target Tonnage (wmt)",
            min_value=1,
            max_value=st.session_state.total_available,
            value=min(5000, st.session_state.total_available),
            step=1,
            format="%d",
            help=f"Maximum available: {st.session_state.total_available:,} wmt",
        )

        min_increment = st.number_input(
            "Minimum Blending Unit (wmt)",
            min_value=1,
            value=100,
            step=1,
            format="%d",
            help="Tonnages will be constrained to multiples of this value",
        )

    with col2:
        optimization_mode = st.radio(
            "Optimization Mode",
            options=["distance", "material", "profit"],
            format_func=lambda x: {
                "distance": "Minimize Distance",
                "material": "Minimize Cost",
                "profit": "Maximize Profit"
            }[x],
            help="Distance: minimize distance. Material: minimize cost/ton. Profit: maximize revenue - cost.",
        )

    # Chemistry targets
    st.subheader("🧪 Chemistry Targets")
    st.markdown("Select elements to target and set their target values")

    chemistry_targets = {}

    if st.session_state.chemistry_columns:
        for element in st.session_state.chemistry_columns:
            col1, col2, col3, col4 = st.columns([1, 1.5, 2, 1.5])

            with col1:
                enabled = st.checkbox(
                    element,
                    key=f"enable_{element}",
                    value=False,
                )

            with col2:
                operator = st.selectbox(
                    f"Operator for {element}",
                    options=["=", "<", ">", "<=", ">=", "range"],
                    key=f"operator_{element}",
                    label_visibility="collapsed",
                    disabled=not enabled,
                )

            with col3:
                if operator == "range":
                    sub1, sub2 = st.columns(2)
                    with sub1:
                        target = st.number_input(
                            f"Min for {element}",
                            min_value=0.0,
                            max_value=100.0,
                            value=0.0,
                            step=0.01,
                            key=f"target_{element}",
                            placeholder="Min",
                            disabled=not enabled,
                        )
                    with sub2:
                        target_max = st.number_input(
                            f"Max for {element}",
                            min_value=0.0,
                            max_value=100.0,
                            value=0.0,
                            step=0.01,
                            key=f"target_max_{element}",
                            placeholder="Max",
                            disabled=not enabled,
                        )
                else:
                    target = st.number_input(
                        f"Target for {element}",
                        min_value=0.0,
                        max_value=100.0,
                        value=0.0,
                        step=0.01,
                        key=f"target_{element}",
                        label_visibility="collapsed",
                        disabled=not enabled,
                    )
                    target_max = None

            with col4:
                mode = st.selectbox(
                    f"Mode for {element}",
                    options=["approximate", "exact"],
                    format_func=lambda x: "Approximate" if x == "approximate" else "Exact",
                    key=f"mode_{element}",
                    label_visibility="collapsed",
                    disabled=not enabled,
                )

            if enabled and target > 0:
                chemistry_targets[element] = ChemistryTarget(
                    operator=operator,
                    target=target,
                    target_max=target_max if operator == "range" else None,
                    mode=mode,
                    weight=1.0,
                )
    else:
        st.info("No chemistry columns detected in CSV")

    # Run optimization
    st.markdown("---")

    if st.button("🚀 Run Optimization", type="primary", use_container_width=True):
        if target_tonnage <= 0:
            st.error("Please enter a valid target tonnage")
            return

        # Build request
        request = OptimizationRequest(
            stockpiles=st.session_state.stockpiles,
            target_tonnage=target_tonnage,
            chemistry_targets=chemistry_targets,
            optimization_mode=optimization_mode,
            min_increment=min_increment,
        )

        with st.spinner("Optimizing blend..."):
            result = optimize_blend(request)

        if not result.success:
            st.error(f"Optimization failed: {result.message}")
            return

        st.success(result.message)

        # Results section
        st.header("📈 Optimization Results")

        # Summary cards
        cost = result.cost_breakdown
        if result.achieved_chemistry:
            chemistry_value = ", ".join(
                f"{chem.achieved:.2f} {chem.element}" for chem in result.achieved_chemistry
            )
        else:
            chemistry_value = "n/a"
        distance_value = f"{cost.distance_avg_km:,.2f}" + (f" {distance_unit}" if distance_unit else "")

        row1 = st.columns(3)
        row2 = st.columns(3)

        with row1[0]:
            render_metric_card(
                "Total Tonnage",
                f"{result.total_tonnage:,} wmt",
            )
        with row1[1]:
            render_metric_card(
                "Chemistry",
                chemistry_value,
            )
        with row1[2]:
            render_metric_card(
                "Avg Distance",
                distance_value,
                selected=optimization_mode == "distance",
            )

        with row2[0]:
            render_metric_card(
                "Revenue",
                format_currency(cost.revenue_total),
                delta=f"{format_currency(cost.revenue_per_ton)}/t",
            )
        with row2[1]:
            render_metric_card(
                "Cost",
                format_currency(cost.cost_total),
                delta=f"{format_currency(cost.cost_per_ton)}/t",
                selected=optimization_mode == "material",
            )
        with row2[2]:
            render_metric_card(
                "Profit",
                format_currency(cost.profit_total),
                delta=f"{format_currency(cost.profit_per_ton)}/t",
                selected=optimization_mode == "profit",
            )

        # Selected stockpiles table
        st.subheader("Selected Stockpiles")

        selected_data = []
        selected_stockpile_map = {sp.name: sp for sp in st.session_state.stockpiles}
        target_elements = list(chemistry_targets.keys())
        cost_header = "Cost"
        revenue_header = "Revenue"
        profit_header = "Profit"
        if currency_unit:
            cost_header = f"Cost ({currency_unit})"
            revenue_header = f"Revenue ({currency_unit})"
            profit_header = f"Profit ({currency_unit})"

        for sp in result.selected_stockpiles:
            row = {
                "Stockpile": sp.name,
                "Tonnage Taken (wmt)": f"{sp.tonnage_taken:,}",
                "Tonnage Available (wmt)": f"{sp.tonnage_available:,}",
            }
            source_stockpile = selected_stockpile_map.get(sp.name)
            for element in target_elements:
                value = None
                if source_stockpile:
                    value = source_stockpile.chemistry.get(element)
                row[element] = f"{value:.2f}" if value is not None else "-"

            row[distance_label] = f"{sp.distance_km:.2f}" if sp.distance_km is not None else "-"
            row[cost_header] = format_currency(sp.material_cost, include_unit=False)
            row[revenue_header] = format_currency(sp.revenue, include_unit=False)
            row[profit_header] = format_currency(sp.profit, include_unit=False)
            selected_data.append(row)

        st.dataframe(pd.DataFrame(selected_data), use_container_width=True, hide_index=True)

        # Achieved chemistry table
        if result.achieved_chemistry:
            st.subheader("Achieved Chemistry")

            chem_data = []
            for chem in result.achieved_chemistry:
                status = "✓" if chem.is_satisfied else "✗"

                # Format target display based on operator
                if chem.operator == "range" and chem.target_max is not None:
                    target_display = f"{chem.target:.2f}% – {chem.target_max:.2f}%"
                elif chem.operator == "=":
                    target_display = f"{chem.target:.2f}%"
                else:
                    target_display = f"{chem.operator} {chem.target:.2f}%"

                chem_data.append({
                    "Element": chem.element,
                    "Mode": chem.mode.capitalize(),
                    "Target": target_display,
                    "Achieved": f"{chem.achieved:.2f}%",
                    "Deviation": f"{chem.deviation:+.4f}",
                    "Status": status,
                })

            st.dataframe(pd.DataFrame(chem_data), use_container_width=True, hide_index=True)

        # Show recommendations if any
        if result.recommendations:
            st.subheader("💡 Recommendations")
            for rec in result.recommendations:
                st.warning(rec)


if __name__ == "__main__":
    main()
