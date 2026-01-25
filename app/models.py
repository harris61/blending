"""Pydantic data models for the Mining Stockpile Blending Optimizer."""

from typing import Optional, Literal
from pydantic import BaseModel, Field


class Stockpile(BaseModel):
    """Represents a single stockpile with its properties."""
    name: str
    tonnage_available: int = Field(gt=0, description="Available tonnage in the stockpile")
    distance_km: Optional[float] = Field(default=None, ge=0, description="Distance from destination")
    cost_per_ton: Optional[float] = Field(default=None, ge=0, description="Cost per ton")
    revenue_per_ton: Optional[float] = Field(default=None, ge=0, description="Revenue per ton")
    chemistry: dict[str, float] = Field(default_factory=dict, description="Element percentages")


class ChemistryTarget(BaseModel):
    """Target and weight for a single chemistry element."""
    target: float = Field(ge=0, le=100, description="Target percentage")
    mode: Literal["exact", "approximate"] = Field(default="approximate", description="Exact or approximate targeting")
    weight: float = Field(ge=0, default=1.0, description="Priority weight for optimization")


class OptimizationRequest(BaseModel):
    """Request payload for the optimization endpoint."""
    stockpiles: list[Stockpile]
    target_tonnage: int = Field(gt=0, description="Total tonnage required")
    chemistry_targets: dict[str, ChemistryTarget] = Field(
        default_factory=dict, description="Element targets with weights"
    )
    optimization_mode: Literal["distance", "material", "profit"] = Field(
        default="distance", description="What to optimize"
    )
    min_increment: int = Field(
        default=100, ge=1, description="Minimum blending unit in wmt"
    )


class SelectedStockpile(BaseModel):
    """Result for a single stockpile in the solution."""
    name: str
    tonnage_taken: int
    tonnage_available: int
    distance_km: Optional[float]
    material_cost: float
    revenue: float
    profit: float


class CostBreakdown(BaseModel):
    """Breakdown of costs and revenue in the optimization result."""
    material_total: float
    material_per_ton: float
    cost_total: float
    cost_per_ton: float
    revenue_total: float
    revenue_per_ton: float
    profit_total: float
    profit_per_ton: float
    distance_avg_km: float


class AchievedChemistry(BaseModel):
    """Chemistry achievement for a single element."""
    element: str
    target: float
    achieved: float
    deviation: float
    mode: str = "approximate"
    is_exact_match: bool = False


class OptimizationResult(BaseModel):
    """Result of the optimization."""
    success: bool
    message: str
    selected_stockpiles: list[SelectedStockpile] = Field(default_factory=list)
    cost_breakdown: Optional[CostBreakdown] = None
    achieved_chemistry: list[AchievedChemistry] = Field(default_factory=list)
    total_tonnage: int = 0
    recommendations: list[str] = Field(default_factory=list)


class CSVUploadResponse(BaseModel):
    """Response from CSV upload endpoint."""
    success: bool
    message: str
    stockpiles: list[Stockpile] = Field(default_factory=list)
    chemistry_columns: list[str] = Field(default_factory=list)
    total_available_tonnage: int = 0
