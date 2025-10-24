"""Adaptive solver selection based on problem characteristics and analysis history."""

from dataclasses import dataclass
from typing import Dict, List, Optional
from enum import Enum

from campro.logging import get_logger
from campro.optimization.solver_analysis import MA57ReadinessReport
from campro.optimization.solver_detection import is_ma57_available

log = get_logger(__name__)


class SolverType(Enum):
    """Available linear solvers."""
    MA27 = "ma27"
    MA57 = "ma57"  # For future use when available
    # Non-HSL solvers are not permitted in this project


@dataclass
class ProblemCharacteristics:
    """Characteristics of the optimization problem."""
    n_variables: int
    n_constraints: int
    problem_type: str  # "thermal", "litvin", "crank_center"
    expected_iterations: int
    linear_solver_ratio: float
    has_convergence_issues: bool


@dataclass
class AnalysisHistory:
    """Historical analysis data for decision making."""
    avg_grade: str
    avg_linear_solver_ratio: float
    avg_iterations: int
    convergence_issues_count: int
    ma57_benefits: List[bool]


class AdaptiveSolverSelector:
    """Select optimal solver based on problem characteristics and history."""
    
    def __init__(self):
        self.analysis_history: Dict[str, AnalysisHistory] = {}
    
    def select_solver(self, problem_chars: ProblemCharacteristics, 
                     phase: str) -> SolverType:
        """
        Select optimal solver for given problem characteristics.
        
        Currently always returns MA27. Future: add logic for MA57 selection
        when available based on problem characteristics.
        """
        # Get historical data for this phase
        history = self.analysis_history.get(phase)
        
        # Revised decision logic: prefer MA57 when available **and** history
        # indicates potential benefit; otherwise default to MA27.

        ma57_available = is_ma57_available()

        if ma57_available and self.should_consider_ma57(phase):
            chosen = SolverType.MA57
        else:
            chosen = SolverType.MA27

        log.debug(
            "Selected solver for %s phase: %s (MA57 available=%s, readiness=%s)",
            phase,
            chosen.value,
            ma57_available,
            self.should_consider_ma57(phase),
        )

        return chosen
    
    def update_history(self, phase: str, analysis: MA57ReadinessReport):
        """Update analysis history for future decisions."""
        # Handle case where analysis is None (optimization failed)
        if analysis is None:
            log.warning(f"No analysis available for phase {phase}, skipping history update")
            return
            
        if phase not in self.analysis_history:
            self.analysis_history[phase] = AnalysisHistory(
                avg_grade=analysis.grade,
                avg_linear_solver_ratio=analysis.stats.get('ls_time_ratio', 0.0),
                avg_iterations=analysis.stats.get('iter_count', 0),
                convergence_issues_count=1 if analysis.grade in ["medium", "high"] else 0,
                ma57_benefits=[analysis.grade in ["medium", "high"]]
            )
        else:
            # Update running averages
            history = self.analysis_history[phase]
            n = len(history.ma57_benefits)
            
            # Moving average for numerical metrics
            history.avg_linear_solver_ratio = (
                (history.avg_linear_solver_ratio * n + analysis.stats.get('ls_time_ratio', 0.0)) / (n + 1)
            )
            history.avg_iterations = int(
                (history.avg_iterations * n + analysis.stats.get('iter_count', 0)) / (n + 1)
            )
            
            # Update counts
            if analysis.grade in ["medium", "high"]:
                history.convergence_issues_count += 1
                history.ma57_benefits.append(True)
            else:
                history.ma57_benefits.append(False)
            
            # Update grade (most recent)
            history.avg_grade = analysis.grade
        
        log.debug(f"Updated analysis history for {phase} phase: grade={analysis.grade}, "
                 f"ls_ratio={analysis.stats.get('ls_time_ratio', 0.0):.3f}")
    
    def get_history_summary(self, phase: str) -> Optional[Dict]:
        """Get summary of analysis history for a phase."""
        if phase not in self.analysis_history:
            return None
        
        history = self.analysis_history[phase]
        return {
            "phase": phase,
            "avg_grade": history.avg_grade,
            "avg_linear_solver_ratio": history.avg_linear_solver_ratio,
            "avg_iterations": history.avg_iterations,
            "convergence_issues_count": history.convergence_issues_count,
            "ma57_benefit_percentage": sum(history.ma57_benefits) / len(history.ma57_benefits) if history.ma57_benefits else 0.0,
            "total_analyses": len(history.ma57_benefits)
        }
    
    def get_all_history_summaries(self) -> Dict[str, Dict]:
        """Get summaries for all phases."""
        return {
            phase: self.get_history_summary(phase)
            for phase in self.analysis_history.keys()
        }
    
    def clear_history(self, phase: Optional[str] = None):
        """Clear analysis history for a phase or all phases."""
        if phase is None:
            self.analysis_history.clear()
            log.info("Cleared all analysis history")
        elif phase in self.analysis_history:
            del self.analysis_history[phase]
            log.info(f"Cleared analysis history for {phase} phase")
        else:
            log.warning(f"No analysis history found for {phase} phase")
    
    def should_consider_ma57(self, phase: str) -> bool:
        """
        Determine if MA57 should be considered for future optimizations.
        
        This is a placeholder for future MA57 availability logic.
        """
        if phase not in self.analysis_history:
            return False
        
        history = self.analysis_history[phase]
        
        # Criteria for considering MA57:
        # 1. High linear solver time ratio
        # 2. Frequent convergence issues
        # 3. Large problem sizes (would need to be passed in)
        
        ma57_benefit_percentage = sum(history.ma57_benefits) / len(history.ma57_benefits) if history.ma57_benefits else 0.0
        
        return (history.avg_linear_solver_ratio > 0.4 or 
                ma57_benefit_percentage > 0.5 or
                history.convergence_issues_count > 3)
    
    def get_recommendation(self, phase: str) -> str:
        """Get solver recommendation for a phase."""
        if self.should_consider_ma57(phase):
            return "Consider MA57 when available"
        else:
            return "MA27 is sufficient"
