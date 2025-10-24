"""
MA57 Migration Analyzer

This module provides comprehensive analysis and migration planning for transitioning
from MA27 to MA57 linear solver in Ipopt optimization problems.
"""

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from campro.logging import get_logger
from campro.optimization.solver_analysis import MA57ReadinessReport

log = get_logger(__name__)


@dataclass
class MigrationDataPoint:
    """Single data point for MA57 migration analysis."""
    timestamp: datetime
    phase: str  # "primary", "secondary", "tertiary"
    problem_size: Tuple[int, int]  # (n_variables, n_constraints)
    ma27_report: MA57ReadinessReport
    ma57_report: Optional[MA57ReadinessReport] = None
    performance_improvement: Optional[float] = None  # Speedup factor
    convergence_improvement: Optional[bool] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MigrationAnalysis:
    """Comprehensive analysis of MA57 migration readiness."""
    total_runs: int
    ma57_beneficial_runs: int
    average_speedup: Optional[float]
    convergence_improvements: int
    problem_size_analysis: Dict[str, Any]
    phase_analysis: Dict[str, Any]
    recommendations: List[str]
    migration_priority: str  # "low", "medium", "high"
    estimated_effort: str  # "low", "medium", "high"


class MA57MigrationAnalyzer:
    """
    Analyzes optimization runs to determine MA57 migration benefits and priorities.
    """

    def __init__(self, data_file: Optional[str] = None):
        """
        Initialize the MA57 migration analyzer.
        
        Args:
            data_file: Path to JSON file for persistent data storage
        """
        self.data_file = data_file or "ma57_migration_data.json"
        self.data_points: List[MigrationDataPoint] = []
        self._load_data()

    def _load_data(self):
        """Load migration data from file."""
        if Path(self.data_file).exists():
            try:
                with open(self.data_file, 'r') as f:
                    data = json.load(f)
                    self.data_points = [
                        MigrationDataPoint(
                            timestamp=datetime.fromisoformat(dp['timestamp']),
                            phase=dp['phase'],
                            problem_size=tuple(dp['problem_size']),
                            ma27_report=MA57ReadinessReport(**dp['ma27_report']),
                            ma57_report=MA57ReadinessReport(**dp['ma57_report']) if dp.get('ma57_report') else None,
                            performance_improvement=dp.get('performance_improvement'),
                            convergence_improvement=dp.get('convergence_improvement'),
                            metadata=dp.get('metadata', {})
                        )
                        for dp in data.get('data_points', [])
                    ]
                log.info(f"Loaded {len(self.data_points)} migration data points")
            except Exception as e:
                log.warning(f"Failed to load migration data: {e}")
                self.data_points = []

    def _save_data(self):
        """Save migration data to file."""
        try:
            data = {
                'data_points': [
                    {
                        'timestamp': dp.timestamp.isoformat(),
                        'phase': dp.phase,
                        'problem_size': list(dp.problem_size),
                        'ma27_report': {
                            'grade': dp.ma27_report.grade,
                            'reasons': dp.ma27_report.reasons,
                            'suggested_action': dp.ma27_report.suggested_action,
                            'stats': dp.ma27_report.stats,
                        },
                        'ma57_report': {
                            'grade': dp.ma57_report.grade,
                            'reasons': dp.ma57_report.reasons,
                            'suggested_action': dp.ma57_report.suggested_action,
                            'stats': dp.ma57_report.stats,
                        } if dp.ma57_report else None,
                        'performance_improvement': dp.performance_improvement,
                        'convergence_improvement': dp.convergence_improvement,
                        'metadata': dp.metadata,
                    }
                    for dp in self.data_points
                ]
            }
            with open(self.data_file, 'w') as f:
                json.dump(data, f, indent=2)
            log.debug(f"Saved {len(self.data_points)} migration data points")
        except Exception as e:
            log.error(f"Failed to save migration data: {e}")

    def add_ma27_run(
        self,
        phase: str,
        problem_size: Tuple[int, int],
        ma27_report: MA57ReadinessReport,
        metadata: Optional[Dict[str, Any]] = None
    ) -> MigrationDataPoint:
        """
        Add a MA27 run to the migration analysis.
        
        Args:
            phase: Optimization phase ("primary", "secondary", "tertiary")
            problem_size: Tuple of (n_variables, n_constraints)
            ma27_report: Analysis report from MA27 run
            metadata: Additional metadata about the run
            
        Returns:
            MigrationDataPoint that was created
        """
        data_point = MigrationDataPoint(
            timestamp=datetime.now(),
            phase=phase,
            problem_size=problem_size,
            ma27_report=ma27_report,
            metadata=metadata or {}
        )
        
        self.data_points.append(data_point)
        self._save_data()
        
        log.info(f"Added MA27 run for {phase} phase (size: {problem_size})")
        return data_point

    def update_with_ma57_run(
        self,
        data_point: MigrationDataPoint,
        ma57_report: MA57ReadinessReport,
        performance_improvement: Optional[float] = None,
        convergence_improvement: Optional[bool] = None
    ):
        """
        Update a data point with MA57 run results.
        
        Args:
            data_point: The data point to update
            ma57_report: Analysis report from MA57 run
            performance_improvement: Speedup factor (MA27_time / MA57_time)
            convergence_improvement: Whether MA57 achieved better convergence
        """
        data_point.ma57_report = ma57_report
        data_point.performance_improvement = performance_improvement
        data_point.convergence_improvement = convergence_improvement
        
        self._save_data()
        log.info(f"Updated data point with MA57 results (speedup: {performance_improvement})")

    def analyze_migration_readiness(self) -> MigrationAnalysis:
        """
        Analyze all collected data to determine migration readiness.
        
        Returns:
            Comprehensive migration analysis
        """
        if not self.data_points:
            return MigrationAnalysis(
                total_runs=0,
                ma57_beneficial_runs=0,
                average_speedup=None,
                convergence_improvements=0,
                problem_size_analysis={},
                phase_analysis={},
                recommendations=["No data available for analysis"],
                migration_priority="low",
                estimated_effort="low"
            )

        # Analyze overall statistics
        total_runs = len(self.data_points)
        ma57_runs = [dp for dp in self.data_points if dp.ma57_report is not None]
        ma57_beneficial_runs = len([dp for dp in ma57_runs if self._is_ma57_beneficial(dp)])
        
        # Calculate average speedup
        speedups = [dp.performance_improvement for dp in ma57_runs if dp.performance_improvement is not None]
        average_speedup = sum(speedups) / len(speedups) if speedups else None
        
        # Count convergence improvements
        convergence_improvements = len([dp for dp in ma57_runs if dp.convergence_improvement is True])
        
        # Analyze by problem size
        problem_size_analysis = self._analyze_by_problem_size()
        
        # Analyze by phase
        phase_analysis = self._analyze_by_phase()
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            total_runs, ma57_beneficial_runs, average_speedup, convergence_improvements
        )
        
        # Determine migration priority
        migration_priority = self._determine_migration_priority(
            ma57_beneficial_runs, total_runs, average_speedup, convergence_improvements
        )
        
        # Estimate effort
        estimated_effort = self._estimate_migration_effort(migration_priority, total_runs)
        
        return MigrationAnalysis(
            total_runs=total_runs,
            ma57_beneficial_runs=ma57_beneficial_runs,
            average_speedup=average_speedup,
            convergence_improvements=convergence_improvements,
            problem_size_analysis=problem_size_analysis,
            phase_analysis=phase_analysis,
            recommendations=recommendations,
            migration_priority=migration_priority,
            estimated_effort=estimated_effort
        )

    def _is_ma57_beneficial(self, data_point: MigrationDataPoint) -> bool:
        """Determine if MA57 was beneficial for a given run."""
        if not data_point.ma57_report:
            return False
        
        # Check performance improvement
        if data_point.performance_improvement and data_point.performance_improvement > 1.2:
            return True
        
        # Check convergence improvement
        if data_point.convergence_improvement:
            return True
        
        # Check grade improvement
        ma27_grade = data_point.ma27_report.grade
        ma57_grade = data_point.ma57_report.grade
        
        grade_benefit = {
            ("high", "medium"): True,
            ("high", "low"): True,
            ("medium", "low"): True,
        }
        
        return grade_benefit.get((ma27_grade, ma57_grade), False)

    def _analyze_by_problem_size(self) -> Dict[str, Any]:
        """Analyze migration benefits by problem size."""
        if not self.data_points:
            return {}
        
        # Group by problem size ranges
        small_problems = [dp for dp in self.data_points if sum(dp.problem_size) < 100]
        medium_problems = [dp for dp in self.data_points if 100 <= sum(dp.problem_size) < 500]
        large_problems = [dp for dp in self.data_points if sum(dp.problem_size) >= 500]
        
        def analyze_group(problems: List[MigrationDataPoint], name: str) -> Dict[str, Any]:
            if not problems:
                return {"count": 0, "ma57_beneficial": 0, "avg_speedup": None}
            
            ma57_problems = [dp for dp in problems if dp.ma57_report is not None]
            beneficial = len([dp for dp in ma57_problems if self._is_ma57_beneficial(dp)])
            speedups = [dp.performance_improvement for dp in ma57_problems if dp.performance_improvement is not None]
            avg_speedup = sum(speedups) / len(speedups) if speedups else None
            
            return {
                "count": len(problems),
                "ma57_beneficial": beneficial,
                "beneficial_percentage": beneficial / len(ma57_problems) if ma57_problems else 0,
                "avg_speedup": avg_speedup
            }
        
        return {
            "small_problems": analyze_group(small_problems, "small"),
            "medium_problems": analyze_group(medium_problems, "medium"),
            "large_problems": analyze_group(large_problems, "large")
        }

    def _analyze_by_phase(self) -> Dict[str, Any]:
        """Analyze migration benefits by optimization phase."""
        phases = ["primary", "secondary", "tertiary"]
        analysis = {}
        
        for phase in phases:
            phase_problems = [dp for dp in self.data_points if dp.phase == phase]
            if not phase_problems:
                analysis[phase] = {"count": 0, "ma57_beneficial": 0, "avg_speedup": None}
                continue
            
            ma57_problems = [dp for dp in phase_problems if dp.ma57_report is not None]
            beneficial = len([dp for dp in ma57_problems if self._is_ma57_beneficial(dp)])
            speedups = [dp.performance_improvement for dp in ma57_problems if dp.performance_improvement is not None]
            avg_speedup = sum(speedups) / len(speedups) if speedups else None
            
            analysis[phase] = {
                "count": len(phase_problems),
                "ma57_beneficial": beneficial,
                "beneficial_percentage": beneficial / len(ma57_problems) if ma57_problems else 0,
                "avg_speedup": avg_speedup
            }
        
        return analysis

    def _generate_recommendations(
        self,
        total_runs: int,
        ma57_beneficial_runs: int,
        average_speedup: Optional[float],
        convergence_improvements: int
    ) -> List[str]:
        """Generate migration recommendations based on analysis."""
        recommendations = []
        
        if total_runs == 0:
            recommendations.append("No optimization runs available for analysis")
            return recommendations
        
        beneficial_percentage = ma57_beneficial_runs / total_runs if total_runs > 0 else 0
        
        if beneficial_percentage > 0.7:
            recommendations.append("Strong evidence for MA57 migration - high benefit rate")
        elif beneficial_percentage > 0.4:
            recommendations.append("Moderate evidence for MA57 migration - consider selective adoption")
        else:
            recommendations.append("Limited evidence for MA57 migration - investigate specific use cases")
        
        if average_speedup and average_speedup > 1.5:
            recommendations.append(f"Significant performance improvement expected (avg {average_speedup:.1f}x speedup)")
        elif average_speedup and average_speedup > 1.2:
            recommendations.append(f"Moderate performance improvement expected (avg {average_speedup:.1f}x speedup)")
        
        if convergence_improvements > 0:
            recommendations.append(f"MA57 improved convergence in {convergence_improvements} cases")
        
        if total_runs < 10:
            recommendations.append("Collect more optimization runs for better statistical confidence")
        
        return recommendations

    def _determine_migration_priority(
        self,
        ma57_beneficial_runs: int,
        total_runs: int,
        average_speedup: Optional[float],
        convergence_improvements: int
    ) -> str:
        """Determine migration priority based on analysis."""
        if total_runs == 0:
            return "low"
        
        beneficial_percentage = ma57_beneficial_runs / total_runs
        
        # High priority conditions
        if (beneficial_percentage > 0.7 and 
            (average_speedup is None or average_speedup > 1.3) and
            convergence_improvements > 0):
            return "high"
        
        # Medium priority conditions
        if (beneficial_percentage > 0.4 or
            (average_speedup and average_speedup > 1.5) or
            convergence_improvements > total_runs * 0.2):
            return "medium"
        
        return "low"

    def _estimate_migration_effort(self, priority: str, total_runs: int) -> str:
        """Estimate migration effort based on priority and data volume."""
        if priority == "high":
            return "medium" if total_runs < 50 else "high"
        elif priority == "medium":
            return "low" if total_runs < 20 else "medium"
        else:
            return "low"

    def get_migration_plan(self) -> Dict[str, Any]:
        """
        Generate a comprehensive migration plan.
        
        Returns:
            Dictionary containing migration plan details
        """
        analysis = self.analyze_migration_readiness()
        
        plan = {
            "analysis_summary": {
                "total_runs": analysis.total_runs,
                "ma57_beneficial_runs": analysis.ma57_beneficial_runs,
                "migration_priority": analysis.migration_priority,
                "estimated_effort": analysis.estimated_effort
            },
            "recommendations": analysis.recommendations,
            "phase_priorities": self._get_phase_priorities(analysis.phase_analysis),
            "implementation_steps": self._get_implementation_steps(analysis.migration_priority),
            "success_metrics": self._get_success_metrics(),
            "rollback_plan": self._get_rollback_plan()
        }
        
        return plan

    def _get_phase_priorities(self, phase_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get phase priorities for migration."""
        phases = []
        for phase, analysis in phase_analysis.items():
            if analysis["count"] > 0:
                priority = "high" if analysis["beneficial_percentage"] > 0.6 else "medium" if analysis["beneficial_percentage"] > 0.3 else "low"
                phases.append({
                    "phase": phase,
                    "priority": priority,
                    "beneficial_percentage": analysis["beneficial_percentage"],
                    "avg_speedup": analysis["avg_speedup"]
                })
        
        # Sort by priority
        priority_order = {"high": 3, "medium": 2, "low": 1}
        phases.sort(key=lambda x: priority_order[x["priority"]], reverse=True)
        
        return phases

    def _get_implementation_steps(self, priority: str) -> List[str]:
        """Get implementation steps based on priority."""
        if priority == "high":
            return [
                "1. Install MA57 linear solver",
                "2. Update Ipopt configuration to use MA57",
                "3. Run parallel MA27/MA57 comparison tests",
                "4. Monitor performance and convergence metrics",
                "5. Gradually migrate all optimization phases",
                "6. Update documentation and training materials"
            ]
        elif priority == "medium":
            return [
                "1. Evaluate MA57 availability and licensing",
                "2. Run targeted comparison tests on problematic cases",
                "3. Implement selective MA57 usage for specific phases",
                "4. Monitor results and expand usage gradually"
            ]
        else:
            return [
                "1. Continue collecting optimization data",
                "2. Monitor for convergence issues that might benefit from MA57",
                "3. Re-evaluate migration decision in 3-6 months"
            ]

    def _get_success_metrics(self) -> List[str]:
        """Get success metrics for migration evaluation."""
        return [
            "Average solve time improvement > 20%",
            "Convergence rate improvement > 10%",
            "Reduction in restoration phase activations",
            "No increase in numerical issues",
            "User satisfaction with optimization performance"
        ]

    def _get_rollback_plan(self) -> List[str]:
        """Get rollback plan in case of issues."""
        return [
            "1. Maintain MA27 as fallback option",
            "2. Implement automatic fallback on convergence failures",
            "3. Keep detailed logs of MA57 vs MA27 performance",
            "4. Have quick configuration switch available",
            "5. Document any issues encountered for future reference"
        ]

    def export_analysis_report(self, output_file: str):
        """
        Export comprehensive analysis report to file.
        
        Args:
            output_file: Path to output file (JSON format)
        """
        analysis = self.analyze_migration_readiness()
        plan = self.get_migration_plan()
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "migration_analysis": {
                "total_runs": analysis.total_runs,
                "ma57_beneficial_runs": analysis.ma57_beneficial_runs,
                "average_speedup": analysis.average_speedup,
                "convergence_improvements": analysis.convergence_improvements,
                "migration_priority": analysis.migration_priority,
                "estimated_effort": analysis.estimated_effort
            },
            "problem_size_analysis": analysis.problem_size_analysis,
            "phase_analysis": analysis.phase_analysis,
            "recommendations": analysis.recommendations,
            "migration_plan": plan,
            "data_points": [
                {
                    "timestamp": dp.timestamp.isoformat(),
                    "phase": dp.phase,
                    "problem_size": list(dp.problem_size),
                    "ma27_grade": dp.ma27_report.grade,
                    "ma57_grade": dp.ma57_report.grade if dp.ma57_report else None,
                    "performance_improvement": dp.performance_improvement,
                    "convergence_improvement": dp.convergence_improvement
                }
                for dp in self.data_points
            ]
        }
        
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        log.info(f"Exported migration analysis report to {output_file}")




