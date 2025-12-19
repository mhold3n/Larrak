"""
Interpretation Core Library.
Provides ANOVA, Response Surface Modeling, and Sensitivity Analysis for DOE data.
"""

import warnings
from typing import Any

import numpy as np
import pandas as pd

# Try importing statsmodels for robust ANOVA
try:
    import statsmodels.api as sm
    from statsmodels.formula.api import ols

    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    # Fallback deps
    from scipy import stats  # noqa: F401
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import PolynomialFeatures


class DOEAnalyzer:
    """
    Analyzes Design of Experiments data.
    """

    def __init__(self, data_path: str | None = None, df: pd.DataFrame | None = None):
        if df is not None:
            self.df = df
        elif isinstance(data_path, str):
            self.df = pd.read_csv(data_path)
        elif isinstance(data_path, pd.DataFrame):
            self.df = data_path
        else:
            raise ValueError("Must provide data_path (str) or df (DataFrame)")

        self.model = None
        self.model_results = None
        self.input_cols: list[str] = []
        self.response_col: str = ""
        self.interactions: bool = True

    def fit_model(
        self, input_cols: list[str], response_col: str, degree: int = 2, interactions: bool = True
    ):
        """
        Fit a Response Surface Model (Polynomial).
        """
        self.input_cols = input_cols
        self.response_col = response_col
        self.interactions = interactions

        # Clean data (drop NaNs)
        df_clean = self.df.dropna(subset=input_cols + [response_col])
        if len(df_clean) < len(self.df):
            print(f"Warning: Dropped {len(self.df) - len(df_clean)} rows with NaNs.")
        self.df_clean = df_clean

        if STATSMODELS_AVAILABLE:
            self._fit_statsmodels(degree, interactions)
        else:
            self._fit_sklearn(degree, interactions)

    def _create_formula(self, degree: int, interactions: bool) -> str:
        """Create R-style formula string."""
        terms = []
        # Main effects and Powers
        for col in self.input_cols:
            terms.append(col)
            if degree >= 2:
                # Use I() to treat as arithmetic
                terms.append(f"I({col}**2)")
            if degree >= 3:
                terms.append(f"I({col}**3)")

        # Interactions (Degree 2 only for now)
        if interactions and len(self.input_cols) > 1:
            import itertools

            for c1, c2 in itertools.combinations(self.input_cols, 2):
                terms.append(f"{c1}:{c2}")

        formula = f"{self.response_col} ~ {' + '.join(terms)}"
        return formula

    def _fit_statsmodels(self, degree: int, interactions: bool):
        formula = self._create_formula(degree, interactions)
        self.model = ols(formula, data=self.df_clean)
        self.model_results = self.model.fit()
        print(f"Model R-squared: {self.model_results.rsquared:.4f}")

    def _fit_sklearn(self, degree: int, interactions: bool):
        # We process interactions manually via PolynomialFeatures if needed,
        # but to keep structure similar to statsmodels, we'll just use PolyFeat.
        poly = PolynomialFeatures(
            degree=degree, include_bias=False, interaction_only=not interactions and degree == 1
        )
        # Note: PolyFeatures(degree=2) includes interactions automatically.
        # If interactions=False but degree=2, we need to specificy... SKLearn makes this tricky to match exactly
        # main + squared WITHOUT interactions.

        X = self.df_clean[self.input_cols]
        y = self.df_clean[self.response_col]

        self.poly_trans = poly
        X_poly = poly.fit_transform(X)
        feat_names = poly.get_feature_names_out(self.input_cols)

        # Filter if interactions=False?
        # For simplicity, if fallback, we just use standard Poly expansion

        self.sklearn_model = LinearRegression()
        self.sklearn_model.fit(X_poly, y)
        self.sklearn_feat_names = feat_names

        # Calculate R2
        self.r2 = self.sklearn_model.score(X_poly, y)
        print(f"Model R-squared (sklearn): {self.r2:.4f}")

    def run_anova(self) -> pd.DataFrame:
        """
        Generate ANOVA table.
        """
        if self.model_results is None and self.sklearn_model is None:
            raise ValueError("Model not fitted. Call fit_model() first.")

        if STATSMODELS_AVAILABLE:
            table = sm.stats.anova_lm(self.model_results, typ=2)
            # Sort by F-value desc
            if "F" in table.columns:
                table = table.sort_values("F", ascending=False)
            return table
        else:
            # Fallback: Approximate "importance" via Coeff magnitude standardized
            # This is NOT a real ANOVA but a proxy for factor screening
            print("Warning: statsmodels not found. Returning Coefficient table instead of ANOVA.")
            coefs = self.sklearn_model.coef_
            names = self.sklearn_feat_names

            df_res = pd.DataFrame({"Term": names, "Coefficient": coefs})
            df_res["AbsCoeff"] = df_res["Coefficient"].abs()
            return df_res.sort_values("AbsCoeff", ascending=False)

    def compute_sensitivity(
        self, bounds: dict[str, tuple[float, float]], n_samples: int = 1000
    ) -> pd.DataFrame:
        """
        Compute Sensitivity Indices (Sobol-like via prediction variance or Gradient norm).
        Here we use "Morris-like" sensitivity: Average of absolute derivatives over the space.
        """
        # Generate random samples in bounds
        samples = pd.DataFrame()
        for col, (vmin, vmax) in bounds.items():
            samples[col] = np.random.uniform(vmin, vmax, n_samples)

        # We need derivatives.
        # For polynomial model, we can compute analytically or via Finite Diff.
        # Finite Diff is generic.

        sensitivities = {}
        delta = 1e-4

        base_preds = self.predict(samples)

        for col in self.input_cols:
            if col not in bounds:
                continue

            # Perturb
            samples_plus = samples.copy()
            samples_plus[col] += delta * (bounds[col][1] - bounds[col][0])  # relative perturbation

            preds_plus = self.predict(samples_plus)

            # dy/dx approx
            diff = (preds_plus - base_preds) / (delta * (bounds[col][1] - bounds[col][0]))

            # Metric: Mean Squared Derivative (similar to Morris mu*)
            sens_val = np.mean(np.abs(diff))
            # Normalize by range of Y?
            sensitivities[col] = sens_val

        # Convert to relative %
        total_sens = sum(sensitivities.values()) + 1e-12
        df_sens = pd.DataFrame(list(sensitivities.items()), columns=["Factor", "SensitivityScore"])
        df_sens["Relative %"] = 100.0 * df_sens["SensitivityScore"] / total_sens
        return df_sens.sort_values("SensitivityScore", ascending=False)

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        """Wrapper for prediction."""
        if STATSMODELS_AVAILABLE:
            return self.model_results.predict(x)
        else:
            x_poly = self.poly_trans.transform(x[self.input_cols])
            return self.sklearn_model.predict(x_poly)

    def get_standardized_effects(self) -> pd.DataFrame:
        """
        Get standardized effects (t-values or scaled coefficients).
        Used for Pareto charts.
        """
        if STATSMODELS_AVAILABLE and self.model_results:
            # t-values are standardized effects
            t_vals = self.model_results.tvalues
            df_eff = pd.DataFrame({"Term": t_vals.index, "Effect": t_vals.values})
            df_eff["AbsEffect"] = df_eff["Effect"].abs()
            # Drop Intercept
            df_eff = df_eff[df_eff["Term"] != "Intercept"]
            return df_eff.sort_values("AbsEffect", ascending=True)
        else:
            # Fallback: Approximate via coefficients (less accurate without StdErr)
            coefs = self.sklearn_model.coef_
            names = self.sklearn_feat_names
            df_eff = pd.DataFrame({"Term": names, "Effect": coefs})
            df_eff["AbsEffect"] = df_eff["Effect"].abs()
            return df_eff.sort_values("AbsEffect", ascending=True)

    def calculate_residuals(self) -> pd.DataFrame:
        """
        Calculate residuals (Observed - Predicted).
        Output DataFrame has columns: [Observed, Predicted, Residual]
        """
        y_true = self.df_clean[self.response_col].values

        if STATSMODELS_AVAILABLE and self.model_results:
            y_pred = self.model_results.predict(self.df_clean)
        else:
            x_poly = self.poly_trans.transform(self.df_clean[self.input_cols])
            y_pred = self.sklearn_model.predict(x_poly)

        df_res = pd.DataFrame(
            {"Observed": y_true, "Predicted": y_pred, "Residual": y_true - y_pred}
        )
        return df_res

    def get_interaction_grid(self, col1: str, col2: str, resolution: int = 20) -> pd.DataFrame:
        """
        Generate a grid of predictions for 2 variables, fixing others at mean/mode.
        Used for Interaction Plots.
        """
        # Create baseline row (mean of all inputs)
        baseline = self.df_clean[self.input_cols].mean().to_frame().T

        # Grid
        x1 = np.linspace(self.df_clean[col1].min(), self.df_clean[col1].max(), resolution)
        x2 = np.linspace(self.df_clean[col2].min(), self.df_clean[col2].max(), resolution)

        x_grid1, x_grid2 = np.meshgrid(x1, x2)
        grid_flat = pd.DataFrame(
            baseline.values.repeat(resolution * resolution, axis=0), columns=self.input_cols
        )

        grid_flat[col1] = x_grid1.ravel()
        grid_flat[col2] = x_grid2.ravel()

        preds = self.predict(grid_flat)
        grid_flat["Prediction"] = preds

        return grid_flat

    def generate_report_text(self) -> str:
        """Generates a text summary of the analysis."""
        lines = []
        lines.append(f"Analysis for Response: {self.response_col}")

        if STATSMODELS_AVAILABLE:
            lines.append(f"R-Squared: {self.model_results.rsquared:.4f}")
            lines.append("\nANOVA Table (Top 5):")
            anova = self.run_anova()
            lines.append(anova.head(5).to_string())
        else:
            lines.append(f"R-Squared: {self.r2:.4f}")
            lines.append("\nTop Coefficients:")
            res = self.run_anova()
            lines.append(res.head(5).to_string())

        return "\n".join(lines)

    def suggest_points_adaptive(
        self,
        bounds: dict[str, tuple[float, float]],
        objective_type: str | None = None,  # "maximize", "minimize", "target"
        target_value: float | None = None,
        min_spacing: float = 0.01,
        threshold_percentile: float = 90.0,
        min_slope_limit: float = 0.0,
        n_candidates: int = 5000,
        n_new: int = 5,
        slope_weight: float = 0.7,  # Balance between Slope (0.7) and Objective (0.3)
    ) -> list[dict[str, float]]:
        """
        Suggests new points using a combined score of Gradient Magnitude and Objective Proximity.
        Includes structured 7-point refinement for high-slope regions.
        """
        if self.model_results is None and self.sklearn_model is None:
            raise ValueError("Model not fitted. Cannot compute suggestions.")

        # 1. Candidate Generation (Global Search)
        candidates, grad_mag = self._compute_gradients(bounds, n_candidates)
        cand_df = candidates.copy()
        cand_df["grad_mag"] = grad_mag

        # Predict Values
        predictions = self.predict(candidates)
        cand_df["predicted"] = predictions

        # 2. Scoring Logic
        # A. Gradient Score (Normalized 0-1)
        g_min, g_max = np.min(grad_mag), np.max(grad_mag)
        if g_max > g_min:
            score_grad = (grad_mag - g_min) / (g_max - g_min)
        else:
            score_grad = np.zeros_like(grad_mag)

        # B. Objective Score (Normalized 0-1)
        score_obj = np.zeros_like(predictions)
        if objective_type:
            if objective_type == "maximize":
                p_min, p_max = np.min(predictions), np.max(predictions)
                if p_max > p_min:
                    score_obj = (predictions - p_min) / (p_max - p_min)
            elif objective_type == "minimize":
                p_min, p_max = np.min(predictions), np.max(predictions)
                if p_max > p_min:
                    score_obj = 1.0 - (predictions - p_min) / (p_max - p_min)
            elif objective_type == "target" and target_value is not None:
                dist = np.abs(predictions - target_value)
                d_min, d_max = np.min(dist), np.max(dist)
                if d_max > d_min:
                    score_obj = 1.0 - (dist - d_min) / (d_max - d_min)
                else:
                    score_obj = np.ones_like(dist)

        # C. Combined Score
        w_g = 1.0 if not objective_type else slope_weight
        w_o = 0.0 if not objective_type else (1.0 - slope_weight)

        total_score = (w_g * score_grad) + (w_o * score_obj)
        cand_df["total_score"] = total_score

        # 4. Selection
        cand_df = cand_df.sort_values("total_score", ascending=False)
        selected = []

        # 4a. Explicit Region Refinement for High Slopes
        # If any candidate has grad > min_slope_limit, trigger LOCAL 7-point design
        avg_slope = cand_df["grad_mag"].mean()
        limit_slope = max(min_slope_limit, avg_slope * 2.0)

        # Reserve some budget for refinement
        n_refine = max(1, n_new // 2) if n_new > 1 else 0
        high_slope_candidates = cand_df[cand_df["grad_mag"] > limit_slope].head(n_refine)

        refinement_points = []
        for _, row in high_slope_candidates.iterrows():
            local_bounds = {}
            for col in self.input_cols:
                b_min, b_max = bounds[col]
                span = b_max - b_min
                center = row[col]
                half_width = span * 0.05  # 10% total width
                local_bounds[col] = (
                    max(b_min, center - half_width),
                    min(b_max, center + half_width),
                )
            local_cluster = self._generate_local_7point(local_bounds)
            refinement_points.extend(local_cluster)

        # Select refinement points first (ignoring spacing to force zooming)
        for pt in refinement_points:
            if len(selected) >= n_new:
                break
            selected.append(pt)

        # Fill remaining budget with global top scorers
        # Here we use greedy spacing
        # Generate norm map for spacing
        scale_map = {k: (bounds[k][1] - bounds[k][0]) for k in bounds}

        def dist_sq(p1, p2):
            d = 0.0
            for k in bounds:
                d += ((p1[k] - p2[k]) / scale_map[k]) ** 2
            return d

        existing_pts = self.df_clean[list(bounds.keys())].to_dict("records")
        min_dist_sq = min_spacing**2

        for _, row in cand_df.iterrows():
            if len(selected) >= n_new:
                break

            pt = row[self.input_cols].to_dict()

            # Check selected
            too_close = False
            for s in selected:
                if dist_sq(pt, s) < min_dist_sq:
                    too_close = True
                    break
            if too_close:
                continue

            # Check existing
            for e in existing_pts:
                if dist_sq(pt, e) < min_dist_sq:
                    too_close = True
                    break
            if too_close:
                continue

            selected.append(pt)

        return selected

    def _generate_local_7point(
        self, bounds: dict[str, tuple[float, float]]
    ) -> list[dict[str, float]]:
        """Generates a 7-point design (Center + Extremes) in the box."""
        points = []
        center = {k: (v[0] + v[1]) / 2.0 for k, v in bounds.items()}
        points.append(center)
        for dim, (b_min, b_max) in bounds.items():
            p_low = center.copy()
            p_low[dim] = b_min
            points.append(p_low)
            p_high = center.copy()
            p_high[dim] = b_max
            points.append(p_high)
        return points

    def _compute_gradients(
        self, bounds: dict[str, tuple[float, float]], n_candidates: int
    ) -> tuple[pd.DataFrame, np.ndarray]:
        """Compute gradients for random candidates."""
        candidates = pd.DataFrame()
        for col, (vmin, vmax) in bounds.items():
            candidates[col] = np.random.uniform(vmin, vmax, n_candidates)

        grads = np.zeros(n_candidates)
        delta = 1e-5
        base_preds = self.predict(candidates)

        for col, (vmin, vmax) in bounds.items():
            span = vmax - vmin
            step = delta * span

            temp_cand = candidates.copy()
            temp_cand[col] += step
            preds_plus = self.predict(temp_cand)

            local_grad = (preds_plus - base_preds) / delta
            grads += local_grad**2

        return candidates, np.sqrt(grads)

    def _select_spaced_candidates(
        self,
        candidates: pd.DataFrame,
        gradients: np.ndarray,
        bounds: dict[str, tuple[float, float]],
        min_spacing: float,
        n_new: int,
    ) -> list[dict[str, float]]:
        """
        Select points respecting minimum spacing, adapted by local gradient.
        Higher gradient -> Smaller effective spacing allowed.
        """
        proposed: list[dict[str, Any]] = []

        def normalize(df):
            norm_df = df.copy()
            for col, (vmin, vmax) in bounds.items():
                if col in norm_df.columns:
                    norm_df[col] = (norm_df[col] - vmin) / (vmax - vmin)
            return norm_df

        existing_norm = normalize(self.df_clean[list(bounds.keys())])
        candidates_norm = normalize(candidates)
        existing_points = existing_norm.values

        # Max gradient for normalization
        max_grad = np.max(gradients) if len(gradients) > 0 else 1.0

        for idx, row in candidates_norm.iterrows():
            if len(proposed) >= n_new:
                break

            # Get integer index for gradient array lookup
            # Since candidates is a slice, idx might not be 0..N
            # We need to pass the gradients ALIGNED with candidates df
            local_grad = gradients[candidates.index.get_loc(idx)]

            # Adaptive Spacing:
            # If grad is high, current spacing req is reduced.
            # Factor: at max_grad, spacing is 25% of min_spacing (4x density).
            grad_factor = local_grad / (max_grad + 1e-9)
            effective_spacing = min_spacing / (1.0 + 3.0 * grad_factor)

            p_cand = np.round(row.values, 2)

            # Check distance to EXISTING
            if np.min(np.linalg.norm(existing_points - p_cand, axis=1)) < effective_spacing:
                continue

            # Check distance to ALREADY PROPOSED
            if proposed:
                dists_prop = np.linalg.norm(
                    np.array([p["norm"] for p in proposed]) - p_cand, axis=1
                )
                if np.min(dists_prop) < effective_spacing:
                    continue

            original_row = candidates.loc[idx].to_dict()
            proposed.append({"data": original_row, "norm": p_cand})

        return [p["data"] for p in proposed]

    def validate_coverage(
        self, bounds: dict[str, tuple[float, float]], n_neighbors: int = 1
    ) -> dict[str, float]:
        """
        Validate the coverage of the design space.
        Returns metrics:
          - 'max_min_dist': The largest distance from any point in space to a sample point (space-filling metric).
          - 'mean_min_dist': Average distance to nearest neighbor.
        """
        from sklearn.neighbors import NearestNeighbors

        if self.df_clean.empty:
            return {"max_min_dist": 0.0, "mean_min_dist": 0.0}

        # Normalize existing points
        def normalize(df):
            norm_df = df.copy()
            for col, (vmin, vmax) in bounds.items():
                if col in norm_df.columns:
                    norm_df[col] = (norm_df[col] - vmin) / (vmax - vmin)
            return norm_df

        existing_norm = normalize(self.df_clean[list(bounds.keys())]).values

        # 1. Mean Min Dist (inter-point spacing)
        nbrs = NearestNeighbors(n_neighbors=n_neighbors + 1).fit(existing_norm)
        distances, _ = nbrs.kneighbors(existing_norm)
        # column 0 is self (0 dist), column 1 is nearest neighbor
        mean_min_dist = np.mean(distances[:, 1])

        # 2. Max Min Dist (Hole finding) - Monte Carlo estimation
        # Generate random probes
        n_probes = 2000
        probes = np.random.rand(n_probes, len(bounds))

        # Find distance from each probe to NEAREST existing point
        # We can use the same NN structure
        dist_probes, _ = nbrs.kneighbors(probes, n_neighbors=1)
        max_min_dist = np.max(dist_probes)

        return {"mean_min_dist": float(mean_min_dist), "max_min_dist": float(max_min_dist)}
