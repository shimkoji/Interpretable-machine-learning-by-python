from itertools import product
from typing import Any, List, Literal, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from joblib import Parallel, delayed


class PartialDependence:
    def __init__(
        self,
        model,
        X: pd.DataFrame,
        var_names: list[str],
        pred_type: Literal["regression", "classification"],
    ):
        """
        Initializes the PartialDependence object.

        Args:
            model: Trained scikit-learn model.
            X: pandas DataFrame used for training.
            var_names: List of feature names.
            pred_type: Type of prediction ("regression" or "classification").
        """
        self.model = model
        self.X = X.copy()
        self.var_names = var_names
        self.pred_type = pred_type

    def partial_dependence(self, var_name: str, n_grid: int = 50):
        """
        Calculates the partial dependence for given variables.

        Args:
            var_name: Variable name.
            n_grid: Number of grid points per variable.

        Returns:
            A pandas DataFrame containing the grid points and average predictions.
        """

        if not isinstance(self.X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame.")

        grid_ranges = {}
        grid_ranges[var_name] = np.linspace(
            self.X[var_name].min(), self.X[var_name].max(), num=n_grid
        )

        grid_points = list(product(*grid_ranges.values()))
        grid_points_df = pd.DataFrame(grid_points, columns=[var_name])

        average_predictions = []
        for _, row in grid_points_df.iterrows():
            average_predictions.append(self._counterfactual_prediction(row).mean())

        grid_points_df["avg_pred"] = average_predictions
        return grid_points_df

    def _counterfactual_prediction(self, grid_point_series):
        """Makes counterfactual predictions."""

        X_counterfactual = self.X.copy()
        for (
            var_name,
            grid_point,
        ) in grid_point_series.items():
            X_counterfactual[var_name] = grid_point

        if self.pred_type == "regression":
            return self.model.predict(X_counterfactual)
        else:
            return self.model.predict_proba(X_counterfactual)[:, 1]


class IndividualConditionalExpectation(PartialDependence):
    """Individual Conditional Expectation"""

    def individual_conditional_expectation(
        self, var_name: str, n_grid: int = 50
    ) -> None:
        """Calculates Individual Conditional Expectation (ICE) values.

        Args:
            var_name: The name of the feature for which to calculate ICE.
            n_grid: The number of grid points to use for evaluating the feature's range.  A finer grid (larger value) can capture more nuanced relationships but may be more computationally expensive and potentially noisy.  A coarser grid may miss important details.  Defaults to 50.
        """
        ids_to_compute = [i for i in range(self.X.shape[0])]
        self.target_var_name = var_name
        value_range = np.linspace(
            self.X[var_name].min(), self.X[var_name].max(), num=n_grid
        )
        individual_prediction = np.array(
            [
                self._counterfactual_prediction({self.target_var_name: x})[
                    ids_to_compute
                ]
                for x in value_range
            ]
        )

        self.df_ice = (
            pd.DataFrame(data=individual_prediction, columns=ids_to_compute)
            .assign(**{var_name: value_range})
            .melt(id_vars=var_name, var_name="instance", value_name="ice")
        )
        min_ice_by_instance = self.df_ice.loc[
            self.df_ice.groupby("instance")[var_name].idxmin()
        ]
        min_ice_dict = dict(
            zip(min_ice_by_instance["instance"], min_ice_by_instance["ice"])
        )
        self.df_ice["ice_diff"] = self.df_ice.apply(
            lambda row: row["ice"] - min_ice_dict[row["instance"]], axis=1
        )

        if self.pred_type == "regression":
            prediction = self.model.predict(self.X.iloc[ids_to_compute])
        else:
            prediction = self.model.predict_proba(self.X.iloc[ids_to_compute])[:, 1]
        self.df_instance = (
            pd.DataFrame(data=self.X.iloc[ids_to_compute], columns=self.var_names)
            .assign(
                instance=ids_to_compute,
                prediction=prediction,
            )
            .loc[:, ["instance", "prediction"] + self.var_names]
        )

    def plot(self, fig=None, ax=None, ylim: Union[List[float], None] = None) -> None:
        """Visualizes ICE plots.

        Args:
            ylim: The range of the y-axis. If not specified, the range of the ICE curves will be used.
        """
        if fig is None or ax is None:
            fig, ax = plt.subplots()
        sns.lineplot(
            x=self.target_var_name,
            y="ice",
            units="instance",
            data=self.df_ice,
            lw=0.8,
            alpha=0.3,
            estimator=None,
            zorder=1,
            ax=ax,
            color="black",
        )
        ax.set(xlabel=self.target_var_name, ylabel="Prediction", ylim=ylim)
        ax.grid(color="gray", linestyle="--", linewidth=0.5, alpha=0.5)
        fig.suptitle(f"Individual Conditional Expectation({self.target_var_name})")

    def plot_cice(
        self, fig=None, ax=None, ylim: Union[List[float], None] = None
    ) -> None:
        """Visualizes ICE (Individual Conditional Expectation) plots.

        Args:
            ylim: Tuple specifying the y-axis limits (min, max). If None, the limits are automatically determined based on the range of the ICE values.
        """
        if fig is None or ax is None:
            fig, ax = plt.subplots()
        sns.lineplot(
            x=self.target_var_name,
            y="ice_diff",
            units="instance",
            data=self.df_ice,
            lw=0.8,
            alpha=0.5,
            estimator=None,
            zorder=1,
            color="black",
            ax=ax,
        )
        average_ice = (
            self.df_ice.groupby(self.target_var_name)["ice_diff"].mean().reset_index()
        )
        sns.lineplot(
            x=self.target_var_name,
            y="ice_diff",
            data=average_ice,
            color="yellow",
            lw=5,
            ax=ax,
        )
        ax.set(xlabel=self.target_var_name, ylabel="Prediction", ylim=ylim)
        ax.grid(color="gray", linestyle="--", linewidth=0.5, alpha=0.5)
        fig.suptitle(f"Individual Conditional Expectation({self.target_var_name})")


class AccumulatedLocalEffects:
    """Accumulated Local Effects Plot(ALE)"""

    def __init__(
        self,
        model,
        X: pd.DataFrame,
    ):
        """
        Initializes the PartialDependence object.

        Args:
            model: Trained scikit-learn model.
            X: pandas DataFrame used for training.
        """
        self.model = model
        self.X = X.copy()

    def accumulated_local_effects(
        self, var_name: str, n_grid: int = 30
    ) -> tuple[np.ndarray, np.ndarray]:
        """Calculates Accumulated Local Effects (ALE).

        Args:
            var_name: Name of the feature for which to compute ALE.
            n_grid: Number of intervals to divide the feature range into.

        Returns:
            A tuple containing:
                - Feature values:  A NumPy array or list of the grid points along the feature dimension.
                - ALE values: A NumPy array or list of the corresponding ALE values for each grid point.
        """
        j = self.X.columns.get_loc(var_name)
        xjks = np.quantile(self.X.iloc[:, j], q=np.arange(0, 1, 1 / n_grid))
        local_effects = np.zeros(n_grid)
        for k in range(1, n_grid):
            mask = (self.X.iloc[:, j] >= xjks[k - 1]) & (self.X.iloc[:, j] <= xjks[k])
            local_effects[k] = self._predict_average(
                self.X[mask], j, xjks[k]
            ) - self._predict_average(self.X[mask], j, xjks[k - 1])

        accumulated_local_effects = np.cumsum(local_effects)
        return pd.DataFrame({var_name: xjks, "ale": accumulated_local_effects})

    def _predict_average(self, X: pd.DataFrame, j: int, xj: float) -> np.ndarray:
        """Replaces feature values, performs predictions, and averages the results.

        Args:
            j: The index of the feature to be replaced.
            xj: The value to replace the feature with.
        """

        X_replaced = X.copy()
        X_replaced.iloc[:, j] = xj
        return self.model.predict(X_replaced).mean()

    def plot_ale(self, ylim: Union[List[float], None] = None) -> None:
        """Visualizes Accumulated Local Effects (ALE) plots.

        Args:
            ylim: A tuple (ymin, ymax) specifying the y-axis limits. If None or omitted, the limits will be automatically determined based on the range of the ALE values.
        """

        fig, ax = plt.subplots()
        sns.lineplot(
            x=self.target_var_name,
            y="ice_diff",
            units="instance",
            data=self.df_ice,
            lw=0.8,
            alpha=0.5,
            estimator=None,
            zorder=1,
            ax=ax,
        )
        ax.set(xlabel=self.target_var_name, ylabel="Prediction", ylim=ylim)
        fig.suptitle(f"Individual Conditional Expectation({self.target_var_name})")
        fig.show()


class PermutationFeatureImportance:
    def __init__(
        self,
        model: Any,
        X: pd.DataFrame,
        y: pd.DataFrame,
        var_names: list[str],
        metric: Any,
        pred_type: Literal["regression", "classification"],
    ):
        self.model = model
        self.X = X
        self.y = y
        self.var_names = var_names
        self.metric = metric
        self.pred_type = pred_type
        if self.pred_type == "regression":
            self.baseline = self.metric(self.y, self.model.predict(self.X))
        elif hasattr(self.model, "predict_proba"):
            self.baseline = self.metric(self.y, self.model.predict_proba(self.X)[:, 1])
        else:
            raise AttributeError("Model does not have 'predict_proba' method.")

    def _permutation_metrics(
        self, idx_to_permute: int, X_permuted: pd.DataFrame
    ) -> float:
        """Calculates prediction accuracy after shuffling the values of a specific feature.

        Args:
            idx_to_permute: The index of the feature whose values will be shuffled.
        """

        X_permuted.iloc[:, idx_to_permute] = np.random.permutation(
            X_permuted.iloc[:, idx_to_permute]
        )
        if self.pred_type == "regression":
            y_pred = self.model.predict(X_permuted)
        else:
            y_pred = self.model.predict_proba(X_permuted)[:, 1]

        return self.metric(self.y, y_pred)

    def permutation_feature_importance(
        self, n_shuffle: int = 10, n_jobs: int = -1
    ) -> None:
        """Calculates Permutation Feature Importance (PFI).

        Args:
            n_shuffle: The number of times to shuffle the feature values.  Higher values generally lead to more stable results. Defaults to 10.
        """

        J = self.X.shape[1]

        results = []
        for j in range(J):
            X_permuted = self.X.copy()
            X_permuted.iloc[:, j] = np.random.permutation(X_permuted.iloc[:, j])
            metrics_permuted = Parallel(n_jobs=n_jobs)(
                delayed(self._permutation_metrics)(j, X_permuted)
                for _ in range(n_shuffle)
            )
            results.append(np.mean(metrics_permuted))
        df_feature_importance = pd.DataFrame(
            data={
                "var_name": self.var_names,
                "permutation": results,
            }
        )
        df_feature_importance["baseline"] = self.baseline
        df_feature_importance["difference"] = (
            df_feature_importance["permutation"] - df_feature_importance["baseline"]
        )
        df_feature_importance["ratio"] = (
            df_feature_importance["permutation"] / df_feature_importance["baseline"]
        )

        self.feature_importance = df_feature_importance.sort_values(
            "permutation", ascending=False
        )
