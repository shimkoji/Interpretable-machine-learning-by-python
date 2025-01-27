from itertools import product
from typing import Any, Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from joblib import Parallel, delayed

# class PartialDependence:
#     def __init__(
#         self,
#         model,
#         X: pd.DataFrame,
#         var_names: list[str],
#         pred_type: Literal["regression", "classification"],
#     ):
#         """
#         Initializes the PartialDependence object.

#         Args:
#             model: Trained scikit-learn model.
#             X: pandas DataFrame used for training (must be a DataFrame).
#             var_names: List of feature names for partial dependence calculation.
#         """
#         self.model = model
#         self.X = X.copy()
#         self.var_names = var_names
#         self.pred_type = pred_type

#     def partial_dependence(self, var_name, n_grid=50):
#         """
#         Calculates the partial dependence for a given variable.

#         Args:
#             var_name: Name of the variable.
#             n_grid: Number of grid points.

#         Returns:
#             A tuple containing the grid points and the average predictions.
#         """

#         if isinstance(self.X, pd.DataFrame):  # Check if it's a DataFrame
#             value_range = np.linspace(
#                 self.X[var_name].min(), self.X[var_name].max(), num=n_grid
#             )

#         else:
#             raise TypeError("X must be a pandas DataFrame.")

#         average_prediction = np.array(
#             [self._counterfactual_prediction(var_name, x).mean() for x in value_range]
#         )

#         return pd.DataFrame(
#             data={var_name: value_range, "avg_pred": average_prediction}
#         )

#     def _counterfactual_prediction(self, var_name, grid_point):
#         X_counterfactual = self.X.copy()
#         X_counterfactual[var_name] = grid_point


#         if self.pred_type == "regression":
#             return self.model.predict(X_counterfactual)
#         else:
#             return self.model.predict_proba(X_counterfactual)[:, 1]
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

    def partial_dependence(self, var_names=None, n_grid=50):
        """
        Calculates the partial dependence for given variables.

        Args:
            var_names: List of variable names. If None, uses all variables specified during initialization.
            n_grid: Number of grid points per variable.

        Returns:
            A pandas DataFrame containing the grid points and average predictions.
        """

        if not isinstance(self.X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame.")

        if var_names is None:
            var_names = self.var_names
        elif isinstance(var_names, str):  #  handle single string input
            var_names = [var_names]

        grid_ranges = {}
        for var_name in var_names:
            grid_ranges[var_name] = np.linspace(
                self.X[var_name].min(), self.X[var_name].max(), num=n_grid
            )

        grid_points = list(
            product(*grid_ranges.values())
        )  # Cartesian product for all variables
        grid_points_df = pd.DataFrame(grid_points, columns=var_names)

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
        ) in grid_point_series.items():  # Iterate through variables
            X_counterfactual[var_name] = grid_point

        if self.pred_type == "regression":
            return self.model.predict(X_counterfactual)
        else:
            return self.model.predict_proba(X_counterfactual)[:, 1]


class IndividualConditionalExpectation(PartialDependence):
    """Individual Conditional Expectation"""

    def individual_conditional_expectation(
        self, var_name: str, ids_to_compute: list[int], n_grid: int = 50
    ) -> None:
        """ICEを求める

        Args:
            var_name:
                ICEを計算したい変数名
            ids_to_compute:
                ICEを計算したいインスタンスのリスト
            n_grid:
                グリッドを何分割するか
                細かすぎると値が荒れるが、粗すぎるとうまく関係をとらえられない
                デフォルトは50
        """

        self.target_var_name = var_name
        # 変数名に対応するインデックスをもってくる
        # var_index = self.var_names.index(var_name)

        # ターゲットの変数を、取りうる値の最大値から最小値まで動かせるようにする
        value_range = np.linspace(
            self.X[var_name].min(), self.X[var_name].max(), num=n_grid
        )
        # value_range = np.linspace(
        #     self.X.iloc[:, var_index].min(), self.X.iloc[:, var_index].max(), num=n_grid
        # )

        # インスタンスごとのモデルの予測値
        # PDの_counterfactual_prediction()をそのまま使っているので
        # 全データに対して予測してからids_to_computeに絞り込んでいるが
        # 本当は絞り込んでから予測をしたほうが速い
        individual_prediction = np.array(
            [
                self._counterfactual_prediction(var_name, x)[ids_to_compute]
                for x in value_range
            ]
        )

        # ICEをデータフレームとしてまとめる
        self.df_ice2 = (
            # ICEの値
            pd.DataFrame(data=individual_prediction, columns=ids_to_compute)
            # ICEで用いた特徴量の値。特徴量名を列名としている
            # .assign(**{var_name: value_range})
            # # 縦持ちに変換して完成
            # .melt(id_vars=var_name, var_name="instance", value_name="ice")
        )
        self.df_ice = (
            # ICEの値
            pd.DataFrame(data=individual_prediction, columns=ids_to_compute)
            # ICEで用いた特徴量の値。特徴量名を列名としている
            .assign(**{var_name: value_range})
            # 縦持ちに変換して完成
            .melt(id_vars=var_name, var_name="instance", value_name="ice")
        )
        min_ice_by_instance = self.df_ice.loc[
            self.df_ice.groupby("instance")[var_name].idxmin()
        ]
        # instanceと最小のiceを辞書に格納
        min_ice_dict = dict(
            zip(min_ice_by_instance["instance"], min_ice_by_instance["ice"])
        )

        # 各行のiceから対応するinstanceの最小のiceを引く
        self.df_ice["ice_diff"] = self.df_ice.apply(
            lambda row: row["ice"] - min_ice_dict[row["instance"]], axis=1
        )

        # ICEを計算したインスタンスについての情報も保存しておく
        # 可視化の際に実際の特徴量の値とその予測値をプロットするために用いる
        if self.pred_type == "regression":
            prediction = self.model.predict(self.X.iloc[ids_to_compute])
        else:
            prediction = self.model.predict_proba(self.X.iloc[ids_to_compute])[:, 1]
        self.df_instance = (
            # インスタンスの特徴量の値
            pd.DataFrame(data=self.X.iloc[ids_to_compute], columns=self.var_names)
            # インスタンスに対する予測値
            .assign(
                instance=ids_to_compute,
                prediction=prediction,
            )
            # 並べ替え
            .loc[:, ["instance", "prediction"] + self.var_names]
        )

    def plot(self, ylim: list[float] | None = None) -> None:
        """ICEを可視化

        Args:
            ylim: Y軸の範囲。特に指定しなければiceの範囲となる。
        """

        fig, ax = plt.subplots()
        # ICEの線
        # sns.lineplot(
        #     self.target_var_name,
        #     self.df_ice["ice"],
        #     units="instance",
        #     data=self.df_ice,
        #     lw=0.8,
        #     alpha=0.5,
        #     estimator=None,
        #     zorder=1,  # zorderを指定することで、線が背面、点が前面にくるようにする
        #     ax=ax,
        # )
        sns.lineplot(
            x=self.target_var_name,  # Correct: string representing column name
            y="ice",  # Correct: string representing column name
            units="instance",  # Correct: string representing column name
            data=self.df_ice,  # Crucial: provide the DataFrame
            lw=0.8,
            alpha=0.5,
            estimator=None,
            zorder=1,
            ax=ax,
        )
        # インスタンスからの実際の予測値を点でプロットしておく
        # sns.scatterplot(
        #     x=self.target_var_name,
        #     y="prediction",
        #     data=self.df_instance,
        #     zorder=2,
        #     ax=ax,
        # )
        ax.set(xlabel=self.target_var_name, ylabel="Prediction", ylim=ylim)
        fig.suptitle(f"Individual Conditional Expectation({self.target_var_name})")

        fig.show()

    def plot_cice(self, ylim: list[float] | None = None) -> None:
        """ICEを可視化

        Args:
            ylim: Y軸の範囲。特に指定しなければiceの範囲となる。
        """

        fig, ax = plt.subplots()
        # ICEの線
        # sns.lineplot(
        #     self.target_var_name,
        #     self.df_ice["ice"],
        #     units="instance",
        #     data=self.df_ice,
        #     lw=0.8,
        #     alpha=0.5,
        #     estimator=None,
        #     zorder=1,  # zorderを指定することで、線が背面、点が前面にくるようにする
        #     ax=ax,
        # )
        sns.lineplot(
            x=self.target_var_name,  # Correct: string representing column name
            y="ice_diff",  # Correct: string representing column name
            units="instance",  # Correct: string representing column name
            data=self.df_ice,  # Crucial: provide the DataFrame
            lw=0.8,
            alpha=0.5,
            estimator=None,
            zorder=1,
            ax=ax,
        )
        # 平均値の計算とプロット
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
            # label="Average ICE",  # 凡例用にラベルを追加
        )
        # インスタンスからの実際の予測値を点でプロットしておく
        # sns.scatterplot(
        #     x=self.target_var_name,
        #     y="prediction",
        #     data=self.df_instance,
        #     zorder=2,
        #     ax=ax,
        # )
        ax.set(xlabel=self.target_var_name, ylabel="Prediction", ylim=ylim)
        fig.suptitle(f"Individual Conditional Expectation({self.target_var_name})")

        fig.show()


class AccumulatedLocalEffects(PartialDependence):
    """Accumulated Local Effects Plot(ALE)"""

    def accumulated_local_effects(
        self, j: int, n_grid: int = 30
    ) -> tuple[np.ndarray, np.ndarray]:
        """ALEを求める

        Args:
            j: ALEを計算したい特徴量のインデックス
            n_grid: グリッドを何分割するか
        Returns:
            特徴量の値とその場合のALE
        """

        # ターゲットの変数を、取りうる値の最大値から最小値まで動かせるようにする
        xjks = np.quantile(self.X.iloc[:, j], q=np.arange(0, 1, 1 / n_grid))

        # 区間ごとに両端での予測値の平均的な差分を求める
        local_effects = np.zeros(n_grid)
        for k in range(1, n_grid):
            mask = (self.X.iloc[:, j] >= xjks[k - 1]) & (self.X.iloc[:, j] <= xjks[k])
            local_effects[k] = self._predict_average(
                self.X[mask], j, xjks[k]
            ) - self._predict_average(self.X[mask], j, xjks[k - 1])

        accumulated_local_effects = np.cumsum(local_effects)

        # return (xjks, accumulated_local_effects)
        return pd.DataFrame({"x": xjks, "ale": accumulated_local_effects})

    def _predict_average(self, X: pd.DataFrame, j: int, xj: float) -> np.ndarray:
        """特徴量の値を置き換えて予測を行い、結果を平均する

        Args:
            j: 値を置き換える特徴量のインデックス
            xj: 置き換える値
        """

        # 特徴量の値を置き換える際、元データが上書きされないようコピー
        # 特徴量の値を置き換えて予測し、平均をとって返す
        X_replaced = X.copy()
        X_replaced.iloc[:, j] = xj

        return self.model.predict(X_replaced).mean()

    def plot_ale(self, ylim: list[float] | None = None) -> None:
        """ALEを可視化

        Args:
            ylim: Y軸の範囲。特に指定しなければiceの範囲となる。
        """

        fig, ax = plt.subplots()
        # ICEの線
        # sns.lineplot(
        #     self.target_var_name,
        #     self.df_ice["ice"],
        #     units="instance",
        #     data=self.df_ice,
        #     lw=0.8,
        #     alpha=0.5,
        #     estimator=None,
        #     zorder=1,  # zorderを指定することで、線が背面、点が前面にくるようにする
        #     ax=ax,
        # )
        sns.lineplot(
            x=self.target_var_name,  # Correct: string representing column name
            y="ice_diff",  # Correct: string representing column name
            units="instance",  # Correct: string representing column name
            data=self.df_ice,  # Crucial: provide the DataFrame
            lw=0.8,
            alpha=0.5,
            estimator=None,
            zorder=1,
            ax=ax,
        )
        # 平均値の計算とプロット
        # average_ice = (
        #     self.df_ice.groupby(self.target_var_name)["ice_diff"].mean().reset_index()
        # )
        # sns.lineplot(
        #     x=self.target_var_name,
        #     y="ice_diff",
        #     data=average_ice,
        #     color="yellow",
        #     lw=5,
        #     ax=ax,
        #     # label="Average ICE",  # 凡例用にラベルを追加
        # )
        # インスタンスからの実際の予測値を点でプロットしておく
        # sns.scatterplot(
        #     x=self.target_var_name,
        #     y="prediction",
        #     data=self.df_instance,
        #     zorder=2,
        #     ax=ax,
        # )
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
        elif hasattr(self.model, "predict_proba"):  # Check if predict_proba exists
            self.baseline = self.metric(self.y, self.model.predict_proba(self.X)[:, 1])
        else:
            raise AttributeError("Model does not have 'predict_proba' method.")

    def _permutation_metrics(
        self, idx_to_permute: int, X_permuted: pd.DataFrame
    ) -> float:
        """ある特徴量の値をシャッフルしたときの予測精度を求める

        Args:
            idx_to_permute: シャッフルする特徴量のインデックス
        """

        # 特徴量の値をシャッフルして予測
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
        """PFIを求める

        Args:
            n_shuffle: シャッフルの回数。多いほど値が安定する。デフォルトは10回
        """

        J = self.X.shape[1]  # 特徴量の数

        # J個の特徴量に対してPFIを求めたい
        # R回シャッフルを繰り返して平均をとることで値を安定させている
        # metrics_permuted = [
        #     float(np.mean([self._permutation_metrics(j) for r in range(n_shuffle)]))
        #     for j in range(J)
        # ]

        results = []
        for j in range(J):
            # シャッフルする際に、元の特徴量が上書きされないよう用にコピーしておく
            X_permuted = self.X.copy()

            # 特徴量の値をシャッフル
            X_permuted.iloc[:, j] = np.random.permutation(X_permuted.iloc[:, j])

            # 並列処理でメトリクス計算
            metrics_permuted = Parallel(n_jobs=n_jobs)(
                delayed(self._permutation_metrics)(j, X_permuted)
                for _ in range(n_shuffle)
            )
            results.append(np.mean(metrics_permuted))
        # データフレームとしてまとめる
        # シャッフルでどのくらい予測精度が落ちるかは、
        # 差(difference)と比率(ratio)の2種類を用意する
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

    # def plot(self, importance_type: str = "difference") -> None:
    #     """PFIを可視化

    #     Args:
    #         importance_type: PFIを差(difference)と比率(ratio)のどちらで計算するか
    #     """

    #     fig, ax = plt.subplots()
    #     ax.barh(
    #         self.feature_importance["var_name"],
    #         self.feature_importance[importance_type],
    #         label=f"baseline: {self.baseline:.2f}",
    #     )
    #     ax.set(xlabel=importance_type, ylabel=None)
    #     ax.invert_yaxis()  # 重要度が高い順に並び替える
    #     ax.legend(loc="lower right")
    #     fig.suptitle(f"Permutationによる特徴量の重要度({importance_type})")

    #     fig.show()
