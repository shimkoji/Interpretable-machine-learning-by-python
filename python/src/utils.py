from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


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
            X: pandas DataFrame used for training (must be a DataFrame).
            var_names: List of feature names for partial dependence calculation.
        """
        self.model = model
        self.X = X.copy()
        self.var_names = var_names
        self.pred_type = pred_type

    def partial_dependence(self, var_name, n_grid=50):
        """
        Calculates the partial dependence for a given variable.

        Args:
            var_name: Name of the variable.
            n_grid: Number of grid points.

        Returns:
            A tuple containing the grid points and the average predictions.
        """

        if isinstance(self.X, pd.DataFrame):  # Check if it's a DataFrame
            value_range = np.linspace(
                self.X[var_name].min(), self.X[var_name].max(), num=n_grid
            )

        else:
            raise TypeError("X must be a pandas DataFrame.")

        average_prediction = np.array(
            [self._counterfactual_prediction(var_name, x).mean() for x in value_range]
        )

        return pd.DataFrame(
            data={var_name: value_range, "avg_pred": average_prediction}
        )

    def _counterfactual_prediction(self, var_name, grid_point):
        X_counterfactual = self.X.copy()
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
