import pandas as pd


LATEX_BASE_DIR = "/home/nkuechen/Documents/Thesis/latex/Bachelor Thesis"

UNDERSCORE = "\\_"
UPARROW = "\\uparrow"
DOWNARROW = "\\downarrow"

HLINE = "\\hline\n"


class LatexTable:
    """A class for comparing experiments in latex tables."""

    df: pd.DataFrame
    HEADER = f"\\begin{{table}}[H]\n    \\centering\n"

    def __init__(
        self,
        df: pd.DataFrame,
    ) -> None:
        """The initializer saving the DataFrame to compare.

        :param df: The DataFrame containing the experiment run information.
        """
        self.df = df

    def _compare_groups(
        self,
        group: int,
        group_df: pd.DataFrame,
        metrics: list[tuple[str]],
        add_model_type: bool = False,
    ):
        """Compares a subgroup of experiments

        :param group: The value expression for the group
        :param group_df: The DataFrame containing the group
        :param metrics: A list of tuples containing metrics that should be compared with their bool ascending indicator
        :param add_model_type: Whether or not to add the model types to each comparison, defaults to False
        :return: The generated comparison of the subgroup
        """
        return_str = ""
        best = {"aug": {}, "no_aug": {}}

        for aug in ["aug", "no_aug"]:
            best[aug] = {}
            for metric, ascending in metrics:
                best[aug][metric] = {}

                if aug == "aug":
                    subgroup = group_df["params.n_aug"] != 0
                else:
                    subgroup = group_df["params.n_aug"] == 0

                best[aug][metric]["metric"] = (
                    group_df[subgroup]
                    .sort_values(by=metric, ascending=ascending)[metric]
                    .iloc[0]
                )
                best[aug][metric]["n_aug"] = (
                    group_df[subgroup]
                    .sort_values(by=metric, ascending=ascending)["params.n_aug"]
                    .iloc[0]
                )
                best[aug][metric]["model"] = (
                    group_df[subgroup]
                    .sort_values(by=metric, ascending=ascending)["params.model_class"]
                    .iloc[0]
                )

        for metric, ascending in metrics:
            best["aug"][metric]["impr"] = (
                best["aug"][metric]["metric"] / best["no_aug"][metric]["metric"] * 100
            ) - 100
            best["no_aug"][metric]["impr"] = (
                best["no_aug"][metric]["metric"] / best["aug"][metric]["metric"] * 100
            ) - 100

        for aug in ["aug", "no_aug"]:
            if aug == "aug":
                return_str += f"        {HLINE}"
            return_str += f"        {group} & "
            for i, metric in enumerate([metric_tuple[0] for metric_tuple in metrics]):
                # Add model type
                if add_model_type:
                    return_str += f"{best[aug][metric]['model'].replace('_', UNDERSCORE)} & "

                # Add n_aug
                return_str += f"{best[aug][metric]['n_aug']} & "

                # Add metric
                if metric in ["metrics.mdt", "metrics.med_dt"]:
                    if best[aug][metric]["impr"] > 0:
                        return_str += "\\textbf{{{metric:.1f}}} ".format(
                            metric=best[aug][metric]["metric"]
                        )
                    else:
                        return_str += f"{best[aug][metric]['metric']:.1f} "
                elif metric == "metrics.c_index_ipcw":
                    if best[aug][metric]["impr"] > 0:
                        return_str += "\\textbf{{{metric:.3f}}} ".format(
                            metric=best[aug][metric]["metric"]
                        )
                    else:
                        return_str += f"{best[aug][metric]['metric']:.3f} "
                else:
                    if best[aug][metric]["impr"] < 0:
                        return_str += "\\textbf{{{metric:.3f}}} ".format(
                            metric=best[aug][metric]["metric"]
                        )
                    else:
                        return_str += f"{best[aug][metric]['metric']:.3f} "

                # Add improvement
                if aug == "aug":
                    plus_minus = "$\\pm$"
                    return_str += f"({'+' if (best[aug][metric]['impr'] > 0) else plus_minus if best[aug][metric]['impr'] == 0 else ''}"
                    return_str += f"{best[aug][metric]['impr']:.1f}~\\%) "

                if i != len(metrics) - 1:
                    return_str += "& "
            return_str += " \\\\\n"

        return return_str

    def _generate_column_definition(
        self, group_by: str, by_metrics: tuple[str, bool], add_model_type: bool
    ) -> str:
        """Generates the definitions for the columns.

        :param group_by: The column the comparison should be grouped by.
        :param by_metrics: A list of tuples containing metrics that should be compared with their bool ascending indicator
        :param add_model_type: Whether or not to add the model type to the columns
        :return: The generated code for the column definition
        """
        return_str = "        \\begin{tabular}{| c "
        for _ in by_metrics:
            return_str += f"| {'l  'if add_model_type else ''}c  l |"
        return_str += "}\n"

        return_str += f"        {HLINE}"

        return_str += (
            f"        \\textit{{{group_by.split('.')[1].replace('_', UNDERSCORE)}}} & "
        )
        for metric, ascending in by_metrics:
            if add_model_type:
                return_str += "Modelltyp & "
            return_str += "\\textit{n\\_aug} & "
            return_str += f"\\textit{{\\gls{{{metric.split('.')[1]}}}}} ${DOWNARROW if ascending else UPARROW}$ "
            if by_metrics[-1][0] == metric:
                return_str += "\\\\\n"
            else:
                return_str += "& "
        return_str += f"        {HLINE}"
        return return_str

    def compare_best_by_metrics_grouped(
        self,
        by_metrics: list[tuple[str, bool]],
        group_by: str,
        model_class: str | None = None,
        add_model_type: bool = False,
        caption: str | None = None,
        label: str | None = None,
    ) -> str:
        """Method to compare the best models, sorted by specified metrics and grouped by another column

        :param by_metrics: A list of tuples containig metric names in the DataFrame and a bool
            indicating if they need to be sorted ascending
        :param group_by: The other column the comparison should be grouped by (e.g. `params.n_aug`)
        :param model_class: The model class to compare, if this is None it will compare the runs of all models, defaults to None
        :param add_model_type: Whether or not to add the type of the model to each row, defaults to False
        :param caption: The caption of the table, defaults to None
        :param label: The label of the table, defaults to None
        :return: The generated latex code for the table
        """
        if model_class is not None:
            tmp_df = self.df[self.df["params.model_class"] == model_class]
        else:
            tmp_df = self.df.copy()
        latex_str = self.HEADER
        latex_str += self._generate_column_definition(group_by, by_metrics, add_model_type)

        for group, group_df in tmp_df.groupby(by=group_by):
            latex_str += self._compare_groups(
                group, group_df, by_metrics, add_model_type
            )
        latex_str += f"        {HLINE}"

        latex_str += "        \\end{tabular}\n"

        # Replace decimal dot with comma
        latex_str = latex_str.replace(".", ",")

        if caption is not None:
            latex_str += f"    \\caption{{{caption}}}\n"
        if label is not None:
            latex_str += f"    \\label{{tab:{label}}}\n"
        latex_str += "\\end{table}\n"

        return latex_str


if __name__ == "__main__":
    test_df = pd.read_csv("../data/cross_validation/surv_cross_validation.csv")
    tab = LatexTable(test_df)
    latex_str = tab.compare_best_by_metrics_grouped(
        [("metrics.c_index_ipcw", False), ("metrics.ibs", True)],
        "params.n_dev",
        caption="Vergleich der besten Metriken für \gls{cph}-Modelle.",
        label="cv_cph",
    )
    print(latex_str)
