import pandas as pd
from pandas import DataFrame
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from credit_risk_lab.shared.logging import setup_logger


class DataAnalyzer:
    """
    Provides simple but robust tools to analyze a credit-risk dataset:
      - dataset summary (missing, types, cardinality, examples)
      - identification of feature types
    """

    def __init__(self, df: DataFrame):
        """
        Store the DataFrame internally to simplify the API.
        """
        if df is None or df.empty:
            raise ValueError("DataAnalyzer requires a non-empty DataFrame.")

        self.df = df
        self.logger = setup_logger(name="data_analyzer")

    # ==========================================================
    # FONCTION 1 : SYNTHÈSE DU DATASET
    # ==========================================================
    def summarize_dataset(self, max_examples: int = 5) -> DataFrame:
        """Generate a detailed dataset summary with:
        - dtype
        - missing count + %
        - cardinality
        - sample examples
        """
        self.logger.info("Génération du résumé du dataset...")
        total_rows = len(self.df)
        summary_rows = []

        for col in self.df.columns:
            col_series = self.df[col]
            missing = col_series.isna().sum()
            missing_pct = round((missing / total_rows) * 100, 2)
            cardinality = col_series.nunique(dropna=True)
            col_type = col_series.dtype

            # Exemples
            unique_values = col_series.dropna().unique()
            examples = (
                unique_values[:max_examples]
                if col_series.dtype == "object" or col_series.dtype.name == "category"
                else sorted(unique_values[:max_examples])
            )

            summary_rows.append(
                {
                    "Column": col,
                    "Type": col_type,
                    "Missing": missing,
                    "% Missing": missing_pct,
                    "Cardinality": cardinality,
                    "Examples": examples,
                }
            )

        summary_df = pd.DataFrame(summary_rows).sort_values(
            "% Missing", ascending=False
        )

        self.logger.info(
            f"Résumé terminé : {len(summary_df)} colonnes, {total_rows} lignes."
        )
        self.logger.debug("Top colonnes les plus incomplètes:\n%s", summary_df.head(10))

        return summary_df

    # ==========================================================
    # FONCTION 2 : IDENTIFICATION DES TYPES DE VARIABLES
    # ==========================================================
    def identify_feature_types(self):
        """Identify numerical and categorical columns."""

        if self.df is None or self.df.empty:
            self.logger.warning("Le DataFrame est vide ou non défini.")
            return [], []

        numeric_cols = self.df.select_dtypes(include=["number"]).columns.tolist()
        categorical_cols = self.df.select_dtypes(
            include=["object", "category", "bool"]
        ).columns.tolist()

        # Cas particulier datetime
        datetime_cols = self.df.select_dtypes(include=["datetime"]).columns.tolist()

        if datetime_cols:
            self.logger.info(
                f"{len(datetime_cols)} colonnes datetime détectées : {datetime_cols}"
            )

        self.logger.info("Identification des types de variables terminée.")
        self.logger.info(
            f"Numériques ({len(numeric_cols)}): {numeric_cols[:5]}{' ...' if len(numeric_cols) > 5 else ''}"
        )
        self.logger.info(
            f"Catégorielles ({len(categorical_cols)}): {categorical_cols[:5]}{' ...' if len(categorical_cols) > 5 else ''}"
        )

        return numeric_cols, categorical_cols

    # ==========================================================
    # FONCTION 3 : DISTRIBUTION DE LA VARIABLE CIBLE
    # ==========================================================
    def plot_target_distribution(self, target_col: str = "loan_status") -> None:
        """
        Affiche côte à côte :
        - un graphique à barres (distribution des catégories),
        - un graphique en secteurs (répartition proportionnelle),
        pour analyser la variable cible (target) de manière synthétique.

        Args
        ----
        df : pd.DataFrame
            Le DataFrame contenant les données.
        target_col : str
            Le nom de la colonne cible catégorielle à analyser.

        Exemple
        --------
        >>> plot_target_distribution("diagnosed_diabetes") or plot_target_distribution()
        """
        # Vérification
        if target_col not in self.df.columns:
            self.logger.error(f"Colonne '{target_col}' introuvable.")
            raise ValueError(
                f"La colonne '{target_col}' n'existe pas dans le DataFrame."
            )

        # Résumé
        value_counts = self.df[target_col].value_counts(dropna=False)
        percent = round(value_counts / len(self.df) * 100, 2)
        summary_df = pd.DataFrame(
            {
                "category": value_counts.index.astype(str),
                "count": value_counts.values,
                "percent": percent.values,
            }
        )

        # 🎨 Palette à 2 couleurs MAX pour un target binaire
        unique_categories = summary_df["category"].tolist()

        # Exemple palette binaire cohérente
        base_palette = ["#1f77b4", "#ff7f0e"]  # bleu et orange

        # Mapping catégorie → couleur (2 max)
        color_map = {cat: base_palette[i] for i, cat in enumerate(unique_categories)}

        # Subplots
        fig = make_subplots(
            rows=1,
            cols=2,
            subplot_titles=(
                f"Répartition des catégories pour '{target_col}'",
                f"Distribution proportionnelle de '{target_col}'",
            ),
            specs=[[{"type": "bar"}, {"type": "domain"}]],
            column_widths=[0.6, 0.4],
        )

        # --- Bar chart
        bar_fig = px.bar(
            summary_df.sort_values("count", ascending=False),
            x="category",
            y="count",
            text="percent",
            color="category",
            color_discrete_map=color_map,
        )
        for trace in bar_fig.data:
            fig.add_trace(trace, row=1, col=1)

        # --- Pie chart
        pie_fig = px.pie(
            summary_df,
            names="category",
            values="count",
            color="category",
            color_discrete_map=color_map,
        )
        for trace in pie_fig.data:
            fig.add_trace(trace, row=1, col=2)

        # Mise en forme
        fig.update_traces(
            texttemplate="%{text:.2f}%", textposition="outside", row=1, col=1
        )
        fig.update_layout(
            title_text=f"Distribution de la variable cible : '{target_col}'",
            showlegend=True,
            template="plotly_white",
            height=500,
            width=1000,
            title_x=0.5,
            legend_title_text="Catégories",
        )

        fig.update_xaxes(title_text="Catégorie", row=1, col=1)
        fig.update_yaxes(title_text="Nombre d'observations", row=1, col=1)

        fig.show()
        self.logger.info(
            f"Visualisation complète affichée pour la cible '{target_col}'."
        )
