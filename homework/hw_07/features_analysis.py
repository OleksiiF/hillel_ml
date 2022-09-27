#!/usr/bin/env python
# -*- coding: utf-8 -*-
import shap
from lazypredict.Supervised import LazyClassifier
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn import preprocessing
from sklearn.datasets import load_digits
from sklearn.ensemble import ExtraTreesClassifier
import pandas as pd


class Analisator:
    def __init__(self, features_to_drop=None, pca_n=None):
        self.pca_n = pca_n
        self.data = load_digits()
        self.x_source = pd.DataFrame(
            self.data.data,
            columns=self.data.feature_names
        )
        self.y_source = self.data.target

        if features_to_drop:
            self.x_source.drop(columns=features_to_drop, axis=1, inplace=True)

        elif pca_n:
            new_source = self.get_pca_np()
            self.x_source = pd.DataFrame(
                new_source,
                columns=[f"pca_{i}" for i in range(self.pca_n)]
            )

        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            self.x_source.values,
            self.y_source,
            test_size=.2,
            random_state=13
        )

    def get_lazy(self):
        clf = LazyClassifier(verbose=1, ignore_warnings=True, custom_metric=None)
        models, predictions = clf.fit(
            self.x_train,
            self.x_test,
            self.y_train,
            self.y_test
        )
        print(models)

    def shap_explainer(self):
        clf = ExtraTreesClassifier(n_estimators=100, random_state=0)
        clf.fit(self.x_source.values, self.y_source)

        explainer = shap.TreeExplainer(clf)
        shap_values = explainer.shap_values(self.x_source)

        shap.summary_plot(
            shap_values,
            self.x_source,
            plot_type="bar",
            max_display=64,
            feature_names=self.x_source.columns
        )

    def get_pca_np(self):
        data_scaled = pd.DataFrame(
            preprocessing.scale(self.x_source),
            columns=self.x_source.columns
        )
        pca_transformer = PCA(n_components=self.pca_n, random_state=13)

        return pca_transformer.fit_transform(data_scaled)


if __name__ == '__main__':
    # analisator = Analisator([
    #     'pixel_2_7',
    #     'pixel_5_7',
    #     'pixel_1_0',
    #     'pixel_3_0',
    #     'pixel_3_7',
    #     'pixel_2_0',
    #     'pixel_6_0',
    #     'pixel_7_0',
    #     'pixel_5_0',
    #     'pixel_4_0',
    #     'pixel_4_7',
    #     'pixel_0_0',
    # ])
    analisator = Analisator(pca_n=13)
    analisator.shap_explainer()
    analisator.get_lazy()
