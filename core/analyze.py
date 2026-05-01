import statistics
import os
import sys
import pandas as pd

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))

if project_root not in sys.path:
    sys.path.insert(0, project_root)
    
from data.load_data import Load_data
from helpers.validate import Validate
from helpers.sanity_check import Sanity_Check
from helpers.handle_missing_values import Imputer
from train_test_split.split import Splitter
from feature_scaling.preprocessing import Preprocessor
# Importing models
from models.OLS import OLS
from models.OLS_GD import OLS_Grad
from models.LR import _LinearRegression
from models.LASSO import Lasso
from models.RIDGE import Ridge
# Metrics
from evaluation_metrics.metrics import Get_metrics
# Analyze
from analysis.performance import Performance
from analysis.residual import Resid_insights
from analysis.convergence import Converge_insights
from analysis.coefficients import Correlation
from analysis.multicollinearity import Multicollinearity
from analysis.sensitivity import Sensitivity
# Plotting classes
from visualization.convergence import Convergence_plot
from visualization.coefficients import Coefficient_plot
from visualization.correlation import Correlation_plot
from visualization.performance import Performance_plot
from visualization.residuals import Residual_plot
from visualization.sensitivity import Sensitivity_plot

class Analyzer: 
    def __init__(self, show_plots = True):
        self.imputer = Imputer()
        self.splitter = Splitter()
        self.preprocessor = Preprocessor(perform_scaling=True)
        self.metrics = Get_metrics()
        self.performance = Performance()
        self.residuals = Resid_insights()
        self.convergence = Converge_insights()
        self.correlation = Correlation()
        self.multicollinearity = Multicollinearity()
        self.sensitivity = Sensitivity()
        self.convergence_plot = Convergence_plot()
        self.coefficient_plot = Coefficient_plot()
        self.correlation_plot = Correlation_plot()
        self.performance_plot = Performance_plot()
        self.residuals_plot = Residual_plot()
        self.sensitivity_plot = Sensitivity_plot()
        self.is_fitted = False
        self.results = None
        self.model_results = None
        self.mse_gd = None
        self.show_plots = show_plots
    
    def run(self, data, target):
        # Load the data
        loader = Load_data()
        loader.load_data(data=data, target=target)

        # Validating and checking the data
        Validate(loader).validate()
        Sanity_Check(loader).check()
        
        # Train test split
        X_train, X_test, y_train, y_test = self.splitter.split(loader.X_df, loader.y_df)
        
        # Imputing if missing values exist
        if X_train.isna().sum().sum() > 0:
            self.imputer.fit(X_train)
            self.imputer.transform(X_train)
            
        if X_train.isna().sum().sum() > 0:
            X_test = self.imputer.transform(X_test)
        
        # Scaler 
        self.preprocessor.fit(X_train)
        X_train_scaled = self.preprocessor.transform(X_train)
        X_test_scaled = self.preprocessor.transform(X_test)

        # # Model object creation
        ols = OLS(fit_intercept=True)
        ols_gd = OLS_Grad(fit_intercept=True, learning_rate=0.01)
        lr = _LinearRegression(fit_intercept=True)
        ridge = Ridge(fit_intercept=True)
        lasso = Lasso(fit_intercept=True)

        # # Fitting 
        ols.fit(X_train, y_train)
        ols_gd.fit(X_train_scaled, y_train)
        lr.fit(X_train_scaled, y_train)
        ridge.fit(X_train_scaled, y_train)
        lasso.fit(X_train_scaled, y_train)
        
        coef_dict_unscaled_all_models = {}
        coef_dict_unscaled_all_models['OLS'] = ols.coefficients
        coef_dict_unscaled_all_models['OLS-GD'] = ols_gd.coefficients
        coef_dict_unscaled_all_models['LASSO'] = lasso.coefficients
        coef_dict_unscaled_all_models['RIDGE'] = ridge.coefficients
        coef_dict_unscaled_all_models['LR'] = lr.coefficients

        # # Predicting
        y_pred_ols = ols.predict(X_test)
        y_pred_gd = ols_gd.predict(X_test_scaled)
        y_pred_lasso = lasso.predict(X_test_scaled)
        y_pred_ridge = ridge.predict(X_test_scaled)
        y_pred_lr = lr.predict(X_test_scaled)
        
        self.model_results = {
            "LinearRegression": {
                "actual": y_test,
                "predicted": y_pred_lr,
                "residuals": y_test - y_pred_lr
            },
            "OLS": {
                "actual": y_test,
                "predicted": y_pred_ols,
                "residuals": y_test - y_pred_ols
            },
            "GradientDescentOLS": {
                "actual": y_test,
                "predicted": y_pred_gd,
                "residuals": y_test - y_pred_gd
            },
            "Lasso": {
                "actual": y_test,
                "predicted": y_pred_lasso,
                "residuals": y_test - y_pred_lasso
            },
            "Ridge": {
                "actual": y_test,
                "predicted": y_pred_ridge,
                "residuals": y_test - y_pred_ridge
            }
        }

        # MSE
        mse_ols_test = self.metrics.mse(y_test, y_pred_ols)
        mse_gd_test = self.metrics.mse(y_test, y_pred_gd)
        mse_lr_test = self.metrics.mse(y_test, y_pred_lr)
        mse_lasso_test = self.metrics.mse(y_test, y_pred_lasso)
        mse_ridge_test = self.metrics.mse(y_test, y_pred_ridge)
        # R2
        r2_ols_test = self.metrics.r2(y_test, y_pred_ols)
        r2_gd_test = self.metrics.r2(y_test, y_pred_gd)
        r2_lr_test = self.metrics.r2(y_test, y_pred_lr)
        r2_lasso_test = self.metrics.r2(y_test, y_pred_lasso)
        r2_ridge_test = self.metrics.r2(y_test, y_pred_ridge)
        
        mse_test_dict = {
            'MSE OLS': mse_ols_test,
            'MSE Gradient descent': mse_gd_test,
            'MSE Linear regression': mse_lr_test,
            'MSE Lasso': mse_lasso_test,
            'MSE Ridge': mse_ridge_test
        }

        r2_test_dict = {
            'R2 OLS': r2_ols_test,
            'R2 Gradient descent': r2_gd_test,
            'R2 Linear regression': r2_lr_test,
            'R2 Lasso': r2_lasso_test,
            'R2 Ridge': r2_ridge_test
        }

        # Performance
        mse_insights = self.performance.compare_mse(mse_test_dict)
        r2_insights = self.performance.compare_r2(r2_test_dict)

        residuals_dict = {
            'Residuals OLS': y_test - y_pred_ols,
            'Residuals Gradient descent': y_test - y_pred_gd,
            'Residuals Linear regression': y_test - y_pred_lr,
            'Residuals Lasso': y_test - y_pred_lasso,
            'Residuals Ridge': y_test - y_pred_ridge
        }
        
        mean_residuals_dict = {k: statistics.mean(v) for k, v in residuals_dict.items()}
        variance_residuals_dict = {k: statistics.variance(v) for k, v in residuals_dict.items()}

        # Residual Insights
        resid_insights = self.residuals.analyze_residuals(mean_residuals_dict)

        # # Convergence 
        ols_gd_unscaled = OLS_Grad(fit_intercept=True, learning_rate=0.01)
        ols_gd_unscaled.fit(X_train, y_train)
        
        self.mse_gd = {
            'scaled-data-mse': ols_gd.cost_history,
            'unscaled-data-mse': ols_gd_unscaled.cost_history
        }
        
        # Convergence Analysis
        convergence_insights = self.convergence.analyze_convergence(
            cost_history_scaled=ols_gd.cost_history,
            cost_history_unscaled=ols_gd_unscaled.cost_history
        )
        
        # Coefficients
        coefficients_results, coefficients_insights = self.correlation.analyze_correlation(loader.dataframe, target)
        coeff_series = pd.Series(ols.coefficients, index=loader.feature_names) # we are taking OLS's coefficients for now
        coefficients_corr_insights = self.correlation.compare_corr_vs_coeff(coeffs=coeff_series) 

        # Multicollinearity
        corr_matrix = loader.X_df.corr()
        multicorr_insights = self.multicollinearity.analyze_multicollinearity(corr_matrix=corr_matrix)

        # Sensitivity 
        coef_scaled_data_dict = dict(zip(loader.feature_names, ols_gd.coefficients))
        coef_unscaled_data_dict = dict(zip(loader.feature_names, ols_gd_unscaled.coefficients))
        top_feat_dict = {
            'unscaled': coef_unscaled_data_dict,
            'scaled': coef_scaled_data_dict
        }
        # Analysis
        self.strong_features = self.sensitivity.check_coefficients(coefficients_dict=top_feat_dict, top_features=3)

        all_insights = []

        all_insights.extend(mse_insights)
        all_insights.extend(r2_insights)
        all_insights.extend(resid_insights)
        all_insights.extend(convergence_insights)
        all_insights.extend(coefficients_insights)
        all_insights.extend(coefficients_corr_insights)
        all_insights.extend(multicorr_insights)

        best_model = min(mse_test_dict, key=mse_test_dict.get)
        summary = [
            f"{best_model} performs best overall based on MSE",
            "Scaling improves gradient descent convergence",
        ]
        if multicorr_insights:
            summary.append("Multicollinearity affects feature relationships")
            
        # Returnable insights dict
        final_dict = {
            'metrics': {
                'values': {
                    'mse': mse_test_dict,
                    'r2': r2_test_dict,
                },
                'insights': {
                    'mse - Insights': mse_insights,
                    'r2 - Insights': r2_insights
                }
            },
            'residuals': {
                'values': {
                    'mean': mean_residuals_dict,
                    'variance': variance_residuals_dict,
                    'residuals': residuals_dict
                },
                'insights': resid_insights
            },
            'convergence': {
                'values': {
                    'cost-history-scaled': ols_gd.cost_history,
                    'cost-history-unscaled': ols_gd_unscaled.cost_history
                },
                'insights': convergence_insights
            },
            'coefficients': {
                'values': {
                    'scaled': coef_scaled_data_dict,
                    'unscaled': coef_unscaled_data_dict,
                    'coef_dict_unscaled_all_models': coef_dict_unscaled_all_models,
                },
                'insights': {
                    'coefficient-insights': coefficients_insights,
                    'coefficient-correlation-insights': coefficients_corr_insights
                }
            },
            'multicollinearity': {
                'values' : corr_matrix,
                'insights': multicorr_insights
            },
            'insights': {
                'summary': summary,
                'detailed-insights': all_insights 
            }
        }
        
        self.is_fitted = True
        self.results = final_dict
        return final_dict
    
    def _check_fitted(self):
        if not self.is_fitted:
            raise RuntimeError("Analyzer has not been run. Call run() first.")
    
    def get_results(self):
        self._check_fitted()
        return self.results
    
    def plot(self):
        self._check_fitted()
        
        self.performance_plot.plot(
            performance_analysis_result=self.results['metrics']['values']['mse']
        )
        
        self.convergence_plot.plot(
            convergence_history=self.results['convergence']['values']['cost-history-scaled'],
            model_name='Gradient Descent'
        )
        
        self.coefficient_plot.plot(
            coef_dict=self.results['coefficients']['values']['coef_dict_unscaled_all_models']
        )
        
        self.correlation_plot.plot(
            corr_matrix=self.results['multicollinearity']['values']
        )
        
        self.residuals_plot.plot(
            self.model_results['OLS']['actual'], self.model_results['OLS']['predicted'], model_name="Linear Regression"
        )
        
        self.sensitivity_plot.plot_mse(
            self.mse_gd
        )