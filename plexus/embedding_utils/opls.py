from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
import numpy as np


class OPLS(BaseEstimator, RegressorMixin, TransformerMixin):
    """
    Orthogonal Projections to Latent Structures (O-PLS) Regressor.

    This class implements the O-PLS algorithm for regression tasks,
    following the scikit-learn estimator API.

    Parameters
    ----------
    n_components : int, default=1
        Number of predictive components to extract.

    Attributes
    ----------
    W_p_ : ndarray of shape (n_features, n_components)
        Predictive weights.
    P_p_ : ndarray of shape (n_features, n_components)
        Predictive loadings.
    T_p_ : ndarray of shape (n_samples, n_components)
        Predictive scores.
    C_p_ : ndarray of shape (n_components,)
        Regression coefficients for predictive components.
    W_o_ : ndarray of shape (n_features, n_components)
        Orthogonal weights.
    P_o_ : ndarray of shape (n_features, n_components)
        Orthogonal loadings.
    T_o_ : ndarray of shape (n_samples, n_components)
        Orthogonal scores.
    coef_ : ndarray of shape (n_features,)
        Coefficients of the linear model.
    intercept_ : float
        Intercept of the linear model.
    """
    def __init__(self, n_components=1):
        self.n_components = n_components

    def fit(self, X, y):
        """
        Fit the O-PLS model to the data.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training data.
        y : ndarray of shape (n_samples,)
            Target values.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        X, y = check_X_y(X, y, y_numeric=True)
        y = y.reshape(-1, 1)

        X_residual = X.copy()
        y_residual = y.copy()

        W_p_list, P_p_list, T_p_list, C_p_list = [], [], [], []
        W_o_list, P_o_list, T_o_list = [], [], []

        for _ in range(self.n_components):
            # Compute predictive weights (w_p)
            w_p = X_residual.T @ y_residual / (y_residual.T @ y_residual)
            w_p /= np.linalg.norm(w_p)
            # Compute predictive scores (t_p)
            t_p = X_residual @ w_p
            # Compute predictive loadings (p_p)
            p_p = X_residual.T @ t_p / (t_p.T @ t_p)
            # Compute regression coefficient (c_p)
            c_p = y_residual.T @ t_p / (t_p.T @ t_p)
            # Compute orthogonal weights (w_o)
            w_o = p_p - w_p * (w_p.T @ p_p)
            w_o /= np.linalg.norm(w_o)
            # Compute orthogonal scores (t_o)
            t_o = X_residual @ w_o
            # Compute orthogonal loadings (p_o)
            p_o = X_residual.T @ t_o / (t_o.T @ t_o)
            # Deflate X_residual
            X_residual -= t_o @ p_o.T
            # Deflate y_residual with predictive component only
            y_residual -= t_p * c_p

            # Store components
            W_p_list.append(w_p)
            P_p_list.append(p_p)
            T_p_list.append(t_p)
            C_p_list.append(c_p.item())
            W_o_list.append(w_o)
            P_o_list.append(p_o)
            T_o_list.append(t_o)

        # Convert lists to arrays
        self.W_p_ = np.column_stack(W_p_list)
        self.P_p_ = np.column_stack(P_p_list)
        self.T_p_ = np.column_stack(T_p_list)
        self.C_p_ = np.array(C_p_list)
        self.W_o_ = np.column_stack(W_o_list)
        self.P_o_ = np.column_stack(P_o_list)
        self.T_o_ = np.column_stack(T_o_list)

        # Compute regression coefficients
        self.coef_ = self.W_p_ @ np.linalg.inv(self.P_p_.T @ self.W_p_) @ self.C_p_
        self.coef_ = self.coef_.flatten()
        # Compute intercept
        self.intercept_ = y.mean() - X.mean(axis=0) @ self.coef_

        return self

    def transform(self, X, return_deflated_X=False):
        """
        Apply the dimensionality reduction on X.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Data to transform.
        return_deflated_X : bool, default=False
            If True, return the deflated X after removing orthogonal components.

        Returns
        -------
        T_p : ndarray of shape (n_samples, n_components)
            Predictive scores.
        T_o : ndarray of shape (n_samples, n_components)
            Orthogonal scores.
        X_deflated : ndarray of shape (n_samples, n_features)
            Deflated X, if return_deflated_X is True.
        """
        check_is_fitted(self, ['W_p_', 'W_o_', 'P_o_'])
        X = check_array(X)

        X_residual = X.copy()
        T_o_list = []

        # Compute orthogonal scores and deflate X
        for k in range(self.n_components):
            w_o = self.W_o_[:, k].reshape(-1, 1)
            p_o = self.P_o_[:, k].reshape(-1, 1)
            t_o = X_residual @ w_o
            T_o_list.append(t_o)
            X_residual -= t_o @ p_o.T

        T_o = np.column_stack(T_o_list)

        # Compute predictive scores
        W_p = self.W_p_
        T_p = X_residual @ W_p

        if return_deflated_X:
            return T_p, T_o, X_residual
        else:
            return T_p, T_o

    def fit_transform(self, X, y, return_deflated_X=False):
        """
        Fit the model and apply the dimensionality reduction on X.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training data.
        y : ndarray of shape (n_samples,)
            Target values.
        return_deflated_X : bool, default=False
            If True, return the deflated X after removing orthogonal components.

        Returns
        -------
        T_p : ndarray of shape (n_samples, n_components)
            Predictive scores.
        T_o : ndarray of shape (n_samples, n_components)
            Orthogonal scores.
        X_deflated : ndarray of shape (n_samples, n_features)
            Deflated X, if return_deflated_X is True.
        """
        self.fit(X, y)
        return self.transform(X, return_deflated_X=return_deflated_X)

    def predict(self, X):
        """
        Predict using the O-PLS model.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Samples.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Returns predicted values.
        """
        check_is_fitted(self, ['coef_', 'intercept_'])
        X = check_array(X)
        y_pred = X @ self.coef_ + self.intercept_
        return y_pred

    def score(self, X, y):
        """
        Return the coefficient of determination R^2 of the prediction.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples.
        y : ndarray of shape (n_samples,)
            True values for X.

        Returns
        -------
        score : float
            R^2 of self.predict(X) wrt. y.
        """
        from sklearn.metrics import r2_score
        y_pred = self.predict(X)
        return r2_score(y, y_pred)