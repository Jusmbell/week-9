import pandas as pd
import numpy as np


class GroupEstimate:
    """
    A class that estimates values based on categorical groupings.
    
    Parameters
    ----------
    estimate : str
        Either "mean" or "median" to determine the estimation method.
    """
    
    def __init__(self, estimate):
        """
        Initialize the GroupEstimate class.
        
        Parameters
        ----------
        estimate : str
            Either "mean" or "median"
        """
        if estimate not in ["mean", "median"]:
            raise ValueError("estimate must be either 'mean' or 'median'")
        self.estimate = estimate
        self.group_estimates_ = None
        self.columns_ = None
        self.default_category_ = None
    
    def fit(self, X, y, default_category=None):
        """
        Fit the model by calculating group estimates.
        
        Parameters
        ----------
        X : pandas DataFrame
            Categorical data
        y : array-like
            Continuous values corresponding to X
        default_category : str, optional
            Column name to use as fallback when combination is missing
        
        Returns
        -------
        self
        """
        # Convert X to DataFrame if needed
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        
        # Store column names for later use
        self.columns_ = X.columns.tolist()
        self.default_category_ = default_category
        
        # Combine X and y into a single DataFrame
        df = X.copy()
        df['_target_'] = y
        
        # Group by all columns in X and calculate estimates
        if self.estimate == "mean":
            self.group_estimates_ = df.groupby(self.columns_)['_target_'].mean()
        else:  # median
            self.group_estimates_ = df.groupby(self.columns_)['_target_'].median()
        
        # If default_category is specified, also store those estimates
        if default_category is not None:
            if default_category not in self.columns_:
                raise ValueError(f"default_category '{default_category}' not found in X columns")
            
            if self.estimate == "mean":
                self.default_estimates_ = df.groupby(default_category)['_target_'].mean()
            else:  # median
                self.default_estimates_ = df.groupby(default_category)['_target_'].median()
        
        return self
    
    def predict(self, X_):
        """
        Predict estimates for new observations.
        
        Parameters
        ----------
        X_ : array-like or pandas DataFrame
            New observations to predict
        
        Returns
        -------
        numpy array
            Predicted estimates
        """
        if self.group_estimates_ is None:
            raise ValueError("Model has not been fitted yet. Call .fit() first.")
        
        # Convert X_ to DataFrame if needed
        if not isinstance(X_, pd.DataFrame):
            X_ = pd.DataFrame(X_, columns=self.columns_)
        
        # Ensure columns match
        if X_.columns.tolist() != self.columns_:
            X_.columns = self.columns_
        
        predictions = []
        missing_count = 0
        
        for idx, row in X_.iterrows():
            # Create a tuple for indexing into the grouped estimates
            if len(self.columns_) == 1:
                key = row[self.columns_[0]]
            else:
                key = tuple(row[self.columns_])
            
            try:
                # Try to get the estimate for this combination
                prediction = self.group_estimates_.loc[key]
                predictions.append(prediction)
            except KeyError:
                # Combination not found
                if self.default_category_ is not None:
                    # Try to use default category
                    default_key = row[self.default_category_]
                    try:
                        prediction = self.default_estimates_.loc[default_key]
                        predictions.append(prediction)
                    except KeyError:
                        # Even default category not found
                        predictions.append(np.nan)
                        missing_count += 1
                else:
                    # No default category specified
                    predictions.append(np.nan)
                    missing_count += 1
        
        if missing_count > 0:
            print(f"Warning: {missing_count} observation(s) with missing group(s)")
        
        return np.array(predictions)