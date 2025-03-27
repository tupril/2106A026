#2106A026
import sys
import numpy as np
import pandas as pd
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                        QHBoxLayout, QTabWidget, QPushButton, QLabel,
                            QComboBox, QFileDialog, QSpinBox, QDoubleSpinBox,
                            QGroupBox, QScrollArea, QTextEdit, QStatusBar,
                            QProgressBar, QCheckBox, QGridLayout, QMessageBox,
                            QDialog, QLineEdit)
from PyQt6.QtCore import Qt
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from sklearn import datasets, preprocessing, model_selection
from sklearn.linear_model import LinearRegression, LogisticRegression, SGDRegressor, SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (accuracy_score, mean_squared_error, confusion_matrix,r2_score)
from sklearn.impute import SimpleImputer

import tensorflow as tf
from tensorflow import keras
from keras import layers, models, optimizers

class MLCourseGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Machine Learning Course GUI")
        self.setGeometry(100, 100, 1400, 800)
        
        # Initializes main widget and layout
        self.main_widget = QWidget()
        self.setCentralWidget(self.main_widget)
        self.layout = QVBoxLayout(self.main_widget)
        
        # Initializes data containers
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.current_model = None
        
        # For deep learning
        self.layer_config = []
        
        # Creates components
        self.create_data_section()
        self.create_tabs()
        self.create_visualization()
        self.create_status_bar()
    
    
    # Data Loading / Preprocessing
    
    def create_data_section(self):
        """Create the data loading and preprocessing section"""
        data_group = QGroupBox("Data Management")
        data_layout = QHBoxLayout()
        
        # Dataset selection
        self.dataset_combo = QComboBox()
        self.dataset_combo.addItems([
            "Load Custom Dataset",
            "Iris Dataset",
            "Breast Cancer Dataset",
            "Digits Dataset",
            "Boston Housing Dataset",
            "MNIST Dataset"
        ])
        self.dataset_combo.currentIndexChanged.connect(self.load_dataset)
        
        # Data loading button
        self.load_btn = QPushButton("Load Data")
        self.load_btn.clicked.connect(self.load_custom_data)
        
        # Preprocessing options (Scaling)
        self.scaling_combo = QComboBox()
        self.scaling_combo.addItems([
            "No Scaling",
            "Standard Scaling",
            "Min-Max Scaling",
            "Robust Scaling"
        ])

        # Missing-data handling options
        self.missing_combo = QComboBox()
        self.missing_combo.addItems([
            "No Handling",
            "Mean Imputation",
            "Interpolation",
            "Forward Fill",
            "Backward Fill"
        ])
        
        # Train-test split options
        self.split_spin = QDoubleSpinBox()
        self.split_spin.setRange(0.1, 0.9)
        self.split_spin.setValue(0.2)
        self.split_spin.setSingleStep(0.1)
        
        # Add widgets to layout
        data_layout.addWidget(QLabel("Dataset:"))
        data_layout.addWidget(self.dataset_combo)
        data_layout.addWidget(self.load_btn)
        data_layout.addWidget(QLabel("Missing Data:"))
        data_layout.addWidget(self.missing_combo)
        data_layout.addWidget(QLabel("Scaling:"))
        data_layout.addWidget(self.scaling_combo)
        data_layout.addWidget(QLabel("Test Split:"))
        data_layout.addWidget(self.split_spin)
        
        data_group.setLayout(data_layout)
        self.layout.addWidget(data_group)

    def load_dataset(self):
        """Load selected built-in dataset"""
        try:
            dataset_name = self.dataset_combo.currentText()
            if dataset_name == "Load Custom Dataset":
                return
            
            if dataset_name == "MNIST Dataset":
                (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
                # Flatten images from 28x28 to 784
                X_train = X_train.reshape((X_train.shape[0], -1)).astype('float32')
                X_test = X_test.reshape((X_test.shape[0], -1)).astype('float32')
                
                # Convert to DataFrame for missing-data operations (though MNIST has no missing)
                df_train = pd.DataFrame(X_train)
                df_test = pd.DataFrame(X_test)
                
                # Apply missing data handling if needed (no real missing data in MNIST, but we keep the flow consistent)
                df_train = self.handle_missing_data(df_train)
                df_test = self.handle_missing_data(df_test)
                
                # Convert back to numpy
                self.X_train = df_train.values
                self.X_test = df_test.values
                self.y_train = y_train
                self.y_test = y_test
                
                # Scale if requested
                self.apply_scaling()
                self.status_bar.showMessage(f"Loaded {dataset_name}")
                return
            
            # Load scikit-learn dataset
            if dataset_name == "Iris Dataset":
                data = datasets.load_iris()
            elif dataset_name == "Breast Cancer Dataset":
                data = datasets.load_breast_cancer()
            elif dataset_name == "Digits Dataset":
                data = datasets.load_digits()
            elif dataset_name == "Boston Housing Dataset":
                # This is deprecated in some versions; can require "from sklearn.datasets import load_boston"
                # but we'll assume it's still accessible or that user has an older version.
                data = datasets.load_boston()
            else:
                return
            
            # Convert to DataFrame so we can apply missing-data methods
            if hasattr(data, 'feature_names'):
                X_df = pd.DataFrame(data.data, columns=data.feature_names)
            else:
                X_df = pd.DataFrame(data.data)
            y = data.target
            
            # Apply missing data handling
            X_df = self.handle_missing_data(X_df)
            
            # Now do train/test split
            test_size = self.split_spin.value()
            X_train, X_test, y_train, y_test = model_selection.train_test_split(
                X_df, y, test_size=test_size, random_state=42
            )
            
            self.X_train = X_train.values
            self.X_test = X_test.values
            self.y_train = y_train
            self.y_test = y_test
            
            # Scale if selected
            self.apply_scaling()
            
            self.status_bar.showMessage(f"Loaded {dataset_name}")
        except Exception as e:
            self.show_error(f"Error loading dataset: {str(e)}")

    def load_custom_data(self):
        """Load custom dataset from CSV file"""
        try:
            file_name, _ = QFileDialog.getOpenFileName(
                self,
                "Load Dataset",
                "",
                "CSV files (*.csv)"
            )
            if file_name:
                data = pd.read_csv(file_name)
                
                # Ask user to select target column
                target_col = self.select_target_column(data.columns)
                if target_col:
                    X = data.drop(target_col, axis=1)
                    y = data[target_col]
                    
                    # Handle missing data first
                    X = self.handle_missing_data(X)
                    
                    # Split data
                    test_size = self.split_spin.value()
                    X_train, X_test, y_train, y_test = model_selection.train_test_split(
                        X, y, test_size=test_size, random_state=42
                    )
                    
                    self.X_train = X_train.values
                    self.X_test = X_test.values
                    self.y_train = y_train.values
                    self.y_test = y_test.values
                    
                    # Apply scaling
                    self.apply_scaling()
                    
                    self.status_bar.showMessage(f"Loaded custom dataset: {file_name}")
        except Exception as e:
            self.show_error(f"Error loading custom dataset: {str(e)}")

    def handle_missing_data(self, df):
        """Handle missing data in DataFrame according to combo selection."""
        method = self.missing_combo.currentText()
        if method == "No Handling":
            return df
        
        if method == "Mean Imputation":
            imp = SimpleImputer(strategy='mean')
            return pd.DataFrame(imp.fit_transform(df), columns=df.columns)
        elif method == "Interpolation":
            return df.interpolate()
        elif method == "Forward Fill":
            return df.fillna(method='ffill')
        elif method == "Backward Fill":
            return df.fillna(method='bfill')
        
        return df  # fallback

    def select_target_column(self, columns):
        """Dialog to select target column from dataset"""
        dialog = QDialog(self)
        dialog.setWindowTitle("Select Target Column")
        layout = QVBoxLayout(dialog)
        
        combo = QComboBox()
        combo.addItems(columns)
        layout.addWidget(combo)
        
        btn = QPushButton("Select")
        btn.clicked.connect(dialog.accept)
        layout.addWidget(btn)
        
        if dialog.exec() == QDialog.DialogCode.Accepted:
            return combo.currentText()
        return None

    def apply_scaling(self):
        """Apply selected scaling method to the data"""
        scaling_method = self.scaling_combo.currentText()
        if scaling_method == "No Scaling":
            return
        try:
            if scaling_method == "Standard Scaling":
                scaler = preprocessing.StandardScaler()
            elif scaling_method == "Min-Max Scaling":
                scaler = preprocessing.MinMaxScaler()
            elif scaling_method == "Robust Scaling":
                scaler = preprocessing.RobustScaler()
            
            self.X_train = scaler.fit_transform(self.X_train)
            self.X_test = scaler.transform(self.X_test)
        except Exception as e:
            self.show_error(f"Error applying scaling: {str(e)}")

    # Main Tabs
    
    def create_tabs(self):
        """Create tabs for different ML topics"""
        self.tab_widget = QTabWidget()
        
        # Create individual tabs
        tabs = [
            ("Classical ML", self.create_classical_ml_tab),
            ("Deep Learning", self.create_deep_learning_tab),
            ("Dimensionality Reduction", self.create_dim_reduction_tab),
            ("Reinforcement Learning", self.create_rl_tab)
        ]
        
        for tab_name, create_func in tabs:
            scroll = QScrollArea()
            tab_widget = create_func()
            scroll.setWidget(tab_widget)
            scroll.setWidgetResizable(True)
            self.tab_widget.addTab(scroll, tab_name)
        
        self.layout.addWidget(self.tab_widget)

    
    # Classical ML Tab
    def create_classical_ml_tab(self):
        """Create the classical machine learning algorithms tab"""
        widget = QWidget()
        layout = QGridLayout(widget)
        
        
        # Regression section
        regression_group = QGroupBox("Regression")
        regression_layout = QVBoxLayout()

        # Linear Regression
        lr_group = self.create_algorithm_group(
            "Linear Regression",
            {
                "fit_intercept": "checkbox",
                "normalize": "checkbox"
            }
        )
        regression_layout.addWidget(lr_group)

        # Support Vector Regression (SVR)
        svr_group = self.create_algorithm_group(
            "Support Vector Regression",
            {
                "C": "double",
                "kernel": ["linear", "rbf", "poly"],
                "degree": "int",
                "epsilon": "double"
            }
        )
        regression_layout.addWidget(svr_group)

        # SGD Regressor (with dynamic loss selection)
        sgd_reg_group = self.create_algorithm_group(
            "SGD Regression",
            {
                "loss_function": ["MSE", "MAE", "Huber"],  # We'll map these to actual sklearn loss
                "max_iter": "int",
                "tol": "double"
            }
        )
        regression_layout.addWidget(sgd_reg_group)

        regression_group.setLayout(regression_layout)
        layout.addWidget(regression_group, 0, 0)
        
        
        # Classification section
        
        classification_group = QGroupBox("Classification")
        classification_layout = QVBoxLayout()
        
        # Logistic Regression
        logistic_group = self.create_algorithm_group(
            "Logistic Regression",
            {
                "C": "double",
                "max_iter": "int",
                "multi_class": ["ovr", "multinomial"]
            }
        )
        classification_layout.addWidget(logistic_group)

        # Naive Bayes (GaussianNB)
        # Add var_smoothing + custom priors
        nb_group = self.create_algorithm_group(
            "GaussianNB",
            {
                "var_smoothing": "double",
                "priors": "lineedit"  # user can type "0.3,0.7" etc.
            }
        )
        classification_layout.addWidget(nb_group)
        
        # SVM (classification)
        svm_group = self.create_algorithm_group(
            "Support Vector Machine",
            {
                "C": "double",
                "kernel": ["linear", "rbf", "poly"],
                "degree": "int"
            }
        )
        classification_layout.addWidget(svm_group)
        
        # Decision Tree
        dt_group = self.create_algorithm_group(
            "Decision Tree",
            {
                "max_depth": "int",
                "min_samples_split": "int",
                "criterion": ["gini", "entropy"]
            }
        )
        classification_layout.addWidget(dt_group)
        
        # Random Forest
        rf_group = self.create_algorithm_group(
            "Random Forest",
            {
                "n_estimators": "int",
                "max_depth": "int",
                "min_samples_split": "int"
            }
        )
        classification_layout.addWidget(rf_group)
        
        # KNN
        knn_group = self.create_algorithm_group(
            "K-Nearest Neighbors",
            {
                "n_neighbors": "int",
                "weights": ["uniform", "distance"],
                "metric": ["euclidean", "manhattan"]
            }
        )
        classification_layout.addWidget(knn_group)
        
        # SGD Classifier (with dynamic loss selection)
        sgd_clf_group = self.create_algorithm_group(
            "SGD Classification",
            {
                "loss_function": ["Cross-Entropy", "Hinge"],
                "max_iter": "int",
                "tol": "double"
            }
        )
        classification_layout.addWidget(sgd_clf_group)
        
        classification_group.setLayout(classification_layout)
        layout.addWidget(classification_group, 0, 1)
        
        return widget

    # Dim Reduction Tab
    
    def create_dim_reduction_tab(self):
        """Create the dimensionality reduction tab"""
        widget = QWidget()
        layout = QGridLayout(widget)
        
        # K-Means
        kmeans_group = QGroupBox("K-Means Clustering")
        kmeans_layout = QVBoxLayout()
        
        kmeans_params = self.create_algorithm_group(
            "K-Means Parameters",
            {"n_clusters": "int", "max_iter": "int", "n_init": "int"}
        )
        kmeans_layout.addWidget(kmeans_params)
        
        kmeans_group.setLayout(kmeans_layout)
        layout.addWidget(kmeans_group, 0, 0)
        
        # PCA
        pca_group = QGroupBox("Principal Component Analysis")
        pca_layout = QVBoxLayout()
        
        pca_params = self.create_algorithm_group(
            "PCA Parameters",
            {"n_components": "int", "whiten": "checkbox"}
        )
        pca_layout.addWidget(pca_params)
        
        pca_group.setLayout(pca_layout)
        layout.addWidget(pca_group, 0, 1)
        
        return widget

    
    # RL Tab
    def create_rl_tab(self):
        """Create the reinforcement learning tab"""
        widget = QWidget()
        layout = QGridLayout(widget)
        
        # Environment
        env_group = QGroupBox("Environment")
        env_layout = QVBoxLayout()
        
        self.env_combo = QComboBox()
        self.env_combo.addItems([
            "CartPole-v1",
            "MountainCar-v0",
            "Acrobot-v1"
        ])
        env_layout.addWidget(self.env_combo)
        
        env_group.setLayout(env_layout)
        layout.addWidget(env_group, 0, 0)
        
        # RL Algorithm
        algo_group = QGroupBox("RL Algorithm")
        algo_layout = QVBoxLayout()
        
        self.rl_algo_combo = QComboBox()
        self.rl_algo_combo.addItems([
            "Q-Learning",
            "SARSA",
            "DQN"
        ])
        algo_layout.addWidget(self.rl_algo_combo)
        
        algo_group.setLayout(algo_layout)
        layout.addWidget(algo_group, 0, 1)
        
        return widget


    # For visualization

    def create_visualization(self):
        """Create the visualization section"""
        viz_group = QGroupBox("Visualization")
        viz_layout = QHBoxLayout()
        
        # Create matplotlib figure
        self.figure = Figure(figsize=(8, 6))
        self.canvas = FigureCanvas(self.figure)
        viz_layout.addWidget(self.canvas)
        
        # Metrics display
        self.metrics_text = QTextEdit()
        self.metrics_text.setReadOnly(True)
        viz_layout.addWidget(self.metrics_text)
        
        viz_group.setLayout(viz_layout)
        self.layout.addWidget(viz_group)

    def create_status_bar(self):
        """Create the status bar"""
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        
        # Add progress bar
        self.progress_bar = QProgressBar()
        self.status_bar.addPermanentWidget(self.progress_bar)
    
    # Utility: create groups for classical algorithms

    def create_algorithm_group(self, name, params):
        """Helper method to create algorithm parameter groups"""
        group = QGroupBox(name)
        layout = QVBoxLayout()
        
        param_widgets = {}
        for param_name, param_type in params.items():
            param_layout = QHBoxLayout()
            param_layout.addWidget(QLabel(f"{param_name}:"))
            
            if param_type == "int":
                widget = QSpinBox()
                widget.setRange(1, 10000)
            elif param_type == "double":
                widget = QDoubleSpinBox()
                widget.setRange(0.0000001, 1e6)
                widget.setSingleStep(0.1)
            elif param_type == "checkbox":
                widget = QCheckBox()
            elif param_type == "lineedit":
                widget = QLineEdit()
                widget.setPlaceholderText("e.g. 0.3,0.7")
            elif isinstance(param_type, list):
                widget = QComboBox()
                widget.addItems(param_type)
            else:
                widget = QLineEdit()  # fallback if needed
            
            param_layout.addWidget(widget)
            param_widgets[param_name] = widget
            layout.addLayout(param_layout)
        
        train_btn = QPushButton(f"Train {name}")
        train_btn.clicked.connect(lambda: self.train_model(name, param_widgets))
        layout.addWidget(train_btn)
        
        group.setLayout(layout)
        return group
    
    
    # Model Training
    def train_model(self, name, param_widgets):
        """Train a classical ML model based on name and parameters from param_widgets."""
        if self.X_train is None or self.y_train is None:
            self.show_error("No training data loaded.")
            return
        
        # Extract parameters
        params = {}
        for k, w in param_widgets.items():
            if isinstance(w, QSpinBox):
                params[k] = w.value()
            elif isinstance(w, QDoubleSpinBox):
                params[k] = w.value()
            elif isinstance(w, QCheckBox):
                params[k] = w.isChecked()
            elif isinstance(w, QComboBox):
                params[k] = w.currentText()
            elif isinstance(w, QLineEdit):
                params[k] = w.text().strip()
            else:
                params[k] = None
        
        try:
            if name == "Linear Regression":
                # Basic sklearn LinearRegression
                fit_intercept = params.get("fit_intercept", True)
                normalize = params.get("normalize", False)
                # NOTE: 'normalize' is deprecated in latest sklearn,
                # but we'll keep for demonstration if user uses older version
                model = LinearRegression(fit_intercept=fit_intercept)
                # 'normalize' param can be set if version allows: 
                # model = LinearRegression(fit_intercept=fit_intercept, normalize=normalize)
                model.fit(self.X_train, self.y_train)
                self.current_model = model

            elif name == "Support Vector Regression":
                C = float(params.get("C", 1.0))
                kernel = params.get("kernel", "rbf")
                degree = int(params.get("degree", 3))
                epsilon = float(params.get("epsilon", 0.1))
                model = SVR(C=C, kernel=kernel, degree=degree, epsilon=epsilon)
                model.fit(self.X_train, self.y_train)
                self.current_model = model

            elif name == "SGD Regression":
                # Map user-friendly loss names to actual sklearn losses
                loss_map = {
                    "MSE": "squared_error",
                    "MAE": "absolute_error",
                    "Huber": "huber"
                }
                chosen_loss = loss_map.get(params.get("loss_function", "MSE"), "squared_error")
                max_iter = int(params.get("max_iter", 1000))
                tol = float(params.get("tol", 1e-3))
                
                model = SGDRegressor(loss=chosen_loss, max_iter=max_iter, tol=tol)
                model.fit(self.X_train, self.y_train)
                self.current_model = model

            elif name == "Logistic Regression":
                C = float(params.get("C", 1.0))
                max_iter = int(params.get("max_iter", 100))
                multi_class = params.get("multi_class", "ovr")
                model = LogisticRegression(C=C, max_iter=max_iter, multi_class=multi_class)
                model.fit(self.X_train, self.y_train)
                self.current_model = model

            elif name == "GaussianNB":
                var_smoothing = float(params.get("var_smoothing", 1e-9))
                priors_str = params.get("priors", "")
                priors = None
                if priors_str:
                    # user typed something like "0.3,0.7"
                    try:
                        priors_list = [float(x) for x in priors_str.split(",")]
                        priors = priors_list
                    except:
                        priors = None
                model = GaussianNB(var_smoothing=var_smoothing, priors=priors)
                model.fit(self.X_train, self.y_train)
                self.current_model = model

            elif name == "Support Vector Machine":
                C = float(params.get("C", 1.0))
                kernel = params.get("kernel", "rbf")
                degree = int(params.get("degree", 3))
                model = SVC(C=C, kernel=kernel, degree=degree)
                model.fit(self.X_train, self.y_train)
                self.current_model = model

            elif name == "Decision Tree":
                max_depth = int(params.get("max_depth", 5))
                min_samples_split = int(params.get("min_samples_split", 2))
                criterion = params.get("criterion", "gini")
                dt = DecisionTreeClassifier(
                    max_depth=max_depth,
                    min_samples_split=min_samples_split,
                    criterion=criterion
                )
                dt.fit(self.X_train, self.y_train)
                self.current_model = dt

            elif name == "Random Forest":
                n_estimators = int(params.get("n_estimators", 100))
                max_depth = int(params.get("max_depth", 5))
                min_samples_split = int(params.get("min_samples_split", 2))
                rf = RandomForestClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    min_samples_split=min_samples_split
                )
                rf.fit(self.X_train, self.y_train)
                self.current_model = rf

            elif name == "K-Nearest Neighbors":
                n_neighbors = int(params.get("n_neighbors", 5))
                weights = params.get("weights", "uniform")
                metric = params.get("metric", "euclidean")
                knn = KNeighborsClassifier(
                    n_neighbors=n_neighbors,
                    weights=weights,
                    metric=metric
                )
                knn.fit(self.X_train, self.y_train)
                self.current_model = knn

            elif name == "SGD Classification":
                loss_map = {
                    "Cross-Entropy": "log_loss",
                    "Hinge": "hinge"
                }
                chosen_loss = loss_map.get(params.get("loss_function", "Cross-Entropy"), "log_loss")
                max_iter = int(params.get("max_iter", 1000))
                tol = float(params.get("tol", 1e-3))
                sgd_clf = SGDClassifier(loss=chosen_loss, max_iter=max_iter, tol=tol)
                sgd_clf.fit(self.X_train, self.y_train)
                self.current_model = sgd_clf

            elif name == "K-Means Parameters":
                # Just example usage
                from sklearn.cluster import KMeans
                n_clusters = int(params.get("n_clusters", 8))
                max_iter = int(params.get("max_iter", 300))
                n_init = int(params.get("n_init", 10))
                kmeans = KMeans(n_clusters=n_clusters, max_iter=max_iter, n_init=n_init)
                kmeans.fit(self.X_train)
                self.current_model = kmeans

            elif name == "PCA Parameters":
                from sklearn.decomposition import PCA
                n_components = int(params.get("n_components", 2))
                whiten = bool(params.get("whiten", False))
                pca = PCA(n_components=n_components, whiten=whiten)
                pca.fit(self.X_train)
                self.current_model = pca

            else:
                self.show_error("Unknown model selection.")
                return

            self.status_bar.showMessage(f"Trained {name} successfully.")
            self.evaluate_and_visualize()

        except Exception as e:
            self.show_error(f"Error training {name}: {str(e)}")

    def evaluate_and_visualize(self):
        """Evaluate the current model on test set and update visualization / metrics."""
        try:
            if self.current_model is None:
                return
            
            # Predict
            y_pred = self.current_model.predict(self.X_test)
            
            # Clear figure
            self.figure.clear()
            
            # Distinguish regression vs classification
            # We'll guess it's regression if y_test has > 15 unique values
            if len(np.unique(self.y_test)) > 15:
                # Regression evaluation
                mse = mean_squared_error(self.y_test, y_pred)
                rmse = np.sqrt(mse)
                r2 = r2_score(self.y_test, y_pred)
                
                metrics_text = (
                    f"Regression Metrics:\n\n"
                    f"MSE: {mse:.4f}\n"
                    f"RMSE: {rmse:.4f}\n"
                    f"R^2: {r2:.4f}\n"
                )
                
                # Simple scatter: Actual vs. Predicted
                ax = self.figure.add_subplot(111)
                ax.scatter(self.y_test, y_pred)
                ax.plot([self.y_test.min(), self.y_test.max()],
                        [self.y_test.min(), self.y_test.max()],
                        'r--', lw=2)
                ax.set_xlabel("Actual")
                ax.set_ylabel("Predicted")
                ax.set_title("Regression: Actual vs Predicted")
                
            else:
                # Classification evaluation
                accuracy = accuracy_score(self.y_test, y_pred)
                cm = confusion_matrix(self.y_test, y_pred)
                
                metrics_text = (
                    f"Classification Metrics:\n\n"
                    f"Accuracy: {accuracy:.4f}\n\n"
                    f"Confusion Matrix:\n{cm}\n"
                )
                
                # Plot confusion matrix in a subplot
                ax1 = self.figure.add_subplot(121)
                ax1.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
                ax1.set_title("Confusion Matrix")
                ax1.set_xlabel("Predicted")
                ax1.set_ylabel("True")
                
                # For some classification visualization, if features > 2 use PCA
                ax2 = self.figure.add_subplot(122)
                if self.X_test.shape[1] > 2:
                    pca = PCA(n_components=2)
                    X_test_2d = pca.fit_transform(self.X_test)
                    scatter = ax2.scatter(X_test_2d[:,0], X_test_2d[:,1], c=y_pred, cmap='viridis')
                    ax2.set_title("Predictions (PCA View)")
                else:
                    scatter = ax2.scatter(self.X_test[:,0], self.X_test[:,1], c=y_pred, cmap='viridis')
                    ax2.set_title("Predictions (2D)")
            
            self.metrics_text.setText(metrics_text)
            self.figure.tight_layout()
            self.canvas.draw()
            
        except Exception as e:
            self.show_error(f"Error during evaluation/visualization: {str(e)}")

    
    # Deep Learning Tab
    
    def create_deep_learning_tab(self):
        widget = QWidget()
        layout = QGridLayout(widget)
        
        # MLP
        mlp_group = QGroupBox("Multi-Layer Perceptron")
        mlp_layout = QVBoxLayout()
        
        # Button to add layers
        layer_btn = QPushButton("Add Layer")
        layer_btn.clicked.connect(self.add_layer_dialog)
        mlp_layout.addWidget(layer_btn)
        
        # Training params
        training_params_group = self.create_training_params_group()
        mlp_layout.addWidget(training_params_group)
        
        # Train button
        train_btn = QPushButton("Train Neural Network")
        train_btn.clicked.connect(self.train_neural_network)
        mlp_layout.addWidget(train_btn)
        
        mlp_group.setLayout(mlp_layout)
        layout.addWidget(mlp_group, 0, 0)
        
        # CNN placeholder
        cnn_group = QGroupBox("Convolutional Neural Network")
        cnn_layout = QVBoxLayout()
        cnn_layout.addWidget(QLabel("CNN Controls (To be implemented)"))
        cnn_group.setLayout(cnn_layout)
        layout.addWidget(cnn_group, 0, 1)
        
        # RNN placeholder
        rnn_group = QGroupBox("Recurrent Neural Network")
        rnn_layout = QVBoxLayout()
        rnn_layout.addWidget(QLabel("RNN Controls (To be implemented)"))
        rnn_group.setLayout(rnn_layout)
        layout.addWidget(rnn_group, 1, 0)
        
        return widget

    def add_layer_dialog(self):
        """Open a dialog to add a neural network layer."""
        dialog = QDialog(self)
        dialog.setWindowTitle("Add Neural Network Layer")
        layout = QVBoxLayout(dialog)
        
        # Layer type
        type_layout = QHBoxLayout()
        type_label = QLabel("Layer Type:")
        type_combo = QComboBox()
        type_combo.addItems(["Dense", "Conv2D", "MaxPooling2D", "Flatten", "Dropout"])
        type_layout.addWidget(type_label)
        type_layout.addWidget(type_combo)
        layout.addLayout(type_layout)
        
        params_group = QGroupBox("Layer Parameters")
        params_layout = QVBoxLayout()
        self.layer_param_inputs = {}
        
        def update_params():
            # Clear existing
            for obj in list(self.layer_param_inputs.values()):
                params_layout.removeWidget(obj)
                obj.deleteLater()
            self.layer_param_inputs.clear()
            
            layer_type = type_combo.currentText()
            if layer_type == "Dense":
                units_label = QLabel("Units:")
                units_input = QSpinBox()
                units_input.setRange(1, 1000)
                units_input.setValue(32)
                self.layer_param_inputs["units"] = units_input
                
                activation_label = QLabel("Activation:")
                activation_combo = QComboBox()
                activation_combo.addItems(["relu", "sigmoid", "tanh", "softmax"])
                self.layer_param_inputs["activation"] = activation_combo
                
                params_layout.addWidget(units_label)
                params_layout.addWidget(units_input)
                params_layout.addWidget(activation_label)
                params_layout.addWidget(activation_combo)
            
            elif layer_type == "Conv2D":
                filters_label = QLabel("Filters:")
                filters_input = QSpinBox()
                filters_input.setRange(1, 1000)
                filters_input.setValue(32)
                self.layer_param_inputs["filters"] = filters_input

                kernel_label = QLabel("Kernel Size (e.g. 3,3):")
                kernel_input = QLineEdit("3,3")
                self.layer_param_inputs["kernel_size"] = kernel_input

                params_layout.addWidget(filters_label)
                params_layout.addWidget(filters_input)
                params_layout.addWidget(kernel_label)
                params_layout.addWidget(kernel_input)

            elif layer_type == "MaxPooling2D":
                # Typically kernel size
                kernel_label = QLabel("Pool Size (e.g. 2,2):")
                kernel_input = QLineEdit("2,2")
                self.layer_param_inputs["pool_size"] = kernel_input
                params_layout.addWidget(kernel_label)
                params_layout.addWidget(kernel_input)

            elif layer_type == "Flatten":
                # no params
                pass
            
            elif layer_type == "Dropout":
                rate_label = QLabel("Dropout Rate:")
                rate_input = QDoubleSpinBox()
                rate_input.setRange(0.0, 1.0)
                rate_input.setValue(0.5)
                rate_input.setSingleStep(0.1)
                self.layer_param_inputs["rate"] = rate_input
                params_layout.addWidget(rate_label)
                params_layout.addWidget(rate_input)

        type_combo.currentIndexChanged.connect(update_params)
        update_params()
        
        params_group.setLayout(params_layout)
        layout.addWidget(params_group)
        
        btn_layout = QHBoxLayout()
        add_btn = QPushButton("Add Layer")
        cancel_btn = QPushButton("Cancel")
        btn_layout.addWidget(add_btn)
        btn_layout.addWidget(cancel_btn)
        layout.addLayout(btn_layout)
        
        def add_layer():
            layer_type = type_combo.currentText()
            layer_params = {}
            for param_name, widget in self.layer_param_inputs.items():
                if isinstance(widget, QSpinBox):
                    layer_params[param_name] = widget.value()
                elif isinstance(widget, QDoubleSpinBox):
                    layer_params[param_name] = widget.value()
                elif isinstance(widget, QLineEdit):
                    text_val = widget.text().strip()
                    if param_name in ["kernel_size", "pool_size"]:
                        # parse "3,3" -> (3,3)
                        try:
                            numbers = [int(x) for x in text_val.split(",")]
                            layer_params[param_name] = tuple(numbers)
                        except:
                            layer_params[param_name] = (3,3)
                    else:
                        layer_params[param_name] = text_val
                elif isinstance(widget, QComboBox):
                    layer_params[param_name] = widget.currentText()
            
            self.layer_config.append({
                "type": layer_type,
                "params": layer_params
            })
            dialog.accept()
        
        add_btn.clicked.connect(add_layer)
        cancel_btn.clicked.connect(dialog.reject)
        
        dialog.exec()

    def create_training_params_group(self):
        group = QGroupBox("Training Parameters")
        layout = QVBoxLayout()

        # Batch size
        batch_layout = QHBoxLayout()
        batch_layout.addWidget(QLabel("Batch Size:"))
        self.batch_size_spin = QSpinBox()
        self.batch_size_spin.setRange(1, 10000)
        self.batch_size_spin.setValue(32)
        batch_layout.addWidget(self.batch_size_spin)
        layout.addLayout(batch_layout)
        
        # Epochs
        epochs_layout = QHBoxLayout()
        epochs_layout.addWidget(QLabel("Epochs:"))
        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 10000)
        self.epochs_spin.setValue(10)
        epochs_layout.addWidget(self.epochs_spin)
        layout.addLayout(epochs_layout)
        
        # Learning rate
        lr_layout = QHBoxLayout()
        lr_layout.addWidget(QLabel("Learning Rate:"))
        self.lr_spin = QDoubleSpinBox()
        self.lr_spin.setRange(1e-7, 1.0)
        self.lr_spin.setValue(0.001)
        self.lr_spin.setSingleStep(0.001)
        lr_layout.addWidget(self.lr_spin)
        layout.addLayout(lr_layout)
        
        group.setLayout(layout)
        return group

    def train_neural_network(self):
        """Train a simple MLP (or custom arch) using Keras, based on self.layer_config."""
        if self.X_train is None or self.y_train is None:
            self.show_error("No data loaded for training the neural network.")
            return
        
        if len(self.layer_config) == 0:
            self.show_error("Please add at least one layer to the network.")
            return

        try:
            # Prepare data: If classification, we one-hot encode
            # But we must guess classification vs regression. Let's guess classification if few unique targets.
            unique_labels = np.unique(self.y_train)
            if len(unique_labels) <= 15:
                # classification
                num_classes = len(unique_labels)
                # Remap y to [0..num_classes-1] if it's not already
                # This is for safety in case labels are [1,2,...]
                label_to_int = {val: i for i, val in enumerate(unique_labels)}
                y_train_int = np.array([label_to_int[val] for val in self.y_train])
                y_test_int = np.array([label_to_int[val] for val in self.y_test])
                
                y_train_oh = tf.keras.utils.to_categorical(y_train_int, num_classes)
                y_test_oh = tf.keras.utils.to_categorical(y_test_int, num_classes)

                # Model
                model = self.build_keras_model(num_classes=num_classes, is_classification=True)
                
                # Compile
                lr = self.lr_spin.value()
                optimizer = optimizers.Adam(learning_rate=lr)
                model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

                # Fit
                history = model.fit(
                    self.X_train, y_train_oh,
                    validation_data=(self.X_test, y_test_oh),
                    epochs=self.epochs_spin.value(),
                    batch_size=self.batch_size_spin.value(),
                    callbacks=[self.create_progress_callback()]
                )
                
            else:
                # regression
                model = self.build_keras_model(num_classes=1, is_classification=False)
                lr = self.lr_spin.value()
                optimizer = optimizers.Adam(learning_rate=lr)
                # We'll use MSE for regression
                model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

                history = model.fit(
                    self.X_train, self.y_train,
                    validation_data=(self.X_test, self.y_test),
                    epochs=self.epochs_spin.value(),
                    batch_size=self.batch_size_spin.value(),
                    callbacks=[self.create_progress_callback()]
                )
            
            self.plot_training_history(history)
            self.status_bar.showMessage("Neural Network Training Complete")
            
        except Exception as e:
            self.show_error(f"Error training neural network: {str(e)}")

    def build_keras_model(self, num_classes=1, is_classification=False):
        """Build a Keras Sequential model from self.layer_config."""
        model = models.Sequential()
        
        input_dim = self.X_train.shape[1]
        first_layer = True
        
        for cfg in self.layer_config:
            ltype = cfg["type"]
            params = cfg["params"]
            
            if ltype == "Dense":
                units = params.get("units", 32)
                activation = params.get("activation", "relu")
                if first_layer:
                    model.add(layers.Dense(units, activation=activation, input_shape=(input_dim,)))
                    first_layer = False
                else:
                    model.add(layers.Dense(units, activation=activation))
            
            elif ltype == "Dropout":
                rate = params.get("rate", 0.5)
                model.add(layers.Dropout(rate))
            
            elif ltype == "Conv2D":
                # This typically requires 4D input: (batch, height, width, channels)
                # For a standard tabular approach, this won't apply. We'll keep code to illustrate.
                filters = params.get("filters", 32)
                kernel_size = params.get("kernel_size", (3,3))
                if first_layer:
                    # We guess the user has data shaped as images in self.X_train
                    # If so, you need to reshape it outside. This is just a placeholder.
                    # e.g. input_shape=(28,28,1) for MNIST
                    # We'll do a fallback for demonstration.
                    # Suppose MNIST shape as (28,28,1):
                    model.add(layers.Reshape((28,28,1), input_shape=(784,)))
                    model.add(layers.Conv2D(filters, kernel_size, activation='relu'))
                    first_layer = False
                else:
                    model.add(layers.Conv2D(filters, kernel_size, activation='relu'))

            elif ltype == "MaxPooling2D":
                pool_size = params.get("pool_size", (2,2))
                model.add(layers.MaxPooling2D(pool_size=pool_size))

            elif ltype == "Flatten":
                model.add(layers.Flatten())
        
        # Finally output layer
        if is_classification:
            model.add(layers.Dense(num_classes, activation='softmax'))
        else:
            model.add(layers.Dense(1))  # regression output
        return model

    def plot_training_history(self, history):
        """Plot neural network training history"""
        self.figure.clear()
        
        # If 'accuracy' in history, classification
        if 'accuracy' in history.history:
            ax1 = self.figure.add_subplot(211)
            ax1.plot(history.history['accuracy'], label='Train Accuracy')
            ax1.plot(history.history['val_accuracy'], label='Val Accuracy')
            ax1.set_title('Accuracy')
            ax1.set_ylabel('Accuracy')
            ax1.set_xlabel('Epoch')
            ax1.legend()

            ax2 = self.figure.add_subplot(212)
            ax2.plot(history.history['loss'], label='Train Loss')
            ax2.plot(history.history['val_loss'], label='Val Loss')
            ax2.set_title('Loss')
            ax2.set_ylabel('Loss')
            ax2.set_xlabel('Epoch')
            ax2.legend()
        
        else:
            # regression (e.g. 'mae')
            ax1 = self.figure.add_subplot(211)
            ax1.plot(history.history['loss'], label='Train Loss')
            ax1.plot(history.history['val_loss'], label='Val Loss')
            ax1.set_title('Loss')
            ax1.set_ylabel('Loss')
            ax1.set_xlabel('Epoch')
            ax1.legend()

            if 'mae' in history.history:
                ax2 = self.figure.add_subplot(212)
                ax2.plot(history.history['mae'], label='Train MAE')
                ax2.plot(history.history['val_mae'], label='Val MAE')
                ax2.set_title('Mean Absolute Error')
                ax2.set_ylabel('MAE')
                ax2.set_xlabel('Epoch')
                ax2.legend()

        self.figure.tight_layout()
        self.canvas.draw()

    def create_progress_callback(self):
        """Create callback for updating progress bar during training"""
        class ProgressCallback(tf.keras.callbacks.Callback):
            def __init__(self, progress_bar, total_epochs):
                super().__init__()
                self.progress_bar = progress_bar
                self.total_epochs = total_epochs

            def on_epoch_end(self, epoch, logs=None):
                progress = int(((epoch + 1) / self.total_epochs) * 100)
                self.progress_bar.setValue(progress)

            def on_train_end(self, logs=None):
                self.progress_bar.setValue(100)

        return ProgressCallback(self.progress_bar, total_epochs=self.epochs_spin.value())
    
    # Error handling
    def show_error(self, message):
        QMessageBox.critical(self, "Error", message)

def main():
    """Main function to start the application"""
    app = QApplication(sys.argv)
    window = MLCourseGUI()
    window.show()
    sys.exit(app.exec())

if __name__ == '__main__':
    main()
