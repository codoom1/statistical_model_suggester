"""Neural Network diagnostic plots."""
import matplotlib.pyplot as plt
import numpy as np
import io
import base64
import pandas as pd
import seaborn as sns
from matplotlib.gridspec import GridSpec
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
from sklearn.metrics import average_precision_score, accuracy_score, classification_report
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.inspection import permutation_importance
from sklearn.model_selection import validation_curve

def get_base64_plot():
    """Convert current matplotlib plot to base64 string"""
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()
    plt.close()
    return base64.b64encode(image_png).decode('utf-8')

def generate_neural_network_plots(model, X=None, y=None, X_test=None, y_test=None, 
                                 feature_names=None, class_names=None, history=None,
                                 is_classifier=None):
    """Generate diagnostic plots for neural network models
    
    Args:
        model: Fitted neural network model (TensorFlow/Keras, PyTorch, or scikit-learn)
        X: Feature matrix
        y: Target variable
        X_test: Test data features (optional)
        y_test: Test data target (optional)
        feature_names: Names of features (optional)
        class_names: Names of classes for classification (optional)
        history: Training history object (from model.fit() in Keras) 
        is_classifier: Whether the model is a classifier (will be inferred if None)
        
    Returns:
        List of dictionaries with plot information
    """
    plots = []
    
    # Determine if model is a classifier if not specified
    if is_classifier is None:
        # Try to infer from model attributes or output shape
        if hasattr(model, 'predict_proba'):
            is_classifier = True
        elif hasattr(model, '_estimator_type'):
            is_classifier = model._estimator_type == 'classifier'
        elif y is not None:
            # Check if y has discrete values
            unique_values = np.unique(y)
            if len(unique_values) < 10 and isinstance(unique_values[0], (int, np.integer, bool)):
                is_classifier = True
            else:
                is_classifier = False
        else:
            # Default to classifier
            is_classifier = True
    
    # Check if the model has a training history (Keras/TensorFlow models)
    has_history = history is not None
    if not has_history and hasattr(model, 'history') and hasattr(model.history, 'history'):
        history = model.history.history
        has_history = True
    
    # Plot 1: Training History
    if has_history:
        plt.figure(figsize=(12, 6))
        
        # Create a 1x2 subplot for loss and metrics
        plt.subplot(1, 2, 1)
        
        # Plot training loss
        if 'loss' in history:
            plt.plot(history['loss'], label='Training Loss')
        
        # Plot validation loss if available
        if 'val_loss' in history:
            plt.plot(history['val_loss'], label='Validation Loss')
            
        plt.title('Loss During Training')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot accuracy or other metrics if available
        plt.subplot(1, 2, 2)
        metrics_plotted = False
        
        # Try common metric names
        for metric in ['accuracy', 'acc', 'r2', 'auc', 'f1_score']:
            if metric in history:
                plt.plot(history[metric], label=f'Training {metric.capitalize()}')
                metrics_plotted = True
            
            val_metric = f'val_{metric}'
            if val_metric in history:
                plt.plot(history[val_metric], label=f'Validation {metric.capitalize()}')
                metrics_plotted = True
        
        if metrics_plotted:
            plt.title('Metrics During Training')
            plt.xlabel('Epoch')
            plt.ylabel('Value')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        plots.append({
            "title": "Training History",
            "img_data": get_base64_plot(),
            "interpretation": "Shows how loss and evaluation metrics changed during model training. Ideally, both training and validation losses should decrease over time and stabilize. If validation loss increases while training loss continues to decrease, this indicates overfitting."
        })
    
    # Plot 2: Model Architecture Visualization (if possible)
    if hasattr(model, 'summary') and not isinstance(model.summary, pd.DataFrame):
        try:
            # Try to use plot_model if available (requires pydot and graphviz)
            from tensorflow.keras.utils import plot_model
            
            # Save model diagram to buffer
            buffer = io.BytesIO()
            plot_model(model, show_shapes=True, to_file=buffer, 
                    show_layer_names=True, dpi=96, format='png')
            buffer.seek(0)
            image_png = buffer.getvalue()
            buffer.close()
            
            plots.append({
                "title": "Model Architecture",
                "img_data": base64.b64encode(image_png).decode('utf-8'),
                "interpretation": "Visualization of the neural network architecture, showing layers, connections, and output shapes. This helps understand the model's complexity and information flow."
            })
        except:
            # If plot_model fails, provide a text description instead
            try:
                # Capture model summary as string
                from io import StringIO
                import sys
                
                # Save original stdout and redirect to StringIO
                original_stdout = sys.stdout
                string_io = StringIO()
                sys.stdout = string_io
                
                # Call model.summary()
                model.summary()
                
                # Get the summary text and restore stdout
                summary_text = string_io.getvalue()
                sys.stdout = original_stdout
                
                # Create a text-based visualization
                plt.figure(figsize=(10, 8))
                plt.text(0.1, 0.5, summary_text, family='monospace')
                plt.axis('off')
                
                plots.append({
                    "title": "Model Architecture Summary",
                    "img_data": get_base64_plot(),
                    "interpretation": "Text summary of the neural network architecture, showing layers, parameter counts, and output shapes. This provides insight into model complexity."
                })
            except:
                pass
    
    # Ensure we have prediction capabilities
    has_predict = hasattr(model, 'predict')
    has_predict_proba = (hasattr(model, 'predict_proba') and is_classifier)
    
    # Plot 3: Confusion Matrix (for classification)
    if is_classifier and has_predict and X is not None and y is not None:
        # Get predictions
        y_pred = model.predict(X)
        
        # Handle different output formats
        if len(y_pred.shape) > 1 and y_pred.shape[1] > 1:
            # Multi-class probabilities, convert to class indices
            y_pred_classes = np.argmax(y_pred, axis=1)
        elif len(y_pred.shape) > 1 and y_pred.shape[1] == 1:
            # Binary probabilities as column vector
            y_pred_classes = (y_pred > 0.5).astype(int).flatten()
        else:
            # Already class indices or binary predictions
            y_pred_classes = y_pred
            
        # Calculate confusion matrix
        cm = confusion_matrix(y, y_pred_classes)
        
        # Get class names if not provided
        if class_names is None:
            class_names = [str(i) for i in range(cm.shape[0])]
            
        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        
        plots.append({
            "title": "Confusion Matrix",
            "img_data": get_base64_plot(),
            "interpretation": "Shows correct and incorrect predictions for each class. Diagonal cells show correctly classified instances, while off-diagonal cells show misclassifications. Higher numbers along the diagonal indicate better performance."
        })
        
        # Test data confusion matrix if available
        if X_test is not None and y_test is not None:
            # Get test set predictions
            y_test_pred = model.predict(X_test)
            
            # Handle different output formats
            if len(y_test_pred.shape) > 1 and y_test_pred.shape[1] > 1:
                y_test_pred_classes = np.argmax(y_test_pred, axis=1)
            elif len(y_test_pred.shape) > 1 and y_test_pred.shape[1] == 1:
                y_test_pred_classes = (y_test_pred > 0.5).astype(int).flatten()
            else:
                y_test_pred_classes = y_test_pred
                
            # Calculate test confusion matrix
            cm_test = confusion_matrix(y_test, y_test_pred_classes)
            
            # Plot test confusion matrix
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm_test, annot=True, fmt='d', cmap='Blues',
                        xticklabels=class_names, yticklabels=class_names)
            plt.title('Test Data Confusion Matrix')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.tight_layout()
            
            plots.append({
                "title": "Test Data Confusion Matrix",
                "img_data": get_base64_plot(),
                "interpretation": "Shows model performance on unseen test data. Compare with the training confusion matrix to assess overfitting. Similar patterns between training and test matrices suggest good generalization."
            })
    
    # Plot 4: ROC Curves (for classification)
    if is_classifier and has_predict and X is not None and y is not None:
        # Get predictions (probabilities)
        if has_predict_proba:
            y_pred_proba = model.predict_proba(X)
        else:
            # If no predict_proba, try to use raw predictions
            y_pred_proba = model.predict(X)
            
            # Handle different output formats
            if len(y_pred_proba.shape) == 1 or y_pred_proba.shape[1] == 1:
                # Binary case: ensure we have probabilities
                if y_pred_proba.min() >= 0 and y_pred_proba.max() <= 1:
                    # These are probabilities
                    if len(y_pred_proba.shape) > 1:
                        y_pred_proba = y_pred_proba.flatten()
                    y_pred_proba = np.column_stack((1 - y_pred_proba, y_pred_proba))
                else:
                    # These are not probabilities, convert to binary classification
                    y_pred_proba = np.column_stack((np.zeros_like(y_pred_proba), 
                                                  (y_pred_proba > 0).astype(float)))
        
        # Prepare one-hot encoded targets if needed
        n_classes = y_pred_proba.shape[1] if len(y_pred_proba.shape) > 1 else 2
        
        if n_classes == 2:
            # Binary classification
            plt.figure(figsize=(8, 6))
            
            fpr, tpr, _ = roc_curve(y, y_pred_proba[:, 1] if y_pred_proba.shape[1] > 1 else y_pred_proba)
            roc_auc = auc(fpr, tpr)
            
            plt.plot(fpr, tpr, lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
            plt.plot([0, 1], [0, 1], 'k--', lw=2)
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver Operating Characteristic (ROC) Curve')
            plt.legend(loc="lower right")
            plt.grid(True, alpha=0.3)
            
            plots.append({
                "title": "ROC Curve",
                "img_data": get_base64_plot(),
                "interpretation": f"Shows the trade-off between true positive rate and false positive rate at different classification thresholds. The Area Under the Curve (AUC) of {roc_auc:.3f} indicates the model's ability to distinguish between classes. A perfect classifier has AUC=1, while random guessing yields AUC=0.5."
            })
            
            # Test data ROC curve if available
            if X_test is not None and y_test is not None:
                plt.figure(figsize=(8, 6))
                
                # Get test set predictions
                if has_predict_proba:
                    y_test_pred_proba = model.predict_proba(X_test)
                else:
                    y_test_pred_proba = model.predict(X_test)
                    if len(y_test_pred_proba.shape) == 1 or y_test_pred_proba.shape[1] == 1:
                        if len(y_test_pred_proba.shape) > 1:
                            y_test_pred_proba = y_test_pred_proba.flatten()
                        y_test_pred_proba = np.column_stack((1 - y_test_pred_proba, y_test_pred_proba))
                
                fpr_test, tpr_test, _ = roc_curve(y_test, 
                                                 y_test_pred_proba[:, 1] if y_test_pred_proba.shape[1] > 1 else y_test_pred_proba)
                roc_auc_test = auc(fpr_test, tpr_test)
                
                plt.plot(fpr_test, tpr_test, lw=2, label=f'ROC curve (AUC = {roc_auc_test:.3f})')
                plt.plot([0, 1], [0, 1], 'k--', lw=2)
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.05])
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title('Test Data ROC Curve')
                plt.legend(loc="lower right")
                plt.grid(True, alpha=0.3)
                
                plots.append({
                    "title": "Test Data ROC Curve",
                    "img_data": get_base64_plot(),
                    "interpretation": f"Shows model performance on unseen test data. The AUC of {roc_auc_test:.3f} should be compared with the training AUC to check for overfitting. Similar values suggest good generalization."
                })
        
        else:
            # Multiclass ROC curves (one-vs-rest)
            plt.figure(figsize=(10, 8))
            
            # Binarize the target if needed
            from sklearn.preprocessing import label_binarize
            
            # Get unique classes
            classes = np.unique(y)
            n_classes = len(classes)
            
            # Prepare one-hot encoded targets
            y_bin = label_binarize(y, classes=classes)
            
            # Compute ROC curve and AUC for each class
            fpr = dict()
            tpr = dict()
            roc_auc = dict()
            
            for i in range(n_classes):
                fpr[i], tpr[i], _ = roc_curve(y_bin[:, i] if y_bin.shape[1] > 1 else (y == classes[i]).astype(int), 
                                             y_pred_proba[:, i])
                roc_auc[i] = auc(fpr[i], tpr[i])
                plt.plot(fpr[i], tpr[i], lw=2, 
                        label=f'Class {class_names[i] if class_names else classes[i]} (AUC = {roc_auc[i]:.3f})')
            
            plt.plot([0, 1], [0, 1], 'k--', lw=2)
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Curves (One-vs-Rest)')
            plt.legend(loc="lower right")
            plt.grid(True, alpha=0.3)
            
            plots.append({
                "title": "ROC Curves (Multiclass)",
                "img_data": get_base64_plot(),
                "interpretation": f"Shows separate ROC curves for each class in a one-vs-rest approach. Each curve shows how well the model distinguishes a specific class from all others. Higher AUC values indicate better class separation."
            })
    
    # Plot 5: Feature Importance (through permutation importance)
    if has_predict and X is not None and y is not None and feature_names is not None:
        try:
            # Calculate permutation importance
            perm_importance = permutation_importance(model, X, y, n_repeats=5, random_state=42)
            
            # Sort features by importance
            sorted_idx = perm_importance.importances_mean.argsort()[::-1]
            sorted_importance = perm_importance.importances_mean[sorted_idx]
            sorted_std = perm_importance.importances_std[sorted_idx]
            sorted_features = np.array(feature_names)[sorted_idx]
            
            # Plot only top 15 features for readability
            top_k = min(15, len(sorted_features))
            
            plt.figure(figsize=(10, 8))
            plt.barh(range(top_k), sorted_importance[:top_k], xerr=sorted_std[:top_k], 
                    align='center', alpha=0.8)
            plt.yticks(range(top_k), sorted_features[:top_k])
            plt.xlabel('Permutation Importance')
            plt.title('Feature Importance (Permutation-Based)')
            plt.tight_layout()
            plt.grid(True, alpha=0.3)
            
            plots.append({
                "title": "Feature Importance",
                "img_data": get_base64_plot(),
                "interpretation": "Shows which features have the most impact on model predictions. Features are ranked by their permutation importance, which measures how much model performance decreases when a feature is randomly shuffled. Higher values indicate more important features."
            })
        except:
            # Skip permutation importance plot if it fails
            pass
    
    # Plot 6: For regression, actual vs predicted values
    if not is_classifier and has_predict and X is not None and y is not None:
        plt.figure(figsize=(10, 8))
        
        # Get predictions
        y_pred = model.predict(X)
        
        # Handle different output formats
        if len(y_pred.shape) > 1 and y_pred.shape[1] == 1:
            y_pred = y_pred.flatten()
            
        # Scatter plot
        plt.scatter(y, y_pred, alpha=0.6)
        
        # Add perfect prediction line
        min_val = min(min(y), min(y_pred))
        max_val = max(max(y), max(y_pred))
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
        
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title('Actual vs Predicted Values (Training Data)')
        plt.grid(True, alpha=0.3)
        
        # Calculate metrics
        mse = mean_squared_error(y, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y, y_pred)
        r2 = r2_score(y, y_pred)
        
        # Add metrics annotation
        plt.annotate(f'RMSE: {rmse:.3f}\nMAE: {mae:.3f}\nR²: {r2:.3f}', 
                   xy=(0.05, 0.95), xycoords='axes fraction',
                   bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
        
        plots.append({
            "title": "Actual vs Predicted Values",
            "img_data": get_base64_plot(),
            "interpretation": f"Shows how well the model's predictions match actual values in the training data. Points closer to the red diagonal line indicate better predictions. RMSE of {rmse:.3f} penalizes large errors, MAE of {mae:.3f} shows average error magnitude, and R² of {r2:.3f} indicates the proportion of variance explained."
        })
        
        # Test data predictions if available
        if X_test is not None and y_test is not None:
            plt.figure(figsize=(10, 8))
            
            # Get test predictions
            y_test_pred = model.predict(X_test)
            
            # Handle different output formats
            if len(y_test_pred.shape) > 1 and y_test_pred.shape[1] == 1:
                y_test_pred = y_test_pred.flatten()
                
            # Scatter plot
            plt.scatter(y_test, y_test_pred, alpha=0.6)
            
            # Add perfect prediction line
            min_val = min(min(y_test), min(y_test_pred))
            max_val = max(max(y_test), max(y_test_pred))
            plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
            
            plt.xlabel('Actual Values')
            plt.ylabel('Predicted Values')
            plt.title('Actual vs Predicted Values (Test Data)')
            plt.grid(True, alpha=0.3)
            
            # Calculate metrics for test data
            test_mse = mean_squared_error(y_test, y_test_pred)
            test_rmse = np.sqrt(test_mse)
            test_mae = mean_absolute_error(y_test, y_test_pred)
            test_r2 = r2_score(y_test, y_test_pred)
            
            # Add metrics annotation
            plt.annotate(f'RMSE: {test_rmse:.3f}\nMAE: {test_mae:.3f}\nR²: {test_r2:.3f}', 
                       xy=(0.05, 0.95), xycoords='axes fraction',
                       bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
            
            plots.append({
                "title": "Test Set Performance",
                "img_data": get_base64_plot(),
                "interpretation": f"Shows model performance on unseen test data. Compare these metrics (RMSE: {test_rmse:.3f}, MAE: {test_mae:.3f}, R²: {test_r2:.3f}) with training metrics to assess overfitting. Similar performance suggests good generalization."
            })
            
            # Residual analysis
            plt.figure(figsize=(10, 6))
            
            # Calculate residuals
            residuals = y_test - y_test_pred
            
            # Scatter plot
            plt.scatter(y_test_pred, residuals, alpha=0.6)
            
            # Add horizontal line at 0
            plt.axhline(y=0, color='r', linestyle='--')
            
            plt.xlabel('Predicted Values')
            plt.ylabel('Residuals')
            plt.title('Residuals vs Predicted Values (Test Data)')
            plt.grid(True, alpha=0.3)
            
            plots.append({
                "title": "Residual Analysis",
                "img_data": get_base64_plot(),
                "interpretation": "Examines patterns in prediction errors. Ideally, residuals should be randomly scattered around zero (red line) with no clear pattern. Systematic patterns indicate that the model hasn't captured some aspect of the data relationship."
            })
    
    # Plot 7: Layer Activations Visualization (if applicable)
    # This is more complex and depends on specific frameworks
    # Simplified implementation for Keras/TensorFlow models
    if hasattr(model, 'layers'):
        try:
            # Try to visualize filter patterns of the first convolutional layer
            for layer in model.layers:
                if 'conv' in layer.name.lower() and hasattr(layer, 'get_weights'):
                    # Get filter weights
                    filters = layer.get_weights()[0]
                    
                    # Plot up to 16 filters from the first layer
                    n_filters = min(16, filters.shape[3])
                    
                    # Determine subplot grid size
                    grid_size = int(np.ceil(np.sqrt(n_filters)))
                    
                    # Calculate filter min/max for normalization
                    f_min, f_max = filters.min(), filters.max()
                    
                    # Plot filters
                    plt.figure(figsize=(12, 10))
                    for i in range(n_filters):
                        plt.subplot(grid_size, grid_size, i+1)
                        
                        # Get the filter
                        f = filters[:, :, 0, i]
                        
                        # Normalize filter to [0, 1]
                        if f_max > f_min:
                            f = (f - f_min) / (f_max - f_min)
                        
                        plt.imshow(f, cmap='viridis')
                        plt.axis('off')
                    
                    plt.suptitle(f'Filters from layer: {layer.name}')
                    plt.tight_layout()
                    
                    plots.append({
                        "title": f"Layer Filters: {layer.name}",
                        "img_data": get_base64_plot(),
                        "interpretation": "Visualizes learned filters from a convolutional layer. These filters show patterns the model has learned to detect in the input data. Different filters capture different features like edges, textures, or shapes."
                    })
                    
                    # Only visualize the first convolutional layer found
                    break
        except:
            # Skip filter visualization if it fails
            pass
    
    return plots 