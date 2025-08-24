import pandas as pd
import numpy as np
from pycaret.classification import *
from pycaret.datasets import get_data
import joblib
import os
from datetime import datetime
from sklearn.model_selection import train_test_split

class CreditCardFraudPipeline:
    def __init__(self, experiment_name="credit_card_fraud_detection"):
        self.experiment_name = experiment_name
        self.model = None
        self.setup_complete = False
        
    def load_cc_data(self, csv_file_path="creditcard_2023.csv"):
        """
        Load credit card dataset from CSV file
        """
        print(f"Loading credit card dataset from {csv_file_path}...")
        
        try:
            # Read the CSV file
            data = pd.read_csv(csv_file_path)
            
            # Display basic info about the dataset
            print(f"Dataset shape: {data.shape}")
            print(f"Columns: {list(data.columns)}")
            
            # Check for missing values
            missing_values = data.isnull().sum().sum()
            print(f"Missing values: {missing_values}")
            
            # Check class distribution
            class_distribution = data['Class'].value_counts()
            print(f"Class distribution:\n{class_distribution}")
            print(f"Fraud percentage: {(class_distribution[1] / len(data)) * 100:.2f}%")
            
            # The target column is 'Class'
            target = 'Class'
            
            # Remove the 'id' column if it exists (it's just an index)
            if 'id' in data.columns:
                data = data.drop('id', axis=1)
                print("Removed 'id' column as it's not needed for modeling")
            
            return data, target
            
        except FileNotFoundError:
            print(f"Error: File {csv_file_path} not found!")
            print("Please ensure the CSV file is in the correct location.")
            return None, None
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            return None, None
    
    def setup_ml_environment(self, data, target, train_size=0.8):
        """
        Setup PyCaret environment for credit card fraud detection
        """
        print("Setting up ML environment for fraud detection...")
        
        # Setup the ML environment with specific configurations for fraud detection
        clf = setup(
            data=data,
            target=target,
            session_id=123,
            train_size=train_size,
            verbose=False,
            experiment_name=self.experiment_name,
            log_experiment=False,
            log_plots=False,
            fix_imbalance=True,  # Important for fraud detection (imbalanced dataset)
            remove_multicollinearity=True,  # Remove highly correlated features
            multicollinearity_threshold=0.95,
            normalize=True,  # Normalize features (Amount might need scaling)
            transformation=True,  # Apply transformations if needed
            #ignore_low_variance=True,  # Remove low variance features
            #combine_rare_levels=True,  # Combine rare categorical levels
            #rare_level_threshold=0.1,
            bin_numeric_features=['Amount']  # Bin the Amount feature
        )
        
        self.setup_complete = True
        print("✅ ML environment setup complete for fraud detection!")
        return clf
    
    def compare_fast_models(self, fold=5):
        """
        Compare two fast models for fraud detection: Logistic Regression and Decision Tree.
        """
        if not self.setup_complete:
            raise Exception("Setup must be completed before comparing models")
            
        print("Comparing two fast models: Logistic Regression and Decision Tree...")
        
        best_model = compare_models(
            include=['lr', 'dt'],
            fold=fold,
            sort='AUC'
        )
        
        print("✅ Model comparison complete!")
        return best_model
    
    def finalize_model(self, model):
        """
        Finalize the model (trains on entire dataset)
        """
        print("Finalizing fraud detection model...")
        
        final_model = finalize_model(model)
        self.model = final_model
        
        print("✅ Model finalized!")
        return final_model
    
    def evaluate_model(self, model):
        """
        Evaluate the model performance with fraud detection specific metrics
        """
        print("Evaluating fraud detection model...")
        
        # Evaluate the model
        evaluation = evaluate_model(model)
        
        # Get predictions on test set
        predictions = predict_model(model)
        
        # Print additional fraud detection metrics
        print("\n📊 Fraud Detection Performance Summary:")
        print("-" * 40)
        
        # Calculate and display key metrics for fraud detection
        from sklearn.metrics import classification_report, confusion_matrix
        
        if hasattr(predictions, 'Class') and hasattr(predictions, 'prediction_label'):
            y_true = predictions['Class']
            y_pred = predictions['prediction_label']
            
            print("\nClassification Report:")
            print(classification_report(y_true, y_pred))
            
            print("\nConfusion Matrix:")
            cm = confusion_matrix(y_true, y_pred)
            print(cm)
            
            # Calculate fraud detection specific metrics
            tn, fp, fn, tp = cm.ravel()
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            print(f"\n🎯 Fraud Detection Metrics:")
            print(f"   Precision: {precision:.4f}")
            print(f"   Recall: {recall:.4f}")
            print(f"   F1-Score: {f1:.4f}")
        
        print("✅ Model evaluation complete!")
        return evaluation, predictions
    
    def save_model(self, model, model_name=None):
        """
        Save the trained fraud detection model
        """
        if model_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_name = f"fraud_detection_model_{timestamp}"
        
        # Create models directory if it doesn't exist
        os.makedirs("models", exist_ok=True)
        
        # Save using PyCaret
        save_model(model, f"models/{model_name}")
        
        # Also save as pickle for Flask API
        joblib.dump(model, f"models/{model_name}.pkl")
        
        print(f"✅ Fraud detection model saved as {model_name}")
        return model_name
    
    def run_complete_pipeline(self, csv_file_path="creditcard_2023.csv"):
        """
        Run the complete fraud detection ML pipeline
        """
        print("🚀 Starting Credit Card Fraud Detection Pipeline...")
        print("=" * 60)
        
        # Step 1: Load Data
        data, target = self.load_cc_data(csv_file_path)
        if data is None:
            print("❌ Pipeline failed: Could not load data")
            return None
            
        print(f"Dataset shape: {data.shape}")
        print(f"Target variable: {target}")
        
        # Step 2: Setup Environment
        self.setup_ml_environment(data, target)
        
        # Step 3: Compare Fast Models
        best_model = self.compare_fast_models()

        # Step 4: Finalize Model
        final_model = self.finalize_model(best_model)
        
        # Step 7: Evaluate Model
        evaluation, predictions = self.evaluate_model(final_model)
        
        # Step 8: Save Model
        model_name = self.save_model(final_model)
        
        print("=" * 60)
        print("🎉 Fraud Detection Pipeline Complete!")
        print(f"📁 Model saved as: {model_name}")
        
        return {
            'model': final_model,
            'model_name': model_name,
            'predictions': predictions,
            'data_shape': data.shape,
            'target': target,
            'class_distribution': data[target].value_counts().to_dict()
        }

# Usage Example
if __name__ == "__main__":
    # Initialize the fraud detection pipeline
    pipeline = CreditCardFraudPipeline("credit_card_fraud_mlops")
    
    # Run complete pipeline
    result = pipeline.run_complete_pipeline("creditcard_2023.csv")
    
    if result:
        print("\n" + "="*60)

        print("FRAUD DETECTION PIPELINE SUMMARY")
        print("="*60)
        print(f"✅ Model trained and saved: {result['model_name']}")
        print(f"✅ Dataset size: {result['data_shape']}")
        print(f"✅ Target variable: {result['target']}")
        print(f"✅ Class distribution: {result['class_distribution']}")
        print(f"✅ Ready for Flask API integration!")
        
        # Display model details
        print(f"\n📊 Model Type: {type(result['model'])}")
        print("🔄 Ready for deployment and real-time fraud detection!")
    else:
        print("❌ Pipeline execution failed. Please check your data file.")