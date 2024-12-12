import multiprocessing
import time
from CNN_class import CNN
from Data.load_data import load_mri_data, train_test_split 


def train_and_evaluate_model(model_name, train_set, val_set, test_set, num_classes, labels_map, image_size, epochs, learning_rate,batch_size):
    print(f"\nTraining {model_name} model...\n")
    
    # Initialize the CNN model
    cnn_model = CNN(
        train=train_set,
        validation=val_set,
        test=test_set,
        num_classes=num_classes,
        model=model_name,
        labels_map=labels_map,
        image_size=image_size  # Only applicable for VisionTransformer
    )

    # Train the model
    print(f"Training {model_name}...")
    cnn_model.train_model(
        epochs=epochs,
        learning_rate=learning_rate,
        optimizer="Adam"  
    )

    histories = cnn_model.history

    # Evaluate the model and plot metrics
    print(f"Evaluating {model_name}...")
    cnn_model.get_metrics_and_plots(saving=True, path= "Data/Results/parallell_test" ,confusion_matrix=True, classification_report=True, plot_history=True, plot_img=True, indexes=(0, 4), prediction=True)

    # Predict using XGBoost
    print(f"Predicting using XGBoost for {model_name}...")
    cnn_model.predict_xgb()

    return model_name, histories


def main():
    num_classes = 4
    epochs = 4
    learning_rate = 0.01
    batch_size = 64
    base_path = "Data/archive"
    image_size = (96, 96)

    # Load and preprocess the MRI data
    print("Loading data...")
    total_dataset, labels_map = load_mri_data(base_path, batch_size=batch_size, image_size=image_size)

    # Split into training and testing datasets
    train_set, val_set, test_set = train_test_split(total_dataset, test_fraction=0.2, batch_size=batch_size)

    # List of models to train and evaluate
    models = ["Densenet", "Resnet", "EfficientNet", "VisionTransformer"]
    all_histories = {}

    # Create a pool of processes to handle parallel training and evaluation
    with multiprocessing.Pool(processes=2) as pool:
        for i in range(0, len(models), 2):
            # Take two models from the list
            models_batch = models[i:i+2]
            
            # Prepare arguments for each model
            args_list = [
                (model_name, train_set, val_set, test_set, num_classes, labels_map, image_size, epochs, learning_rate, batch_size)
                for model_name in models_batch
            ]
            
            # Run the training and evaluation for two models at a time
            result = pool.starmap(train_and_evaluate_model, args_list)
            
            # Store results
            for model_name, history in result:
                all_histories[model_name] = history
            print(f"\nFinished training and evaluating {models_batch[0]} and {models_batch[1]}.")

    # All models trained and evaluated
    print("\nAll models trained and evaluated!")
    

if __name__ == "__main__":
    start_time = time.time()
    main()
    print(f"Execution time: {time.time() - start_time:.2f} seconds")
