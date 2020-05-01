class Config:
    
    # If using a folder of images for training 
    train_data_path = ''
    # If using a folder of images for testing 
    test_data_path = ''

    # Num of epochs
    num_epochs = 5

    # Num of class to classify
    num_classes = 10

    # Specify optimizer 0: SGD, 1: RMSProp, 2: Adam
    optimizer_id = 2

    # Specify loss 0: BinaryCrossEntropy, 1: CategoricalCrossEntropy, 2: SparseCategoricalCrossentropy, 3: MSE
    loss_id = 2

    # Load strategy for training, 0: Trainer
    trainer_id = 0

    # Specify Learning rate for optimizer
    lr = 1e-3
    
    # Which Network to use, 0: MLP
    network_id = 0

    # Create a tensorboard session
    use_tensorboard = True

    # Dataset to use for training: 0: MNISTDataset
    train_loader_id = 0

    # Dataset to use for test: 0: MNISTDataset
    test_loader_id = 0


