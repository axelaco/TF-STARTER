import sys
from trainer.Constructor import Constructor
import tensorflow as tf
import numpy as np
def main(args):
    import Config
    config = Config.Config()
    constructor = Constructor(config)

    model = constructor.get_network()
    
    optimizer = constructor.get_optimizer()
    train_loader = constructor.get_train_loader()
    test_loader = constructor.get_test_loader()
    loss = constructor.get_loss()

    trainer = constructor.get_trainer(config, model, optimizer, loss, train_loader, test_loader)

    if config.resume:
        trainer.load_checkpoint(config.checkpoint_path)
    trainer.run()
    
if __name__ == '__main__':
    main(sys.argv[1:])