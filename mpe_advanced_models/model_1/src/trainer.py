import torch
from loguru import logger
from nn_scripts import train, validate
from test_and_eval import test_and_process_outputs

from lib_mpe.project_utils.experiment_utils import test_assertions


class Trainer:
    def __init__(self, cfg, fabric, data_manager, model_manager):
        """
        Initializes the Trainer class.

        Args:
            cfg (ConfigManager): Configuration manager.
            fabric (Fabric): Fabric object.
            data_manager (DataManager): Data manager.
            model_manager (ModelManager): Model manager.
        """
        self.cfg = cfg
        self.data_manager = data_manager
        self.model_manager = model_manager
        self.best_loss = float("inf")
        self.counter = 0
        self.patience = 5  # Number of epochs to wait for the validation loss to improve
        self.all_train_losses = []
        self.all_val_losses = []
        self.best_model_info = data_manager.best_model_info
        self.fabric = fabric

    def train_model(self):
        """
        Trains the model.
        """
        logger.info("First validation")
        self.validate()

        logger.info("Training the model")
        for epoch in range(1, self.cfg.epochs + 1):
            train_loss = self.train_loop(epoch)

            self.all_train_losses.append(train_loss)

            if epoch % 1 == 0:
                self.validate()

            self.update_lr_scheduler(train_loss)

            if self.counter >= self.patience:
                print(
                    f"Validation loss hasn't improved for {self.patience} epochs, stopping training..."
                )
                break

        logger.info("Training completed!")
        self.save_best_model()

    def validate(self):
        """
        Validates the model.
        """
        (
            self.best_loss,
            val_loss,
            self.counter,
            all_unprocessed_data,
            all_nn_outputs,
            _,
            _,
        ) = validate(
            self.cfg,
            self.model_manager.model,
            self.data_manager.torch_pgm,
            self.cfg.device,
            self.data_manager.val_loader,
            self.best_loss,
            self.counter,
        )
        self.all_val_losses.append(val_loss)

        if val_loss < getattr(self, "best_val_loss", float("inf")):
            self.best_val_loss = val_loss
            self.best_model_info = {
                "epoch": len(self.all_val_losses),
                "model_state": self.model_manager.model.state_dict(),
                "optimizer_state": self.model_manager.optimizer.state_dict(),
            }

    def train_loop(self, epoch):
        """
        Trains the model.
        """
        return train(
            self.cfg,
            self.model_manager.model,
            self.data_manager.torch_pgm,
            self.cfg.device,
            self.fabric,
            self.data_manager.train_loader,
            self.model_manager.optimizer,
            epoch,
            schedular=self.model_manager.lr_scheduler,
        )

    def update_lr_scheduler(self, train_loss):
        """
        Updates the learning rate scheduler.
        """
        if self.cfg.epochs > 2 and self.model_manager.lr_scheduler is not None:
            if not isinstance(
                self.model_manager.lr_scheduler,
                (
                    torch.optim.lr_scheduler.CyclicLR,
                    torch.optim.lr_scheduler.OneCycleLR,
                ),
            ):
                if isinstance(
                    self.model_manager.lr_scheduler,
                    torch.optim.lr_scheduler.ReduceLROnPlateau,
                ):
                    self.model_manager.lr_scheduler.step(self.all_val_losses[-1])
                else:
                    self.model_manager.lr_scheduler.step()
        print("Learning rate:", self.model_manager.optimizer.param_groups[0]["lr"])

    def save_best_model(self):
        """
        Saves the best model.
        """
        logger.info("Saving model...")
        logger.info(f"Best model saved at {self.cfg.model_dir}/model.pt")
        torch.save(self.best_model_info, f"{self.cfg.model_dir}/model.pt")
        self.cfg.model_path = f"{self.cfg.model_dir}/model.pt"

    def test_and_save_model(self):
        """
        Tests the model and saves the model.
        """
        test_assertions(self.cfg, self.best_model_info)
        if self.best_model_info["epoch"] is None:
            # If the model has not been trained - take a random model
            self.best_model_info = {
                "epoch": 0,
                "model_state": self.model_manager.model.state_dict(),
                "optimizer_state": self.model_manager.optimizer.state_dict(),
            }
        test_and_process_outputs(
            self.cfg,
            self.cfg.device,
            self.fabric,
            self.data_manager.torch_pgm,
            self.data_manager.torch_pgm_initial,
            self.cfg.model_dir,
            self.cfg.model_outputs_dir,
            self.data_manager.train_loader,
            self.data_manager.test_loader,
            self.data_manager.val_loader,
            self.data_manager.mpe_solutions,
            self.model_manager.model,
            self.model_manager.optimizer,
            self.best_loss,
            self.counter,
            self.all_train_losses,
            self.all_val_losses,
            self.best_model_info,
            self.data_manager.num_data_features,
            self.data_manager.num_pgm_feature,
            self.data_manager.num_outputs,
            self.data_manager.num_query_variables,
        )

    def return_info(self):
        """
        Returns the information.
        """
        return (
            self.best_loss,
            self.counter,
            self.all_train_losses,
            self.all_val_losses,
            self.best_model_info,
        )
