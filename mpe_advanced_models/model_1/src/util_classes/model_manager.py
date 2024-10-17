from lib_mpe.project_utils.model_utils import select_lr_scheduler
from loguru import logger
from mpe_advanced_models.model_1.src.get_model import init_model_and_optimizer


class ModelManager:
    def __init__(
        self,
        cfg,
        device,
        fabric,
        num_data_features,
        num_pgm_feature,
        num_outputs,
        num_query_variables,
        train_loader,
    ):
        """
        Initializes the ModelManager class.

        Args:
            cfg (ConfigManager): Configuration manager.
            library_pgm (str): Library used for PGM.
            device (str): Device to run the model on.
            fabric (Fabric): Fabric object.
            num_data_features (int): Number of data features.
            num_pgm_feature (int): Number of PGM features.
            num_outputs (int): Number of outputs.
            num_query_variables (int): Number of query variables.
            train_loader (DataLoader): DataLoader object for training data.
        """
        self.cfg = cfg
        self.model = None
        self.optimizer = None
        self.teacher_model = None
        self.teacher_optimizer = None
        self.lr_scheduler = None
        self.init_model_and_optimizer(
            device,
            fabric,
            num_data_features,
            num_pgm_feature,
            num_outputs,
            num_query_variables,
        )
        self.init_lr_scheduler(train_loader)

    def return_info(self):
        """
        Returns the information about the model.

        Returns:
            tuple: Model, optimizer, teacher model, teacher optimizer, and learning rate scheduler.
        """
        return (
            self.model,
            self.optimizer,
            self.teacher_model,
            self.teacher_optimizer,
            self.lr_scheduler,
        )

    def init_model_and_optimizer(
        self,
        device,
        fabric,
        num_data_features,
        num_pgm_feature,
        num_outputs,
        num_query_variables,
    ):
        """
        Initializes the model and optimizer.

        Args:
            cfg (ConfigManager): Configuration manager.
            library_pgm (str): Library used for PGM.
            device (str): Device to run the model on.
            fabric (Fabric): Fabric object.
            num_data_features (int): Number of data features.
            num_pgm_feature (int): Number of PGM features.
            num_outputs (int): Number of outputs.
            num_query_variables (int): Number of query variables.
        """
        models_and_optimizers = init_model_and_optimizer(
            self.cfg,
            device,
            fabric,
            num_data_features,
            num_pgm_feature,
            num_outputs,
            num_query_variables=num_query_variables,
            run_type="train",
        )

        self.model, self.optimizer = models_and_optimizers
        logger.info(f"Model: {self.model}")

    def init_lr_scheduler(self, train_loader):
        """
        Initializes the learning rate scheduler.

        Args:
            train_loader (DataLoader): DataLoader object for training data.
        """
        self.lr_scheduler = select_lr_scheduler(
            self.cfg, self.cfg.lr_scheduler, self.optimizer, train_loader, verbose=True
        )
