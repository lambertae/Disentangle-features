from src.models.autoencoder import AutoEncoder
from src.models.deep_autoencoder import DeepAutoEncoder
from src.models.direct_solver import DirectSolver

# All models
ALL_MODELS = {
    "AutoEncoder": AutoEncoder,
    "DeepAutoEncoder": DeepAutoEncoder,
    "DirectSolver": DirectSolver
}