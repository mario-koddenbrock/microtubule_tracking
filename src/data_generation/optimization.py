import optuna
from transformers import SegformerFeatureExtractor, SegformerModel

from data_generation.config import TuningConfig, SyntheticDataConfig
from data_generation.metric import load_reference_embeddings, evaluate


def main():
    tuning_cfg = TuningConfig.from_json("tuning_config.json")
    tuning_cfg.validate()

    model = SegformerModel.from_pretrained(tuning_cfg.model_name)
    extractor = SegformerFeatureExtractor.from_pretrained(tuning_cfg.model_name)
    ref_embeddings = load_reference_embeddings(tuning_cfg, model, extractor)

    def objective(trial):
        trial_cfg = {
            "grow_amp": trial.suggest_float("grow_amp", *tuning_cfg.grow_amp_range),
            "grow_freq": trial.suggest_float("grow_freq", *tuning_cfg.grow_freq_range),
            "shrink_amp": trial.suggest_float("shrink_amp", *tuning_cfg.shrink_amp_range),
            "shrink_freq": trial.suggest_float("shrink_freq", *tuning_cfg.shrink_freq_range),
            "motion": trial.suggest_float("motion", *tuning_cfg.motion_range),
            "max_length": trial.suggest_float("max_length", *tuning_cfg.max_length_range),
            "min_length": trial.suggest_float("min_length", *tuning_cfg.min_length_range),
            "snr": trial.suggest_float("snr", *tuning_cfg.snr_range),
            "sigma": [trial.suggest_float("sigma", *tuning_cfg.sigma_range)] * 2,
            "num_tubulus": int(trial.suggest_int("num_tubulus", *tuning_cfg.num_tubulus_range)),
        }
        return evaluate(trial_cfg, ref_embeddings, tuning_cfg, model, extractor)

    study = optuna.create_study(direction=tuning_cfg.direction)
    study.optimize(objective, n_trials=tuning_cfg.num_trials)

    best_cfg = SyntheticDataConfig(**study.best_trial.params)
    best_cfg.to_json("best_synthetic_config.json")
    print("Best config saved to best_synthetic_config.json")


if __name__ == "__main__":
    main()
