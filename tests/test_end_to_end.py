import json

import pytest

from racing import config, dagger, es, evaluate, expert, ga, imitation, nets, paths
from racing.backend import keras_cpu


@pytest.mark.slow
def test_pipeline_end_to_end(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DAGGER_MAPS", 2)
    monkeypatch.setattr(config, "DAGGER_EPOCHS", 2)
    runs, champions, data_path = tmp_path / "runs", tmp_path / "champions", tmp_path / "dataset.npz"
    splits = {"train": [(0, 1), (0, 2)], "val": [(80, config.VALIDATION_LEVEL)], "test": [(0, config.TEST_LEVEL)]}
    data, _ = expert.collect_dataset(data_path, splits=splits, n_jobs=2)
    for seed in (0, 1):
        imitation.train_baseline(data, out_dir=paths.run_dir("baseline", seed, runs),
                                 epochs=2, seed=seed)
    imitation.run_tuner("kt_random", data, out_dir=paths.search_dir("kt_random", runs), max_trials=2, max_epochs=2)
    imitation.retrain("kt_random", data, seed=0, search=paths.search_dir("kt_random", runs),
                      out_dir=paths.run_dir("kt_random", 0, runs), epochs=2)
    ga.run_ga(pop_size=20, max_generations=2, seed=0, out_dir=paths.run_dir("ga", 0, runs), champion_dir=champions)
    es.run_es(pop_size=20, max_generations=2, seed=0, out_dir=paths.run_dir("es", 0, runs), champion_dir=champions)
    for approach, seed in (("baseline", 0), ("baseline", 1), ("kt_random", 0)):
        dagger.refine(approach, seed, data, runs_dir=runs, max_rounds=1)
        refined = keras_cpu().models.load_model(paths.stage_path(approach, seed, "dagger", runs), compile=False)
        hidden, activation = nets.dense_spec(refined)
        # what dagger.speed_push does, with 20 cars instead of config.ES_POP
        es.run_es(pop_size=20, max_generations=1, seed=seed, start_level=config.MAX_LEVEL,
                  initial=nets.to_flat(refined), approach=approach, hidden=hidden, activation=activation,
                  out_dir=paths.speed_dir(approach, seed, runs), champion_dir=champions)
    es.run_es(pop_size=20, max_generations=2, seed=1, initial=es.dagger_baseline(1, runs_dir=runs),
              approach="es_seeded", out_dir=paths.run_dir("es_seeded", 1, runs), champion_dir=champions)
    results = evaluate.main(champions_dir=champions, runs_dir=runs, data_path=data_path, out_dir=tmp_path / "out",
                            test_seeds=[0], val_seeds=[80], n_jobs=2, latency_calls=20)
    assert list(results["comparison"]["Method"]) == ["Baseline", "KT Random", "ES", "GA", "Seeded ES"]
    assert list(results["driving_selected"]["Method"]) == ["KT Random"]
    for name in ("results", "head_to_head", "training_cost", "stages", "comparison", "search_cost",
                 "driving_selected", "dataset"):
        assert (tmp_path / "out" / "tables" / f"{name}.md").exists()
    best = json.loads((champions / "best_seeds.json").read_text())
    assert best["baseline"] in (0, 1) and best["es_seeded"] == 1
    assert json.loads((paths.run_dir("ga", 0, runs) / "summary.json").read_text())["evals"] == 40
