"""Run Aureas with the settings in config.yaml (like bacpipe's run_pipeline.py).

    python run_pipeline.py            # explore audio_dir, then open the dashboard
"""
import aureas
from aureas.cli import _dashboard

cfg = aureas.load_config("config.yaml")
res = aureas.explore(config=cfg)
print(f"{res.meta['n_segments']} segments in {len(res.clusters) - 1} classes -> {res.path}")
_dashboard(res.path)
