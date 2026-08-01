"""Global test isolation for configuration loaded during module imports."""

import os

# API tests provide mocked pipelines. Prevent a developer's ignored local
# meeting_config.yaml from downloading/loading real models during collection.
os.environ.setdefault("MEETASR_CONFIG", "tests/.missing-meeting-config.yaml")
os.environ.setdefault("DATABASE_URL", "sqlite://")
