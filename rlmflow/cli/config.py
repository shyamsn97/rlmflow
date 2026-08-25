"""``rlmflow config``: what a run would use, where it came from, how to change it."""

from __future__ import annotations

from pathlib import Path

from rlmflow.cli.options import CONFIG_NAME, CliError, RunOptions, env_var, resolve, user_config

STARTER = """# rlmflow defaults for this directory. Flags still win, then the
# RLMFLOW_* environment variables, then this file.
[run]
model = "{model}"
fast_model = "{fast_model}"
max_iters = {max_iters}
"""


class ConfigCLI:
    """The settings `rlmflow run` resolves, and the files they come from."""

    def show(self) -> None:
        """Print every setting, its value, and which source won it."""
        options, origins = resolve()
        width = max(len(name) for name in origins)
        for name, origin in origins.items():
            print(
                f"{name:<{width}}  {str(getattr(options, name)):<24}  {origin:<15}  {env_var(name)}"
            )

    def path(self) -> None:
        """Print the config files that are read, in the order they are read."""
        for label, path in (("user", user_config()), ("project", Path.cwd() / CONFIG_NAME)):
            print(f"{label:<8} {path}  {'found' if path.is_file() else 'absent'}")

    def init(self, out: str = CONFIG_NAME) -> None:
        """Write a starter config file holding the settings in effect right now.

        Args:
          out: Where to write it. Defaults to ./rlmflow.toml.
        """
        path = Path(out).expanduser()
        if path.exists():
            raise CliError(f"{path} already exists")
        options: RunOptions = resolve()[0]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            STARTER.format(
                model=options.model,
                fast_model=options.fast_model,
                max_iters=options.max_iters,
            )
        )
        print(f"wrote {path}")


__all__ = ["ConfigCLI"]
