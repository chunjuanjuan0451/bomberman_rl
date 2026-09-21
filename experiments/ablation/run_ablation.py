"""Entry point for repeatable ablation runs (PER, augmentation, curriculum)."""

from pathlib import Path

CONFIG_DIR = Path(__file__).with_name("configs")


def main() -> None:
    print(f"Add experiment configs under {CONFIG_DIR}")


if __name__ == "__main__":
    main()
