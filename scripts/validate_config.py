import argparse

from laglayer3.config import load_yaml_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    args = parser.parse_args()
    config = load_yaml_config(args.config)
    required = {"runtime", "route", "head", "execution", "flight_control", "simulation", "training"}
    missing = required.difference(config)
    if missing:
        raise SystemExit("missing config sections: {}".format(sorted(missing)))
    print("valid configuration: {}".format(args.config))


if __name__ == "__main__":
    main()
