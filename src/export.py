import setproctitle
from llamafactory.train.tuner import export_model, run_exp


def main():
    export_model()


if __name__ == "__main__":
    setproctitle.setproctitle("python")

    main()
