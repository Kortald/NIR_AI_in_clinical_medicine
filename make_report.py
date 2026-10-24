# Сборка Excel-отчёта из сохранённых результатов (обучение не выполняется).
#
# Запуск:
#   python make_report.py                       отчёт из results/
#   python make_report.py --dir results/quick   отчёт из быстрого прогона
#
# Лист «Тест» добавляется, только если есть файл test_metrics.csv.

import argparse

from main import PREFIX, RESULTS_DIR
from src.report import build_report


def main(results_dir):
    path, sheets = build_report(results_dir, prefix=PREFIX)
    print(f"Отчёт: {path}")
    print("Листы: " + ", ".join(sheets))
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default=str(RESULTS_DIR),
                        help="папка с файлами l2_*.csv (по умолчанию results)")
    args = parser.parse_args()

    main(args.dir)
