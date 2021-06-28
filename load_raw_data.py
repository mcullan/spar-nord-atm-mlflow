"""
Downloads ATM data and concatenates part 1 and 2. Saves to an artifact.
"""

import os
import tempfile
import zipfile

import click
import mlflow
import requests


@click.command(
    help="Extracts feature data from raw atm data and stores it"
    "in an mlflow artifact called 'atm_csv_path'"
)
@click.option("--uri")
def load_raw_data(uri):
    """Extracts feature data from raw atm data and stores it
    in an mlflow artifact called 'transactions-csv-uri'"""

    with mlflow.start_run() as mlrun:

        with tempfile.TemporaryDirectory() as local_dir:
            # local_dir = tempfile.mkdtemp()
            local_filename = os.path.join(local_dir, "open_dataset.zip")


            print("Downloading %s to %s" % (uri, local_filename))
            r = requests.get(uri, stream=True)

            with open(local_filename, "wb") as f:
                for chunk in r.iter_content(chunk_size=1024):
                    if chunk:  # filter out keep-alive new chunks
                        f.write(chunk)
            print("Downloaded")

            print("Unzipping")
            with zipfile.ZipFile(local_filename, "r") as zip_ref:
                zip_ref.extractall(local_dir)
            print("Unzipped")

            part1 = os.path.join(local_dir, "atm_data_part1.csv")
            part2 = os.path.join(local_dir, "atm_data_part2.csv")

            atm_csv_path = os.path.join(local_dir, "atm_data.csv")

            print("Combining")
            with open(part1, "r") as f1, open(part2, "r") as f2, open(
                atm_csv_path, "w"
            ) as out:

                for line in f1:
                    out.write(line)

                # Discard header from part 2 csv
                header_2 = f2.readline()

                for line in f2:
                    out.write(line)

            print("Logging")
            mlflow.log_artifact(atm_csv_path, "atm_data")


if __name__ == "__main__":
    load_raw_data()
