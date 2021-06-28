"""
Converts the raw CSV form to aggregate_atms features for training models.
"""
import tempfile
import os

import click
import mlflow
import pandas as pd

def get_df_prophet(transactions, freq="D", aggregate_atms=True, fill_dates=True):
    """Prepares transactions data for clustering and forecasting models.
    Aggregates to get total transactions according to frequency, e.g. "D" for daily.
    If aggregate_atms is True, then this returns total daily transactions.
    Otherwise, it returns daily transactions per ATM.,
    """

    if aggregate_atms:
        grouping =  pd.Grouper(key="timestamp", freq=freq)
    else:
        grouping = ["atm_id", pd.Grouper(key="timestamp", freq=freq)]

    daily_demand = (transactions
            .groupby(grouping)["currency"]
            .count()
            .reset_index())

    if fill_dates:
        dates = (pd.date_range(
            start=daily_demand["timestamp"].min(), end=daily_demand["timestamp"].max()
        ).rename("timestamp"))

        if aggregate_atms:
            daily_demand = (
                daily_demand.set_index("timestamp")
                .reindex(dates, fill_value=0)
                .reset_index()
            )
        else:
            daily_demand = (
                daily_demand.groupby("atm_id")
                .apply(
                    lambda x: (
                        x.drop(columns=["atm_id"])
                        .set_index("timestamp")
                        .reindex(dates, fill_value=0)
                    )
                )
                .reset_index()
            )

    # Prophet requires time series data to use these column names
    df_prophet = daily_demand.rename(columns={"timestamp": "ds", "currency": "y"})

    return df_prophet


@click.command(
    help="Extracts feature data from raw atm data and stores it"
    "in an mlflow artifact called 'transactions-csv-uri'"
)
@click.option("--atm_csv_uri", type=str)
@click.option("--model_name", type=str)
@click.option("--aggregate_atms", type=int)
@click.option("--train_split_date", type=str)
def process_data(atm_csv_uri, model_name, aggregate_atms, train_split_date):
    with mlflow.start_run() as mlrun:

        print("process data: loading df")
        df = pd.read_csv(atm_csv_uri)
        df = df[df["atm_status"] != "Inactive"]
        print("process data: df loaded")

        df["timestamp"] = pd.to_datetime(
            (
                df["year"].astype(str)
                + " "
                + df["month"]
                + " "
                + df["day"].astype(str)
                + " "
                + df["hour"].astype(str)
            ),
            format="%Y %B %d %H",
        )

        # Will be used in a future version to get proportion of transactions by customers.
        # df["on-us"] = df["card_type"].str.contains("on-us")

        df["in_bank"] = df["atm_location"].str.contains("Intern")

        atm_cols = [
            "atm_id",
            "atm_status",
            "atm_manufacturer",
            "atm_location",
            "atm_streetname",
            "atm_street_number",
            "atm_zipcode",
            "atm_lat",
            "atm_lon",
            "weather_lon",
            "weather_lat",
            "weather_city_name",
            "weather_city_id",
            "in_bank",
        ]

        transaction_cols = [
            "atm_id",
            "currency",
            "card_type",
            "message_code",
            "timestamp",
        ]

        print("process data: splitting df")
        atms = df[atm_cols]

        transactions = df[transaction_cols]

        print("process data: logging csv")

        df_prophet = get_df_prophet(
            transactions, "D", aggregate_atms, fill_dates=True
        )

        mlflow.log_text(df_prophet.to_csv(index=False), "df_prophet.csv")


if __name__ == "__main__":
    process_data()
