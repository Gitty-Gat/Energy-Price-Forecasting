from __future__ import annotations

import pandera as pa
from pandera import Check


raw_price_schema = pa.DataFrameSchema(
    {
        "date": pa.Column(pa.DateTime, nullable=False),
        "price": pa.Column(float, checks=[Check.gt(0)], nullable=False),
    },
    strict=False,
    coerce=True,
)

weather_schema = pa.DataFrameSchema(
    {
        "date": pa.Column(pa.DateTime, nullable=False),
        "HDD": pa.Column(float, nullable=True),
        "CDD": pa.Column(float, nullable=True),
    },
    strict=False,
    coerce=True,
)

sentiment_schema = pa.DataFrameSchema(
    {
        "date": pa.Column(pa.DateTime, nullable=False),
        "sentiment_ng": pa.Column(float, nullable=True),
        "sentiment_ol": pa.Column(float, nullable=True),
    },
    strict=False,
    coerce=True,
)

merged_exog_schema = pa.DataFrameSchema(
    {
        "date": pa.Column(pa.DateTime, nullable=False),
        "PRICE_NG": pa.Column(float, checks=[Check.gt(0)], nullable=False),
        "PRICE_OL": pa.Column(float, checks=[Check.gt(0)], nullable=False),
    },
    strict=False,
    coerce=True,
)

forecast_schema = pa.DataFrameSchema(
    {
        "step": pa.Column(object, nullable=False),
        "ng_return_mean": pa.Column(float, nullable=False),
        "ng_return_lower": pa.Column(float, nullable=False),
        "ng_return_upper": pa.Column(float, nullable=False),
        "ol_return_mean": pa.Column(float, nullable=False),
        "ol_return_lower": pa.Column(float, nullable=False),
        "ol_return_upper": pa.Column(float, nullable=False),
    },
    strict=False,
    coerce=True,
)

backtest_schema = pa.DataFrameSchema(
    {
        "date": pa.Column(pa.DateTime, nullable=False),
        "actual": pa.Column(float, nullable=False),
        "predicted": pa.Column(float, nullable=False),
    },
    strict=False,
    coerce=True,
)
