import pytest


@pytest.fixture
def replace_pfxobs_attrs():
    # Replaces the fields of each processed forecast and obs with kwargs
    def fn(report_obj, **kwargs):
        raw_report = report_obj.raw_report
        missing_data = report_obj.replace(
            raw_report=raw_report.replace(
                processed_forecasts_observations=tuple(
                    pfxobs.replace(**kwargs)
                    for pfxobs in raw_report.processed_forecasts_observations
                )
            )
        )
        return missing_data
    return fn
