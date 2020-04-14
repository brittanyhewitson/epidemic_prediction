# Set the random seed for splitting data
RANDOM_SEED = 42

REGEX = r"^\d+-\d+-\d+.csv$"

DATA_CHOICES = [
    "big_data",
    "small_data",
    "smote",
    "by_date",
    "single_date",
]

ZIKA_DATAFIELD_TO_KEEP = [
    "zika_confirmed_laboratory",
    "zika_reported_local",
    "total_zika_confirmed_autochthonous",
    #"total_zika_confirmed_imported",
    "zika_lab_positive",
    "cumulative_confirmed_local_cases",
    #"weekly_zika_confirmed",
    "gbs_confirmed_cumulative",
]

TEMP_FIELDS = [
    "max_temp", 
    "max_temp1", 
    "max_temp2",
    "mean_temp",
    "mean_temp1",
    "mean_temp2",
    "min_temp",
    "min_temp1",
    "min_temp2",
    "dew_point",
    "dew_point1",
    "dew_point2"
]

PRECIP_FIELDS = [
    "precipitation",
    "precipitation1",
    "precipitation2"
]

DISTANCES_FIELDS = [
    "airport_dist_any",
    "airport_dist_large",
    "mosquito_dist"
]