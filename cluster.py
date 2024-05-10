from tslearn.utils import to_time_series

my_first_time_series = [1,3,4,3]
formatted_time_series = to_time_series(my_first_time_series)

print(formatted_time_series.shape)
