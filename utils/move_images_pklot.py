import os
import shutil

directory = '../data/datasets/pklot/PKLotSegmented/'
class_list = ['empty', 'occupied']
parking_views = ['UFPR04', 'UFPR05', 'PUC']

output_directory = '../data/datasets/pklot/PKLotFormatted/'


def move_dataset_into_views():
    local_output_directory = output_directory + 'split_by_views/'
    # if file was in the format UFPR04/Sunny/2012-12-11/occupied/2012-12-11_17_11_09.jpg
    # it will be moved to the directory UFPR04/Sunny/occupied/2012-12-11_17_11_09.jpg
    for parking_view in parking_views:  # UFPR04, UFPR05, PUC
        parking_view_directory = os.path.join(directory, parking_view)
        for weather in os.listdir(parking_view_directory):  # Cloudy, Rainy, Sunny
            weather_directory = os.path.join(parking_view_directory, weather)
            for date in os.listdir(weather_directory):  # 2012-09-12, 2012-09-28, ...
                date_directory = os.path.join(weather_directory, date)
                for label in os.listdir(date_directory):  # empty, occupied
                    label_directory = os.path.join(date_directory, label)

                    if not os.path.exists(os.path.join(local_output_directory, parking_view, weather, label)):
                        os.makedirs(os.path.join(local_output_directory, parking_view, weather, label))

                    for filename in os.listdir(label_directory):  # 2012-09-28_07_16_00#001.jpg, ...
                        shutil.copy2(os.path.join(label_directory, filename),
                                     os.path.join(local_output_directory, parking_view, weather, label, filename))


def move_dataset_into_one():
    local_output_directory = output_directory + 'all_in_one/'
    # if file was in the format UFPR04/Sunny/2012-12-11/occupied/2012-12-11_17_11_09.jpg
    # it will be moved to the directory occupied/2012-12-11_17_11_09.jpg
    for parking_view in parking_views:
        parking_view_directory = os.path.join(directory, parking_view)
        for weather in os.listdir(parking_view_directory):
            weather_directory = os.path.join(parking_view_directory, weather)
            for date in os.listdir(weather_directory):
                date_directory = os.path.join(weather_directory, date)
                for label in os.listdir(date_directory):
                    label_directory = os.path.join(date_directory, label)

                    if not os.path.exists(os.path.join(local_output_directory, label)):
                        os.makedirs(os.path.join(local_output_directory, label))

                    for filename in os.listdir(label_directory):
                        shutil.copy2(os.path.join(label_directory, filename),
                                     os.path.join(local_output_directory, label, filename))


def move_dataset_into_one_weather():
    local_output_directory = output_directory + 'all_in_one_weather/'
    # if file was in the format UFPR04/Sunny/2012-12-11/occupied/2012-12-11_17_11_09.jpg
    # it will be moved to the directory Sunny/occupied/2012-12-11_17_11_09.jpg
    for parking_view in parking_views:
        parking_view_directory = os.path.join(directory, parking_view)
        for weather in os.listdir(parking_view_directory):
            weather_directory = os.path.join(parking_view_directory, weather)
            for date in os.listdir(weather_directory):
                date_directory = os.path.join(weather_directory, date)
                for label in os.listdir(date_directory):
                    label_directory = os.path.join(date_directory, label)

                    if not os.path.exists(os.path.join(local_output_directory, weather, label)):
                        os.makedirs(os.path.join(local_output_directory, weather, label))

                    for filename in os.listdir(label_directory):
                        shutil.copy2(os.path.join(label_directory, filename),
                                     os.path.join(local_output_directory, weather, label, filename))


def move_dataset_into_views_without_weather():
    local_output_directory = output_directory + 'split_by_views_without_weather/'
    # if file was in the format UFPR04/Sunny/2012-12-11/occupied/2012-12-11_17_11_09.jpg
    # it will be moved to the directory UFPR04/occupied/2012-12-11_17_11_09.jpg
    for parking_view in parking_views:
        parking_view_directory = os.path.join(directory, parking_view)
        for weather in os.listdir(parking_view_directory):
            weather_directory = os.path.join(parking_view_directory, weather)
            for date in os.listdir(weather_directory):
                date_directory = os.path.join(weather_directory, date)
                for label in os.listdir(date_directory):
                    label_directory = os.path.join(date_directory, label)

                    if not os.path.exists(os.path.join(local_output_directory, parking_view, label)):
                        os.makedirs(os.path.join(local_output_directory, parking_view, label))

                    for filename in os.listdir(label_directory):
                        shutil.copy2(os.path.join(label_directory, filename),
                                     os.path.join(local_output_directory, parking_view, label, filename))


def main():
    print("Uncomment the function you want to use in the main function")
    # move_dataset_into_views()
    # move_dataset_into_one()
    # move_dataset_into_one_weather()
    # move_dataset_into_views_without_weather()


if __name__ == '__main__':
    main()
