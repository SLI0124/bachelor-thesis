import os
import shutil

directory = '../data/datasets/pklot/PKLotSegmented/'
parking_views = ['UFPR04', 'UFPR05', 'PUC']


# output_directory = '../data/datasets/pklot/PKLotFormatted/'


# def move_dataset_into_views():
#     local_output_directory = output_directory + 'split_by_views/'
#     # if file was in the format UFPR04/Sunny/2012-12-11/occupied/2012-12-11_17_11_09.jpg
#     # it will be moved to the directory UFPR04/Sunny/occupied/2012-12-11_17_11_09.jpg
#     for parking_view in parking_views:  # UFPR04, UFPR05, PUC
#         parking_view_directory = os.path.join(directory, parking_view)
#         for weather in os.listdir(parking_view_directory):  # Cloudy, Rainy, Sunny
#             weather_directory = os.path.join(parking_view_directory, weather)
#             for date in os.listdir(weather_directory):  # 2012-09-12, 2012-09-28, ...
#                 date_directory = os.path.join(weather_directory, date)
#                 for label in os.listdir(date_directory):  # empty, occupied
#                     label_directory = os.path.join(date_directory, label)
#
#                     if not os.path.exists(os.path.join(local_output_directory, parking_view, weather, label)):
#                         os.makedirs(os.path.join(local_output_directory, parking_view, weather, label))
#
#                     for filename in os.listdir(label_directory):  # 2012-09-28_07_16_00#001.jpg, ...
#                         shutil.copy2(os.path.join(label_directory, filename),
#                                      os.path.join(local_output_directory, parking_view, weather, label, filename))


def move_dataset_into_one():
    print("Moving all images from PKLot to all/")

    output_directory = '../data/datasets/pklot_all/'

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    local_output_directory = output_directory + 'all/'
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

    # rename folders Empty to Free and Occupied to Busy
    os.rename(os.path.join(output_directory, 'all', 'empty'), os.path.join(output_directory, 'all', 'Free'))
    os.rename(os.path.join(output_directory, 'all', 'occupied'), os.path.join(output_directory, 'all', 'Busy'))

    print("Moved all images from PKLot to all/")


# def move_dataset_into_one_weather():
#     local_output_directory = output_directory + 'all_in_one_weather/'
#     # if file was in the format UFPR04/Sunny/2012-12-11/occupied/2012-12-11_17_11_09.jpg
#     # it will be moved to the directory Sunny/occupied/2012-12-11_17_11_09.jpg
#     for parking_view in parking_views:
#         parking_view_directory = os.path.join(directory, parking_view)
#         for weather in os.listdir(parking_view_directory):
#             weather_directory = os.path.join(parking_view_directory, weather)
#             for date in os.listdir(weather_directory):
#                 date_directory = os.path.join(weather_directory, date)
#                 for label in os.listdir(date_directory):
#                     label_directory = os.path.join(date_directory, label)
#
#                     if not os.path.exists(os.path.join(local_output_directory, weather, label)):
#                         os.makedirs(os.path.join(local_output_directory, weather, label))
#
#                     for filename in os.listdir(label_directory):
#                         shutil.copy2(os.path.join(label_directory, filename),
#                                      os.path.join(local_output_directory, weather, label, filename))


def move_dataset_into_views_without_weather():
    print("Moving all images from PKLot to pklot_puc, pklot_ufpr04, pklot_ufpr05")

    source_directory = '../data/datasets/pklot/PKLotSegmented/'
    dataset_views = os.listdir(source_directory)

    # I have PUC, UFPR04, UFPR05
    # puc directory will be moved into ../data/datasets/pklot_puc/
    # ufpr04 directory will be moved into ../data/datasets/pklot_ufpr04/
    # ufpr05 directory will be moved into ../data/datasets/pklot_ufpr05/

    destination_directory = ''
    for view in dataset_views:

        if view == 'PUC':
            destination_directory = '../data/datasets/pklot_puc/'
        elif view == 'UFPR04':
            destination_directory = '../data/datasets/pklot_ufpr04/'
        elif view == 'UFPR05':
            destination_directory = '../data/datasets/pklot_ufpr05/'

        print(f"Moving all images from PKLot to {destination_directory}")

        for weather in os.listdir(os.path.join(source_directory, view)):
            weather_directory = os.path.join(source_directory, view, weather)
            for date in os.listdir(weather_directory):
                date_directory = os.path.join(weather_directory, date)
                for label in os.listdir(date_directory):
                    label_directory = os.path.join(date_directory, label)

                    if not os.path.exists(os.path.join(destination_directory, label)):
                        os.makedirs(os.path.join(destination_directory, label))

                    for filename in os.listdir(label_directory):
                        shutil.copy2(os.path.join(label_directory, filename),
                                     os.path.join(destination_directory, label, filename))

        # rename folders Empty to Free and Occupied to Busy
        os.rename(os.path.join(destination_directory, 'empty'), os.path.join(destination_directory, 'Free'))
        os.rename(os.path.join(destination_directory, 'occupied'), os.path.join(destination_directory, 'Busy'))
        print(f"Moved all images from PKLot to {destination_directory}")

    # in puc, ufpr04, ufpr05 put Busy and Free folders into the all folder
    for view in dataset_views:
        if view == 'PUC':
            destination_directory = '../data/datasets/pklot_puc/'
        elif view == 'UFPR04':
            destination_directory = '../data/datasets/pklot_ufpr04/'
        elif view == 'UFPR05':
            destination_directory = '../data/datasets/pklot_ufpr05/'

        if not os.path.exists(os.path.join(destination_directory, 'all')):
            os.makedirs(os.path.join(destination_directory, 'all'))
        else:
            raise Exception("Directory already exists.")

        shutil.move(os.path.join(destination_directory, 'Free'), os.path.join(destination_directory, 'all', 'Free'))
        shutil.move(os.path.join(destination_directory, 'Busy'), os.path.join(destination_directory, 'all', 'Busy'))

    print("Moved all images from PKLot to pklot_puc, pklot_ufpr04, pklot_ufpr05")


def main():
    # move_dataset_into_views()
    # move_dataset_into_one_weather()
    move_dataset_into_one()
    move_dataset_into_views_without_weather()


if __name__ == '__main__':
    main()
