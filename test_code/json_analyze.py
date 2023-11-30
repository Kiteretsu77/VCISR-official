
import os, json
import pandas as pd
import collections


def check_has_sub_dictionary(dict):
    for key in dict:
        if type(dict[key]) is type(dict):
            return True
    return False


def analyze_json(dir, specfic_model_needed):
    ''' Analyze metric_result.json to organize all value in a single pandas
    Args:
        dir (str): Path to the metric folder
        specfic_model_needed ([str]): a list of models we want to print
    Returns:
        mass_report (dict): dictionary report
    '''

    if os.path.exists(dir):
        with open(dir) as json_file:
            metric_result = json.load(json_file)
    else:
        print("No such path {} exists ".format(dir))
        os._exit(0)

    # Init the dictionary we want to show
    mass_report = collections.defaultdict(dict)

    for model_name in metric_result:
        if len(specfic_model_needed) != 0 and '_'.join(model_name.split('_')[:-1]) not in specfic_model_needed:
            # Only need those in the lists
            continue

        metric_report = collections.defaultdict(dict)
        for dataset_path in metric_result[model_name]:
            if '/' in dataset_path:
                dataset_name = dataset_path.split('/')[-1]
            elif '\\' in dataset_path:
                dataset_name = dataset_path.split('\\')[-1]


            # Check if the subdirectory is still dict; if is, they belong to the same dataset
            if check_has_sub_dictionary(metric_result[model_name][dataset_path]):
                temp_store = collections.defaultdict(list)
                # Recursively calculate for each metric and find the average
                for sub_dataset in metric_result[model_name][dataset_path]:
                    for metric, value in metric_result[model_name][dataset_path][sub_dataset].items():
                        temp_store[metric].append(value)

                # Calculate the average of each video sequence
                for metric in temp_store:
                    value_lists = temp_store[metric]
                    metric_report[dataset_name][metric] = sum(value_lists)/len(value_lists)

            else:
                # Single metric, just copy the metric and that's ok
                for metric in metric_result[model_name][dataset_path]:
                    metric_report[dataset_name][metric] = metric_result[model_name][dataset_path][metric]
        
        mass_report[model_name] = metric_report

    return mass_report


def save_as_csv(mass_report, csv_name):
    ''' Save report in csv file
    Args:
        mass_report (dict): dictionary report
        csv_name (str): the csv file name we want to store as
    '''
    lower_better_metrics = ["niqe", "brisque", "pi"]
    # lower_better_idx = 2        # 前几个是lower is better （这里规定一个norm让计算变得更加简单）

    # Save format
    csv_info = collections.defaultdict(dict)

    # Sort the model name
    sorted_model_name = []
    for model_name in mass_report:
        sorted_model_name.append(model_name)
    
    # Rename the report
    for model_name in sorted(sorted_model_name):
        for dataset_name in mass_report[model_name]:
            collected_info = ""
            for metric in mass_report[model_name][dataset_name]:
                collected_info += metric.upper() + " " + str(round(mass_report[model_name][dataset_name][metric], 3)) +"/ "

            # Put inside csv
            csv_info[dataset_name][model_name] = collected_info
    

    # Obtain the best result among all
    for dataset_name in csv_info:
        temp_store = {}

        # Find the best in all models
        for model_name in sorted(sorted_model_name):
            
            for idx, metric in enumerate(mass_report[model_name][dataset_name]):
                if metric in lower_better_metrics:
                    # This means that this metric is lower better
                    if metric not in temp_store:
                        temp_store[metric] = [float('inf'), ""]

                    # Choose the minimum one
                    if mass_report[model_name][dataset_name][metric] < temp_store[metric][0]:
                        temp_store[metric] = [mass_report[model_name][dataset_name][metric], model_name]
                else:
                    # This means that this metric is higher better
                    if metric not in temp_store:
                        temp_store[metric] = [float('-inf'), ""]

                    # Choose the maximum one
                    if mass_report[model_name][dataset_name][metric] > temp_store[metric][0]:
                        temp_store[metric] = [mass_report[model_name][dataset_name][metric], model_name]


        # Transform the best result and save them inside csv_info
        info = ""
        for metric in temp_store:
            info += metric.upper() + " \t" + str(round(temp_store[metric][0], 3)) + " (" + temp_store[metric][1] + ")\n"
        # csv_info[dataset_name]["Best among all"] = info 


    # Transform and store as csv 
    df = pd.DataFrame(csv_info)
    df.to_csv(csv_name)


def main(file_name, specfic_model_needed):
    ''' main function files
    Args:
        specfic_dir_needed ([str]): a list of models we want to print
    '''
    # Analyze the json
    mass_report = analyze_json(file_name, specfic_model_needed)

    # Save the result in csv
    save_as_csv(mass_report, "metric_analysis.csv")


if __name__ == "__main__":

    file_name = "metric_result_paper.json"
    specfic_model_needed = [
        # "4x_bsrgan_official",
        # "4x_esrgan_with_usm_official",
        # # "4x_swinirGANM_official",
        # "4x_swinirGANL_official",
        # "4x_GRLBASE_official",
        "4x_RealBasicVSR_official",

        # "4x_esrnet_v2_with_usm_without_weakened_with_v1skip_sample_training",
        # "4x_esrgan_v2_with_usm_without_weakened_with_v1skip_gan_weight=0.05",
        # "4x_GRL_v2_with_usm_without_weakened_with_v1skip",
        # "4x_GRLGAN_v2_with_usm_without_weakened_with_v1skip",
    ]

    main(file_name, specfic_model_needed)

    

    