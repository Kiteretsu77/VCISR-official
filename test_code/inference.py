import os, sys, cv2, shutil, json, warnings, collections, time
import torch
from torchvision.transforms import ToTensor
from torchvision.utils import save_image
warnings.simplefilter("default")
os.environ["PYTHONWARNINGS"] = "default"


# Import files from the local folder
root_path = os.path.abspath('.')
sys.path.append(root_path)
from opt import opt
from test_code.utils import load_grl



def file_check(file_dir):
    if not os.path.exists(file_dir):
        print(f"No such {file_dir} exists!")
        os._exit(0)


@torch.no_grad()
def testify(generator, input_dir, store_dir = None, scale = 2, crop_for_4x=False):
    ''' Test RRDB network int a directory
    Args:
        generator (torch):              the generator class that is already loaded
        input_dir (str):                the directory of the input lr images
        store_dir (str):                the directory to store generated images
        scale (int):                    the scalability we use for RRDB network
        crop_for_4x (bool):             whether we crop the lr images to match 4x scale
    '''

    # File check
    file_check(input_dir)


    # Load model weight
    if generator == None:
        print("The generator should not be None")
        os._exit(0)


    # Managae the store path
    if store_dir != None:

        if os.path.exists(store_dir):
            shutil.rmtree(store_dir)
        os.makedirs(store_dir, exist_ok = True)

    
    # Iterate each image in the directory and process one by one
    for idx, img_name in enumerate(sorted(os.listdir(input_dir))):
        if img_name.split('.')[-1] not in ['png', 'jpg']:
            # If they are not supported format, then we need to skip it
            continue

        print("Processing image {}".format(img_name))

        img_lr = cv2.imread(os.path.join(input_dir, img_name))
        # Crop if needed
        if crop_for_4x:
            h, w, _ = img_lr.shape
            if h % scale != 0:
                img_lr = img_lr[:scale*(h//scale),:,:]
            if w % scale != 0:
                img_lr = img_lr[:,:scale*(w//scale),:]


        img_lr = cv2.cvtColor(img_lr, cv2.COLOR_BGR2RGB)
        img_lr = ToTensor()(img_lr).unsqueeze(0).cuda()     # Use tensor format
        
        # Generate the processed images
        print("lr shape is ", img_lr.shape)
        gen_hr = generator(img_lr)

        # Store the image
        if store_dir != None:
            save_image(gen_hr, os.path.join(store_dir, img_name))       # save with the same name as the input img name
        
        # Empty the cache everytime you finish processing one image
        torch.cuda.empty_cache() 



def SR_inference(generator, dataset_lr_path, dataset_gen_path, scale=4, crop_for_4x=False):
    ''' Inference LR images to HR ones
    Args:
        generator (torch):              the generator class that is already loaded
        dataset_lr_path (str):  LR images input location
        dataset_gen_path (str): Where we save the generated HR ones
        scale (int):            The scale we will use for this inference
        crop_for_4x (bool):     Whether we crop the lr images to match 4x scale
    '''

    # Call the pre-built inference API
    testify(generator, dataset_lr_path, dataset_gen_path, scale=scale, crop_for_4x=crop_for_4x)

    print("------------------Finish the SR inference!------------------")



def process(generator, lr_path, gen_path):
    ''' Inference lr_path to generate HR one
    Args:
        generator (torch):                      The generator class that is already loaded
        lr_path (str):                          Low Resolution path
        gen_path (str):                         Generated resolution path
    Returns:
        sub_metric_result (dict): a sub dictionary result to return  [key: metric(str); value: metric_value (float)]
    '''

    print("We are processing path ", lr_path)

    # Prepare path
    assert(os.path.exists(lr_path))

    # SR Inference
    SR_inference(generator, lr_path, gen_path)



def has_subfolder(dir):
    ''' check if the this directory include other folders
    Args:
        dir (str): directory path
    Returns:
        result (bool): True, if has; False, if not
    '''
    for file in os.listdir(dir):
        location=os.path.join(dir, file)
        if os.path.isdir(location):
            return True
    return False


def store_json(metric_result):
    ''' Store the metric_result as a json
    Args:
        metric_result (dict): the nested dictionary we need to store
    '''
    if os.path.exists('metric_result.json'):
        if os.path.exists('metric_result_archived.json'):
            os.remove('metric_result_archived.json')
        os.rename('metric_result.json', 'metric_result_archived.json')      # Rename as the archived json file
    json_data = json.dumps(metric_result, indent=4)
    with open('metric_result.json', 'w') as f:
        # Append new result at the end
        f.write(json_data)


def test(generator, dataset_lr_paths, dataset_gen_paths):
    ''' Iterative loop Start here
    Args:
        generator (torch):                The generator class that is already loaded
        dataset_lr_paths (str):           Low Resolution path
        dataset_gen_paths (str):          Generated resolution path
    '''
    
    # Prepare Setting
    assert(len(dataset_lr_paths) == len(dataset_gen_paths))
    for path in dataset_lr_paths:
        if not os.path.exists(path):
            print("We cannot find ", path)
            os._exit(0)


    # Iterate each dataset
    for idx, lr_path in enumerate(dataset_lr_paths):
        gen_path = dataset_gen_paths[idx]
        

        # Folder in Folder case (Video dataset)
        if has_subfolder(lr_path):
            # Rebuild the Path Again
            if os.path.exists(gen_path):
                shutil.rmtree(gen_path)
            os.makedirs(gen_path)

            # If this is a folder we will handle it recursively
            for sub_folder_name in sorted(os.listdir(lr_path)):
                sub_lr_path = os.path.join(lr_path, sub_folder_name)
                sub_gen_path = os.path.join(gen_path, sub_folder_name)

                # Process
                process(generator, sub_lr_path, sub_gen_path)


        else:   # Single Folder
            
            # Process
            process(generator, lr_path, gen_path)



def load_generator(weight_path):
    ''' load the generator based on our exp_name, model_name, and weight_name
    Args:
        weith_path (str): the experiment weight path
    Returns:
        generator (torch): the generator instance of the model
    '''

    print("We want to process ", weight_path)
    assert(os.path.exists(weight_path))

    # Read the generator, The preset one is GRL
    generator = load_grl(weight_path)


    return generator



if __name__ == "__main__":
    # Fundamental setting
    weight_path = "weights/4x_VCISR_generator.pth"    # The directory you store all   
    dataset_lr_paths = [
        # "PUT YOUR INPUT DATASETS HERE",
        "../VC-RealLQ_39/", 
    ]
    dataset_gen_paths = [
        # "PUT YOUR STORE PATH HERE (corresponding to each dataset_lr_paths)", 
        "result/test_gen", 
    ] # Can support folders in folders cases



    # Inference
    generator = load_generator(weight_path)
    test(generator, dataset_lr_paths, dataset_gen_paths)



        
