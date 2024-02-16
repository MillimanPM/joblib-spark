from collections import defaultdict
import sys
import os
import psutil
import pathlib
import pickle
from sklearn.metrics import mean_squared_error
import hashlib
import pyAesCrypt

def nested_ddict():
    """
    Create an empty nested default dict.
    
    Parameters
    __________
    
    """
    return defaultdict(nested_ddict)

def ddict_to_rdict(ddict):
    """
    Convert nested default dict to regular nested dict.
    
    Parameters
    __________
    
    ddict: default dict
        Original nested default dict.
    """
    return {k: ddict_to_rdict(v) for k, v in ddict.items()} if isinstance(ddict, dict) else ddict

def print_local_variable_size(local_var_dict):
    """
    Print local variable size. Normally used in notebook.
    
    Parameters
    __________
    
    local_var_dict: dict
        locals(). Or other local variable dict supported by user.
    """
    def sizeof_fmt(num, suffix='B'):
        ''' by Fred Cirera,  https://stackoverflow.com/a/1094933/1870254, modified'''
        for unit in ['','Ki','Mi','Gi','Ti','Pi','Ei','Zi']:
            if abs(num) < 1024.0:
                return "%3.1f %s%s" % (num, unit, suffix)
            num /= 1024.0
        return "%.1f %s%s" % (num, 'Yi', suffix)

    for name, size in sorted(((name, sys.getsizeof(value)) for name, value in local_var_dict.items()),
                             key= lambda x: -x[1])[:10]:
        print("{:>30}: {:>8}".format(name, sizeof_fmt(size)))
        
def print_ram_cpu_usage():
    """
    Print CPU and RAM usage.
    
    Parameters
    __________
    
    """
    width=50

    total_memory, free_memory=psutil.virtual_memory().total,psutil.virtual_memory().available
    ram_msg_0=f'RAM memory % used: {(1-free_memory/total_memory)*100:.2f}'
    ram_msg_1=f'RAM Used (GB): {(total_memory-free_memory)/(1024**3):.2f}'

    cpu_msg_0=f'The CPU usage % is: {psutil.cpu_percent(4):.2f}'
    
    print(f'{"Process Usage Report".center(width+2,"*")}')
    print(f'*{cpu_msg_0.center(width," ")}*')
    print(f'*{ram_msg_0.center(width," ")}*')
    print(f'*{ram_msg_1.center(width," ")}*')
    print(f'{"*"*(width+2)}')
    
def dict_value_str_replace(d,oldvalue,newvalue):
    """
    Traverse a dictionary and replace substring when value is a string.
    
    Parameters
    __________
    
    d: dict
        Dictionary for traversal.
    oldvalue: str
        Substring to be replaced.
    newvalue: str
        Replacement string.
    """
    for k, v in d.items():
        d[k]=v.replace(oldvalue,newvalue) if isinstance(v, str) else dict_value_str_replace(v,oldvalue,newvalue) if isinstance(v, dict) else v
    return d

def print_dir_tree(dir_path: pathlib.Path, prefix: str=''):
    """
    A recursive generator, given a directory pathlib.Path object will yield a visual tree structure line by line with each line prefixed by the same characters.
    
    Parameters
    __________
    
    dir_path: pathlib.Path
        pathlib.Path of the root directory.
    prefix: str
        Prefix.
    """
    def tree(dir_path, prefix):
        # prefix components:
        space =  '    '
        branch = '│   '
        # pointers:
        tee =    '├── '
        last =   '└── '
        contents = list(dir_path.iterdir())
        # contents each get pointers that are ├── with a final └── :
        pointers = [tee] * (len(contents) - 1) + [last]
        for pointer, path in zip(pointers, contents):
            yield prefix + pointer + path.name
            if path.is_dir(): # extend the prefix and recurse:
                extension = branch if pointer == tee else space 
                # i.e. space because last, └── , above so no more |
                yield from tree(path, prefix+extension)
    
    for line in tree(dir_path, prefix):
        print(line)

def delete_s3_folder_file(s3_client,bucket,folder_file_path):
    """
    A recursive generator, given a directory pathlib.Path object will yield a visual tree structure line by line with each line prefixed by the same characters.
    
    Parameters
    __________
    
    s3_client: boto3.client
        boto3.client.
    bucket: str
        S3 bucket name.
    folder_file_path: str
        Folder path in s3 bucket.
    """
    objects = s3_client.list_objects(Bucket=bucket, Prefix=folder_file_path)
    keys = {'Objects' : []}
    keys['Objects'] = [{'Key' : k} for k in [obj['Key'] for obj in objects.get('Contents', [])]]
    s3_client.delete_objects(Bucket=bucket, Delete=keys)

def custom_mean_squared_error(y_true,y_pred,*,sample_weight=None,multioutput='uniform_average',squared=True,full_sample_weight=None):
    """
    Custom mean squared error for hyperparameter search method. The goal is to use sample weight in metrics calculation during hyperparameter search.

    Parameters
    _________

    y_true: array-like of shape (n_samples,) or (n_samples, n_outputs)
        Ground truth (correct) target values.
    y_pred: array-like of shape (n_samples,) or (n_samples, n_outputs)
        Estimated target values.
    sample_weight: array-like of shape (n_samples,)
        Sample weights.
    multioutput: {‘raw_values’, ‘uniform_average’} or array-like of shape (n_outputs,)
        Defines aggregating of multiple output values. Array-like value defines weights used to average errors.
        raw_values: Returns a full set of errors in case of multioutput input.
        uniform_average: Errors of all outputs are averaged with uniform weight.
    squared: bool
        If True returns MSE value, if False returns RMSE value.
    full_sample_weight: Series
        Pandas series of the full training set.
    """
    if full_sample_weight is not None:
        sample_weight=full_sample_weight.loc[y_true.index]
    return mean_squared_error(y_true=y_true,y_pred=y_pred,sample_weight=sample_weight,multioutput=multioutput,squared=squared)

def encrypt_config(config,encrypted_config_file_path,key_file_path):
    """
    Save config dictionary as an encrypted file and a key file.

    Parameters
    _________

    config: dict
        Config dictionary.
    encrypted_config_file_path: str
        Encrypted confile file path.
    key_file_path: str
        Key file path.
    """
    password=hashlib.sha3_256(pickle.dumps(config)).hexdigest()
    with open(key_file_path, 'wb') as filehandler:
        pickle.dump(password, filehandler)
    with open(encrypted_config_file_path+'.tmp', 'wb') as filehandler:
        pickle.dump(config, filehandler)
    pyAesCrypt.encryptFile(encrypted_config_file_path+'.tmp', encrypted_config_file_path, password)
    os.remove(encrypted_config_file_path+'.tmp')
    
def decrypt_config(decrypted_config_file_path,encrypted_config_file_path,key_path):
    """
    Decrypt an encrypted file using a key file. Save the decrypted file.

    Parameters
    _________

    decrypted_config_file_path: str
        Decrypted confile file path.
    encrypted_config_file_path: str
        Encrypted confile file path.
    key_file_path: str
        Key file path.
    """
    with open(key_path, 'rb') as filehandler:
        password=pickle.load(filehandler)
    pyAesCrypt.decryptFile(encrypted_config_file_path,decrypted_config_file_path,password)
