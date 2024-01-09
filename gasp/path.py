import os
import gasp

config = {
    'gasp_path': os.path.join(os.path.dirname(gasp.__file__)),
    'project_path': os.path.join(os.path.dirname(gasp.__file__), '../')
}

def get_gasp_path():
    ''' Retrieved gasp module path '''
    return config['gasp_path']

def get_project_path():
    ''' Retrieves gasp root project path '''
    return config['project_path']