from setuptools import find_packages, setup
from typing import List

hyphen_e_dot = '-e .'
def get_requirements(file:str)->List[str]:
    ## Return the list of requirements
    requirements = []
    with open(file) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace("\n", "") for req in requirements] 
    
    if hyphen_e_dot in requirements:
        requirements.remove(hyphen_e_dot)

setup(
    name='mlproject',
    version = '0.0.1',
    author = 'saksham',
    author_email = 'sakshambansal.ubs@gmail.com',
    packages = find_packages(),
    install_requires = get_requirements('requirements.txt')
)