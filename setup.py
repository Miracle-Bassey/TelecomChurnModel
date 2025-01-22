from setuptools import find_packages,setup
#this will fnd all packages used in this project

from typing import List

var = "-e ." # this is evident in requirement.txt, used to trigger setup.py.
# we do not want this as a part of the packages list.


def get_requirements(file_path:str) -> List[str]:
    """
    This function returns all packages installed in the requirement.txt file in a list
    :param file_path:the requirement.txt file path
    :return: a list separated by ","
    """
    requirements = []
    # open the file
    with open(file_path, encoding="utf-8") as fileobject:
        requirements = fileobject.readlines()
        # remove all \n after the pckage names.
        requirements = [req.replace("\n","") for req in requirements]
        # strip white spaces and empty lines
        requirements = [req.strip() for req in requirements if req.strip()]

        if var in requirements:
            requirements.remove(var)

    return requirements


setup(
    name='TelecomChurnModel',
    version='0.0.1',
    author='Miracle',
    author_email='<basseymiracleosa@gmail.com>',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')




)