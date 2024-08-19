from pathlib import Path

def find_extensions(dir_path): #,  excluded = ['', '.txt', '.lnk']):
    """
    Get all the file extensions in the given directory
    From https://stackoverflow.com/questions/45256250
    """
    extensions = set()
    for _, _, files in Path(dir_path).walk():   
        for f in files:
            extensions.add(Path(f).suffix)
            # ext = Path(f).suffix.lower()
            # if not ext in excluded:
            #     extensions.add(ext)
    return extensions


def year_path(project, deployment_split):
    """
    From the glider deployment name, 
    generate and return the year string to use in file paths 
    for ESD glider deployments

    For the FREEBYRD project, this will be the year of the second 
    half of the Antarctic season. For instance, hypothetical
    FREEBYRD deployments amlr01-20181231 and amlr01-20190101 are both 
    during season '2018-19', and thus would return '2019'.
    """

    year = deployment_split[1][0:4]

    if project == 'FREEBYRD':
        month = deployment_split[1][4:6]
        if int(month) <= 7: 
            year = f'{int(year)}'
        else:
            year = f'{int(year)+1}'

    return year
