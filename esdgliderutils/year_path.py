def year_path(project, deployment_split):
    """
    From the glider deployment name, 
    generate and return the year string to use in file paths 
    for ESD glider deployments

    For the FREEBYRD project, this will be the year of the second 
    half of the Antarctic season. For instance, 
    FREEBYRD deployments amlr01-20181201 and amlr01-20190101 are both 
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
