from setuptools import setup, find_packages

setup(name='esdglider',
      version="0.1.0-dev1",
      description='Utility functions for processing ESD glider data',
      url='http://github.com/swfsc/glider-utils',
      author='Sam Woodman',
      author_email='sam.woodman@noaa.gov',
      license='Apache 2.0',
      # packages=find_packages(exclude=('tests', 'docs')), 
      packages=find_packages(), 
      package_data = {'esdglider': ["esdglider/data/*"]},
      include_package_data=True,    
      python_requires='>=3.10',
      install_requires=[
            'google-crc32c>=1.1',
            'google-cloud-secret-manager>=2.12',
            'numpy', 
            'pandas',
            'xarray', 
            "pyyaml", 
            "pyglider", 
            "netCDF4", 
            "SQLAlchemy"
      ],
      zip_safe=False)

