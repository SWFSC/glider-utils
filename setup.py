from setuptools import setup, find_packages


setup(name='esdglider',
      version='0.1a.9000',
      description='Utility functions for processing ESD glider data',
      url='http://github.com/swfsc/glider-utils',
      author='Sam Woodman',
      author_email='sam.woodman@noaa.gov',
      license='CC0-1.0',
      packages=find_packages(),
      package_data = {'esdglider': ["esdglider/data/*"]},
      include_package_data=True,    
      python_requires='>=3.10',
      install_requires=[
            # 'google-crc32c==1.1',
            # 'google-cloud-secret-manager==2.12',
            'pandas', 
            'numpy', 
            'scipy', 
            'SQLAlchemy'
      ],
      zip_safe=False)
