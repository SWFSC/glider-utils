from setuptools import setup

setup(name='esdgliderutils',
      version='0.1a',
      description='Utility functions for processing ESD glider data',
      url='http://github.com/swfsc/glider-utils',
      author='Sam Woodman',
      author_email='sam.woodman@noaa.gov',
      license='CC0-1.0',
      packages=['esdgliderutils'],
      python_requires='>=3.10',
      install_requires=[
            # 'google-crc32c==1.1',
            # 'google-cloud-secret-manager==2.12',
            'pandas', 
            # 'numpy', 
            'SQLAlchemy'
      ],
      zip_safe=False)