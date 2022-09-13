from setuptools import setup

setup(
    name='RandomForestExplorer',
    version='0.1.0',
    description='A small package for rule discovery in Random Forest',
    url='https://github.com/NimaSarajpoor/RandomForestExplorer',
    author='Nima Sarajpoor',
    author_email='nimasarajpoor@gmail.com',
    license='BSD 3-Clause',
    packages=['RandomForestExplorer'],
    install_requires=['numpy>=1.21',
                      'mlxtend>=0.20',
                      'scikit-learn>=1.1'],

    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
        #'License :: OSI Approved :: BSD License',
        #'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 3.9',
    ],
)
