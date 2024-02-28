from distutils.core import setup
from setuptools import find_packages

with open("README.md", "r", encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='scenario-wise-rec',
    version='0.0.1',
    description='An open-sourced benchmark for Multi-Scenario Recommendation.',
    long_description=long_description,
    long_description_content_type="text/markdown",
    author='Xiaopeng Li',
    author_email='xiaopli2-c@my.cityu.edu.hk',
    url='https://github.com/Xiaopengli1/Scenario-Wise-Rec',
    install_requires=['numpy>=1.23.5', 'torch>=1.13.1', 'pandas>=1.5.3', 'tqdm>=4.64.1', 'scikit_learn>=1.2.1', 'joblib>=1.1.1'],
    packages=find_packages(),
    platforms=["all"],
    classifiers=[
        'Intended Audience :: Developers',
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'License :: OSI Approved :: MIT License',
    ],
    keywords=['ctr', 'click through rate', 'deep learning', 'pytorch', 'recsys', 'recommendation'],
)