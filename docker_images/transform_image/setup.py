import setuptools

setuptools.setup(
    name='tft',
    version='0.1',
    install_requires=['tensorflow==2.5.0', 'tensorflow-transform==1.1.0'],
    packages=setuptools.find_packages(),
)
