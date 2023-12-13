from setuptools import setup

import vd


setup(
    name='vd',
    version=vd.__version__,
    description='My Python implementation of vidir',
    author='Chang-Yen Chih',
    author_email='michael66230@gmail.com',
    url=f'https://github.com/pi314/vd',
    py_modules=['vd'],
    keywords=['vidir'],
    entry_points = {
        'console_scripts': ['vd=vd:main', 'vd2=vd2:main'],
    },
    python_requires='>=3.6',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Environment :: Console',
        'Intended Audience :: End Users/Desktop',
        'License :: OSI Approved :: BSD License',
        'Natural Language :: English',
        'Operating System :: MacOS :: MacOS X',
        'Programming Language :: Python :: 3',
        'Topic :: Utilities',
    ],
)
