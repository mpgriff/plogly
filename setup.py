from setuptools import setup, find_packages
__version__='0.0.1'

import sys, os

if '-skip_click' in sys.argv:
    sys.argv.pop(sys.argv.index('-skip_click'))
else:
    #import clickable_extensions
    #clickable_extensions.run_association_opener_cmd()

    cmd_file = __file__.replace('setup.py', 'scripts\\plog_opener.cmd')
    plotter_file = cmd_file.replace('plog_opener.cmd', 'plog_plotter.py')
    secret_exe = sys.executable.replace('python.exe', 'pythonw.exe')

    if 'conda' in sys.executable:
        env = sys.executable.split(os.sep)[-2]
        activate_line = f'call conda activate {env}\n'
    else:
        activate_line = f'call {sys.executable.replace("python.exe", "activate.bat")}\n'

    with open(cmd_file, 'wt') as f:
        f.write(activate_line)
        # f.write('call echo %1\n')
        f.write(f'call {secret_exe} {plotter_file} %1 \n')


setup(
    name='plogly',
    author='Matt Griffiths',
    author_email='mpg@geo.au.dk',
    packages=find_packages('src'),
    package_dir={'':'src'},
    install_requires=['numpy', 'plotly', 'dill'],
    version=__version__,
    license='MIT?',
    description='plot logs',
    python_requires=">=3.8",
)
