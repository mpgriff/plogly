import sys, os
import ctypes
print(__file__)


def associate_direct(extension, abbreviation, script):
    """ this directly associates a python script with a given file.
        no environment calling is done, and so it is not compatible with QGIS clicking for example.
    """
    abs_path = os.path.split(__file__)[0]+'\\'

    cmd_string = f"assoc .{extension}={abbreviation}"
    os.system(cmd_string)
    cmd_string  = f"ftype {abbreviation}=\"{sys.executable}\" \""
    cmd_string += abs_path+script
    cmd_string += "\" \"%1\" %*"
    print(cmd_string)
    os.system(cmd_string)

def associate(extension, abbreviation, script='scripts\\plog_opener.cmd'):
    """ this function associates a file type with a python script in the package.
    """
    abs_path = os.path.split(__file__)[0]+'\\'
    cmd_string = f"assoc .{extension}={abbreviation}"
    os.system(cmd_string)
    cmd_string  = f"ftype {abbreviation}=\""
    
    cmd_string += abs_path+script
    cmd_string += "\" \"%1\" %*"
    print(cmd_string)
    os.system(cmd_string)


def run_association_opener_cmd():
    try:
        is_admin = os.getuid() == 0
    except AttributeError:
        is_admin = ctypes.windll.shell32.IsUserAnAdmin() != 0

    if is_admin:
        print("running file association")
        associate('plog', 'borehole')
    else:
        print("no admin rights, setup will not associate special file extensions with viewer / reader scripts")

if __name__=='__main__':
    try:
        is_admin = os.getuid() == 0
    except AttributeError:
        is_admin = ctypes.windll.shell32.IsUserAnAdmin() != 0

    if is_admin:
        print("running file association")
        associate_direct('plog', 'borehole', 'plog_plotter')
    else:
        print("no admin rights, setup will not associate special file extensions with viewer / reader scripts")
