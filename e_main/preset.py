import random
import socket
import string
def get_hostname():
    hostname = socket.gethostname()
    return hostname
class Preset:
    HOSTNAME = get_hostname()

    # HOSTNAME use print(Preset.HOSTNAME) to check out
    if HOSTNAME == 'PC_Harry_1':
        # the absolute path to your 'research_project' directory
        root = r'C:\Users\PC\Dropbox\_Research_Course\project\cis_research_project'

    # please filling here
    elif HOSTNAME == '':
        pass
        root = ''




if __name__ == '__main__':
    pass
