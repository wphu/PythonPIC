import numpy as np
import time
import subprocess


def l2_norm(reference, test):
    return np.sum((reference - test)**2) / np.sum(reference**2)


def l2_test(reference, test, rtol=1e-3):
    norm = l2_norm(reference, test)
    print("L2 norm: ", norm)
    return norm < rtol

def date_version_string():
	run_time = time.ctime()
	git_version = subprocess.check_output(['git', 'describe', '--always']).decode('utf-8')[:-1]
	dv_string = "{}\nPrevious git version: {}".format(run_time, git_version)
	return dv_string

if __name__=="__main__":
	print(date_version_string())