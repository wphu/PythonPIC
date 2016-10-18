import numpy as np
import time

def l2_norm(reference, test):
    return np.sum((reference - test)**2) / np.sum(reference**2)


def l2_test(reference, test, rtol=1e-3):
    norm = l2_norm(reference, test)
    print("L2 norm: ", norm)
    return norm < rtol

def date_version_string():
	run_time = time.ctime()
	git_version = "POTATO" # TODO: read current git commit ID, branch
	dv_string = "{}\n{}".format(run_time, git_version)
	return dv_string

if __name__=="__main__":
	print(date_version_string())