from util import mkdir_results
from asa24.asa24_experiment_1 import asa24_experiment_1_run
from asa24.asa24_experiment_2 import asa24_experiment_2_run
from nhanes.nhanes_experiment_1 import nhanes_experiment_1_run
from nhanes.nhanes_experiment_2 import nhanes_experiment_2_run
from nhanes.nhanes_experiment_3 import nhanes_experiment_3_run
from nhanes.nhanes_experiment_4 import nhanes_experiment_4_run

if __name__ == "__main__":
    mkdir_results()

    asa24_experiment_1_run()
    asa24_experiment_2_run()
    nhanes_experiment_1_run()
    nhanes_experiment_2_run()
    nhanes_experiment_3_run()
    nhanes_experiment_4_run()