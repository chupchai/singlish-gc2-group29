# Setup
# -----
import subprocess
import sys

subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])


import seaborn as sns

# Basic Data Manipulation
# -----------------------
#
# Use the seaborn tips dataset to generate a best fitting linear regression line

tips = sns.load_dataset("tips")
sns.set(font="DejaVu Sans")
sns.jointplot(data=tips, x="total_bill", y="tip", kind='reg').fig.suptitle("Tips Regression", y=1.01)

# Examine the difference between smokers and non smokers
sns.lmplot(data=tips, x="total_bill", y="tip", col="smoker").fig.suptitle("Tips Regression - categorized by smoker", y=1.05)

# Explore the dataframe
tips.head()

# Using IPython's Rich Display System
# -----------------------------------
#
# IPython has a [rich display system](bit.ly/HHPOac) for
# interactive widgets.

from IPython.display import IFrame, display

# Define a google maps function.
def gmaps(query):
  url = "https://maps.google.com/maps?q={0}&output=embed".format(query)
  display(IFrame(url, '700px', '450px'))

gmaps("Golden Gate Bridge")

# Worker Engines
# -----------------
#
# You can launch worker engines to distribute your work across a cluster.
# Uncomment the following to launch two workers with 2 cpu cores and 0.5GB
# memory each.

# import cml.workers_v1 as workers
# launched_workers = workers.launch_workers(n=2, cpu=0.2, memory=0.5, code="print('Hello from a CML Worker')")
