# ml-topol

The code was first written when Tensorflow was at version 1.0 and finally run with version 1.3.
Since then major changes have been carried out in the API to the current stable version 1.14

The code may need to be substantially modified in order to be used on the current version and may completely break in TF 2.0 which is now in beta phase.

Example Usage:
python ModelRun.py --export=/tmp/k500 --data=4-0.10.dat --gammaRange=0.10 --modeRange=1  --output=tripleCell/O5k  --trendFilter-off --maxSteps=500000

