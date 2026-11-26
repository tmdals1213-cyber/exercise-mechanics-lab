"""Inspect a separately supplied OpenSim model; do not synthesize muscle forces."""
import argparse
import json
import sys
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=Path)
args = parser.parse_args()
try:
    import opensim as osim
except ImportError:
    print('OpenSim is not installed in this Python environment.')
    print('Optional setup: python -m venv .venv')
    print(r'Then: .venv\Scripts\python -m pip install -r requirements-opensim.txt')
    sys.exit(2)

print('OpenSim:', osim.GetVersionAndDate())
if args.model:
    model = osim.Model(str(args.model.resolve()))
    state = model.initSystem()
    muscles = model.getMuscles()
    coordinates = model.getCoordinateSet()
    print(json.dumps({'model': str(args.model), 'body_mass_kg': model.getTotalMass(state),
                      'muscles': [muscles.get(i).getName() for i in range(muscles.getSize())],
                      'coordinates': [coordinates.get(i).getName() for i in range(coordinates.getSize())]}, indent=2))
    print('Inspection only. No scaling, feasibility assessment or force estimation has been run.')
