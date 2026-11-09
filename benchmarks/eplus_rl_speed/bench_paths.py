"""Where EnergyPlus and SBSim are on this machine.

Every script in this folder reads the two folders from here. To run on another
machine, change the defaults below, or set the environment variables
ENERGYPLUS_DIR and SBSIM_DIR.
"""
import os
from pathlib import Path

# EnergyPlus install folder (holds the energyplus program, pyenergyplus and WeatherData)
ENERGYPLUS_DIR = Path(os.environ.get("ENERGYPLUS_DIR", r"C:\EnergyPlusV26-2-0"))

# SBSim source folder (holds smart_control). The default is the repository this file is in.
SBSIM_DIR = Path(os.environ.get("SBSIM_DIR", Path(__file__).resolve().parents[2]))

# Chicago O'Hare weather file, shipped with EnergyPlus
WEATHER = ENERGYPLUS_DIR / "WeatherData" / "USA_IL_Chicago-OHare.Intl.AP.725300_TMY3.epw"
