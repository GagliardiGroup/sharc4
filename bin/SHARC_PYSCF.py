#!/usr/bin/env python3

# ******************************************
#
#    SHARC Program Suite
#
#    Copyright (c) 2026 University of Vienna
#    Copyright (c) 2026 University of Minnesota
#    Copyright (c) 2026 University of Chicago
#
#    This file is part of SHARC.
#
#    SHARC is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    SHARC is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    inside the SHARC manual.  If not, see <http://www.gnu.org/licenses/>.
#
# ******************************************

from SHARC_OLD import SHARC_OLD
class SHARC_PYSCF(SHARC_OLD):
    pass


from multiprocessing import Pool
import os
import shutil
import sys
import re
import datetime
import pprint
import numpy as np
from copy import deepcopy

from socket import gethostname
import time

from pyscf import lib, gto, scf, mcscf

_version_ = "4.0"
_versiondate_ = datetime.date(2025, 4, 1)
_change_log_ = """"""

START_TIME = datetime.datetime.now()

# Global Variables for printing.
DEBUG = False  # Raw output
PRINT = True  # Formatted output

IToMult = {
    1: "Singlet",
    2: "Doublet",
    3: "Triplet",
    4: "Quartet",
    5: "Quintet",
    6: "Sextet",
    7: "Septet",
    8: "Octet",
    "Singlet": 1,
    "Doublet": 2,
    "Triplet": 3,
    "Quartet": 4,
    "Quintet": 5,
    "Sextet": 6,
    "Septet": 7,
    "Octet": 8,
}

# conversion factors
AU_TO_ANG = 0.529177211
rcm_to_Eh = 4.556335e-6


def eformat(f, prec, exp_digits):
    """Formats a float f into scientific notation with prec number of decimals and exp_digits number of exponent digits.

    String looks like:
    [ -][0-9]\\.[0-9]*E[+-][0-9]*

    Arguments:
    1 float: Number to format
    2 integer: Number of decimals
    3 integer: Number of exponent digits

    Returns:
    1 string: formatted number"""

    s = "% .*e" % (prec, f)
    mantissa, exp = s.split("e")
    return "%sE%+0*d" % (mantissa, exp_digits + 1, int(exp))


def measure_time():
    """Calculates the time difference between global variable starttime and the time of the call of measuretime.

    Prints the Runtime, if PRINT or DEBUG are enabled.

    Arguments:
    none

    Returns:
    1 float: runtime in seconds"""

    endtime = datetime.datetime.now()
    runtime = endtime - START_TIME
    if PRINT or DEBUG:
        hours = runtime.seconds // 3600
        minutes = runtime.seconds // 60 - hours * 60
        seconds = runtime.seconds % 60
        print(
            "==> Runtime:\n%i Days\t%i Hours\t%i Minutes\t%i Seconds\n\n"
            % (runtime.days, hours, minutes, seconds)
        )
    total_seconds = (
        runtime.days * 24 * 3600 + runtime.seconds + runtime.microseconds // 1.0e6
    )
    return total_seconds


def print_header():
    """Prints the formatted header of the log file. Prints version number and version date"""
    print(START_TIME, gethostname(), os.getcwd())
    if not PRINT:
        return
    string = "\n"
    string += "  " + "=" * 80 + "\n"
    string += "||" + " " * 80 + "||\n"
    string += "||" + " " * 27 + "SHARC - PySCF - Interface" + " " * 28 + "||\n"
    string += "||" + " " * 80 + "||\n"
    string += "||" + " " * 25 + "Authors: Matthew R. Hennefarth" + " " * 25 + "||\n"    # citation: https://pubs.acs.org/doi/10.1021/acs.jctc.4c01061
    string += "||" + " " * 80 + "||\n"
    string += (
        "||"
        + " " * (36 - (len(_version_) + 1) // 2)
        + f"Version: {_version_}"
        + " " * (35 - (len(_version_)) // 2)
        + "||\n"
    )
    lens = len(_versiondate_.strftime("%d.%m.%y"))
    string += (
        "||"
        + " " * (37 - lens // 2)
        + f"Date: {_versiondate_.strftime('%d.%m.%y')}"
        + " " * (37 - (lens + 1) // 2)
        + "||\n"
    )
    string += "||" + " " * 80 + "||\n"
    string += "  " + "=" * 80 + "\n\n"
    print(string)
    if DEBUG:
        print(_change_log_)


def print_qmin(qmin):
    if DEBUG:
        pprint.pprint(qmin)

    if not PRINT:
        return

    print(f"==> QMin Job description for:\n{qmin['comment']}")
    TASK_TO_STRING = {
        "h": "H",
        "soc": "SOC",
        "dm": "DM",
        "grad": "Grad",
        "nacdr": "Nac(ddr)",
        "nacdt": "Nac(ddt)",
        "overlap": "Overlaps",
        "angular": "Angular",
        "ion": "Dyson norms",
        "dmdr": "DM-Grad",
        "socdr": "SOC-Grad",
        "phases": "Phases",
    }
    output = "Tasks: "
    for key, string in TASK_TO_STRING.items():
        if key in qmin:
            output += f"\t{string}"

    print(output)

    output = "States: "
    for i in itmult(qmin["states"]):
        output += f"\t{qmin['states'][i-1]} {IToMult[i]}"

    print(output)

    output = "Method: \t"
    tmp = "|".join([str(r) for r in qmin["template"]["roots"]])
    if qmin["template"]["method"].upper() == "L-PDFT":
        output += f"L({tmp})-PDFT"

    elif qmin["template"]["method"].upper() == "MC-PDFT":
        output += f"SA({tmp})-PDFT"

    else:
        output += f"SA({tmp})-{qmin['template']['method'].upper()}"

    output += f"({qmin['template']['nelecas']}, {qmin['template']['ncas']})/{qmin['template']['basis']}"
    print(output)

    oddmults = False
    for i in qmin["statemap"].values():
        if (qmin["template"]["nelecas"] + i[0]) % 2 == 0:
            oddmults = True

    if oddmults:
        output = "\t\t" + ["Even ", "Odd "][qmin["template"]["nelecas"]] % 2 == 0
        output += f"numbers of electrons are treated wiht CAS({qmin['template']['nelecas']}, {qmin['template']['ncas']})"
        print(output)

    output = "Found Geo"
    if "veloc" in qmin:
        output += " and Veloc! "

    else:
        output += "! "

    output += f"NAtom is {qmin['natom']}.\n"
    print(output)

    output = "\nGeometry in Bohrs:\n"
    for atom in qmin["geo"]:
        element = atom[0]
        coords = atom[1:]
        output += f"{element} "
        for x in coords:
            output += f"{x:7.4f} "

        output += "\n"

    print(output)

    if "veloc" in qmin:
        output = ""
        for index, veloc in enumerate(qmin["veloc"]):
            element = qmin["geo"][index][0]
            output += f"{element} "
            for v in veloc:
                output += f"{v:7.4f} "

            output += "\n"

        print(output)

    if "grad" in qmin:
        output = "Gradients:   "
        for i in range(1, qmin["nmstates"] + 1):
            if i in qmin["grad"]:
                output += "X"

            else:
                output += "."

        output += "\n"
        print(output)

    if "nacdr" in qmin:
        output = "Nonadiabatic couplings:\n"
        for i in range(1, qmin["nmstates"] + 1):
            for j in range(1, qmin["nmstates"] + 1):
                if [i, j] in qmin["nacdr"] or [j, i] in qmin["nacdr"]:
                    output += "X"

                else:
                    output += "."
            output += "\n"
        print(output)

    if "overlap" in qmin:
        output = "Overlaps:\n"
        for i in range(1, qmin["nmstates"] + 1):
            for j in range(1, qmin["nmstates"] + 1):
                if (i, j) in qmin["overlap"] or (j, i) in qmin["overlap"]:
                    output += "X"

                else:
                    output += "."
            output += "\n"
        print(output)

    print("\n")
    sys.stdout.flush()


def itmult(states):
    for i in range(len(states)):
        if states[i] < 1:
            continue
        yield i + 1
    return


def itnmstates(states):
    for i in range(len(states)):
        if states[i] < 1:
            continue
        for k in range(i + 1):
            for j in range(states[i]):
                yield i + 1, j + 1, k - i / 2.0
    return


def get_pairs(lines, i):
    nacpairs = []
    while True:
        i += 1
        try:
            line = lines[i].lower()
        except IndexError:
            print('"keyword select" has to be completed with an "end" on another line!')
            sys.exit(1)
        if "end" in line:
            break

        fields = line.split()
        try:
            nacpairs.append([int(fields[0]), int(fields[1])])

        except ValueError:
            print(
                '"nacdr select" is followed by pairs of state indices, each pair on a new line!'
            )
            sys.exit(1)

    return nacpairs, i


def removequotes(string):
    if string.startswith("'") and string.endswith("'"):
        return string[1:-1]
    elif string.startswith('"') and string.endswith('"'):
        return string[1:-1]
    else:
        return string


def getsh2caskey(sh2cas, key):
    for line in sh2cas:
        line = re.sub("#.*$", "", line)
        line = line.split(None, 1)
        if line == []:
            continue

        if key.lower() in line[0].lower():
            return line

    return ["", ""]


def get_sh2cas_environ(sh2cas, key, environ=True, crucial=True):
    line = getsh2caskey(sh2cas, key)
    if line[0]:
        line = line[1]
        line = removequotes(line).strip()
    else:
        if environ:
            line = os.getenv(key.upper())
            if not line:
                if crucial:
                    print(
                        f"Either set ${key.upper()} or give path to {key.upper()} in PYSCF.resources"
                    )
                    sys.exit(1)

                else:
                    return ""

        else:
            if crucial:
                print(f"Give path to {key.upper()} in PYSCF.resources")
                sys.exit(1)

            else:
                return ""

    line = os.path.expandvars(line)
    line = os.path.expanduser(line)
    if ";" in line:
        print(
            f"${key.upper()} contains a semicolon. Do you probably want to execute another command after {key.upper()}? I can't do that for you..."
        )
        sys.exit(1)
    return line


def check_directory(dir):
    """Checks where dir is a file or directory. If a file, quits with exit code 1. If a directory, it passes. If does not exist, then we try and create the directory"""

    if os.path.exists(dir):
        if not os.path.isdir(dir):
            print(f"{dir} exists but is not a directory! Quiting...")
            sys.exit(1)

    else:
        os.makedirs(dir)

    return True


def get_version():
    from pyscf import __version__ as pyscf_version

    min_version = (2, 6, 0)
    line = pyscf_version.split(".")
    line = [int(i) for i in line]
    for min, actual in zip(min_version, line):
        if actual < min:
            print(f"PySCF version {pyscf_version} not supported!")
            sys.exit(1)

        if actual > min:
            break

    if DEBUG:
        print(f"PySCF version {pyscf_version}")

    return pyscf_version


def readqmin(filename):
    with open(filename, "r") as f:
        lines = f.readlines()

    qmin = {}
    try:
        natom = int(lines[0])

    except ValueError as e:
        print("First line must contain the number of atoms!")
        raise e

    qmin["natom"] = natom
    if len(lines) < natom + 4:
        print(
            """Input file must contain at least:
natom
comment
geometry
keyword 'states'
at least one task"""
        )
        sys.exit(1)

    qmin["comment"] = lines[1]

    # Get geometry and possibly velocity (for backup-analytical nonadiabatic couplings)
    qmin["geo"] = []
    qmin["veloc"] = []
    has_veloc = True
    for line in lines[2 : natom + 2]:
        line = line.split()
        element = line[0]
        coords = [float(x) for x in line[1:]]
        qmin["geo"].append([element] + coords[:3])
        if len(coords) >= 6:
            qmin["veloc"].append(coords[3:6])

        else:
            has_veloc = False

    if not has_veloc:
        del qmin["veloc"]

    # TODO: I hate this and this should be fixed up....ugh!
    i = natom + 1
    while i + 1 < len(lines):
        i += 1
        line = lines[i]
        line = re.sub("#.*$", "", line)
        if len(line.split()) == 0:
            continue

        key = line.lower().split()[0]
        if "savedir" in key:
            args = line.split()[1:]

        else:
            args = line.lower().split()[1:]

        if key in qmin:
            print(f"Repeated keyword {key} in line {i+1} in input file!")
            continue

        if len(args) >= 1 and "select" in args[0]:
            pairs, i = get_pairs(lines, i)
            qmin[key] = pairs
        else:
            qmin[key] = args

    if "unit" in qmin:
        if qmin["unit"][0] == "angstrom":
            factor = 1.0 / AU_TO_ANG

        elif qmin["unit"][0] == "bohr":
            factor = 1.0

        else:
            print(f"Don't know input unit {qmin['unit'][0]}")
            sys.exit(1)

    else:
        factor = 1.0 / AU_TO_ANG

    for atom in qmin["geo"]:
        atom[1:] = [xyz * factor for xyz in atom[1:]]

    if "states" not in qmin:
        print("Keyword 'states' not given!")
        sys.exit(1)

    qmin["states"] = [int(state) for state in qmin["states"]]

    reduc = 0
    for i in reversed(qmin["states"]):
        if i == 0:
            reduc += 1

        else:
            break

    if reduc > 0:
        qmin["states"] = qmin["states"][:-reduc]

    nstates = 0
    nmstates = 0
    for index, state in enumerate(qmin["states"]):
        nstates += state
        nmstates += state * (index + 1)

    qmin["nstates"] = nstates
    qmin["nmstates"] = nmstates

    unsupported_multiplicities = [
        mult
        for mult, nroots in enumerate(qmin["states"], 1)
        if nroots > 0 and mult not in (1, 3)
    ]
    if unsupported_multiplicities:
        raise NotImplementedError(
            "The SHARC-PySCF interface supports only singlets, triplets, or mixed "
            "singlet-triplet calculations currently; requested unsupported "
            f"multiplicities: {unsupported_multiplicities}."
        )

    possible_tasks = [
        "h",
        "soc",
        "dm",
        "grad",
        "overlap",
        "dmdr",
        "socdr",
        "ion",
        "phases",
    ]
    if not any([i in qmin for i in possible_tasks]):
        print(f"No tasks found! Tasks are {possible_tasks}")
        sys.exit(1)

    if "samestep" in qmin and "init" in qmin:
        print("'init' and 'samestep' cannot both be present in inputfile")
        sys.exit(1)

    if "phases" in qmin:
        qmin["overlap"] = []

    if "overlap" in qmin and "init" in qmin:
        print("'overlap' and 'phases' cannot both be calculated in the first timestep")
        sys.exit(1)

    if "init" not in qmin and "samestep" not in qmin:
        qmin["newstep"] = []

    if not any([i in qmin for i in ["h", "soc", "dm", "grad"]]) and "overlap" in qmin:
        qmin["h"] = []

    if len(qmin["states"]) > 8:
        print("Higher multiplicities than octets are not supported!")
        sys.exit(1)

    not_implemented_tasks = ["overlap", "nacdt", "dmdr", "ion", "theodore"]
    for task in not_implemented_tasks:
        if task in qmin:
            print(f"Within the SHARC-PySCF interface, '{task}' is not supported")
            sys.exit(1)

    if "h" in qmin and "soc" in qmin:
        del qmin["h"]

    if "molden" in qmin and "samestep" in qmin:
        print("HINT: not producing Molden files in 'samestep' mode!")
        del qmin["molden"]

    if "grad" in qmin:
        if len(qmin["grad"]) == 0 or qmin["grad"][0] == "all":
            qmin["grad"] = [i + 1 for i in range(nmstates)]

        else:
            for i in range(len(qmin["grad"])):
                try:
                    qmin["grad"][i] = int(qmin["grad"][i])

                except ValueError:
                    print(
                        "Arguments to keyword 'grad' must be 'all' or a list of integers"
                    )
                    sys.exit(1)

                if qmin["grad"][i] > nmstates:
                    print(
                        "State for requested gradient does not correspond to any state in QM input file state list!"
                    )
                    sys.exit(1)

    if "overlap" in qmin:
        if len(qmin["overlap"]) >= 1:
            overlap_pairs = qmin["overlap"]
            for pair in overlap_pairs:
                if pair[0] > nmstates or pair[1] > nmstates:
                    print(
                        "State for requested overlap does not correspond to any state in QM input file state list!"
                    )
                    sys.exit(1)

        else:
            qmin["overlap"] = [
                [j + i, i + 1] for i in range(nmstates) for j in range(i + 1)
            ]

    if "nacdr" in qmin:
        if len(qmin["nacdr"]) >= 1:
            nac_pairs = qmin["nacdr"]
            for pair in nac_pairs:
                if pair[0] > nmstates or pair[1] > nmstates:
                    print(
                        "State for requested nacdr does not correspond to any state in QM input file state list!"
                    )

        else:
            qmin["nacdr"] = [[j + 1, i + 1] for i in range(nmstates) for j in range(i)]

    # obtain the statemap
    statemap = {}
    i = 1
    for imult, istate, ims in itnmstates(qmin["states"]):
        statemap[i] = [imult, istate, ims]
        i += 1

    qmin["statemap"] = statemap

    gradmap = set()
    if "grad" in qmin:
        for state in qmin["grad"]:
            gradmap.add(tuple(statemap[state][0:2]))

    gradmap = sorted(gradmap)
    qmin["gradmap"] = gradmap

    nacmap = set()
    if "nacdr" in qmin:
        for pair in qmin["nacdr"]:
            s1 = statemap[pair[0]][:-1]
            s2 = statemap[pair[1]][:-1]
            if s1[0] != s2[0] or s1 == s2:
                continue
            nacmap.add(tuple(s1 + s2))

    nacmap = list(nacmap)
    nacmap.sort()
    qmin["nacmap"] = nacmap

    # TODO from 2222 of SHARC_MOLCAS, the MOLCAS.resources file loading stuff...

    pyscf_resource_filename = "PYSCF.resources"
    with open(pyscf_resource_filename, "r") as f:
        sh2cas = f.readlines()

    qmin["pwd"] = os.getcwd()

    line = get_sh2cas_environ(sh2cas, "scratchdir", environ=False, crucial=False)
    if line is None:
        line = os.path.join(qmin["pwd"], "SCRATCHDIR")

    line = line.replace("$$", str(os.getpid()))
    line = os.path.expandvars(line)
    line = os.path.expanduser(line)
    line = os.path.abspath(line)
    qmin["scratchdir"] = line

    # Set up savedir
    if "savedir" in qmin:
        line = qmin["savedir"][0]

    else:
        line = get_sh2cas_environ(sh2cas, "savedir", environ=False, crucial=False)
        if line is None or line == "":
            line = os.path.join(qmin["pwd"], "SAVEDIR")

    line = os.path.expandvars(line)
    line = os.path.expanduser(line)
    line = os.path.abspath(line)

    if "init" in qmin:
        check_directory(line)

    qmin["savedir"] = line

    line = getsh2caskey(sh2cas, "debug")
    if line[0]:
        if len(line) <= 1 or "true" in line[1].lower():
            global DEBUG
            DEBUG = True

    line = getsh2caskey(sh2cas, "no_print")
    if line[0]:
        if len(line) <= 1 or "true" in line[1].lower():
            global PRINT
            PRINT = False

    qmin["memory"] = 4000
    line = getsh2caskey(sh2cas, "memory")
    if line[0]:
        try:
            qmin["memory"] = int(line[1])

        except ValueError:
            print("PYSCF memory does not evaluate to integer value!")
            sys.exit(1)

    else:
        print(
            "WARNING: Please set memory for PySCF in PYSCF.resources (in MB)! Using 4000 MB default value!"
        )

    os.environ["PYSCF_MAX_MEMORY"] = str(qmin["memory"])

    qmin["ncpu"] = 1
    line = getsh2caskey(sh2cas, "ncpu")
    if line[0]:
        try:
            qmin["ncpu"] = int(line[1])

        except ValueError:
            print("Number of CPUs does not evaluate to integer value!")
            sys.exit(1)

    qmin["delay"] = 0.0
    line = getsh2caskey(sh2cas, "delay")
    if line[0]:
        try:
            qmin["delay"] = float(line[1])

        except ValueError:
            print("Submit delay does not evaluate to numerical value!")
            sys.exit(1)

    line = getsh2caskey(sh2cas, "always_orb_init")
    if line[0]:
        qmin["always_orb_init"] = []

    line = getsh2caskey(sh2cas, "always_guess")
    if line[0]:
        qmin["always_guess"] = []

    if "always_orb_init" in qmin and "always_guess" in qmin:
        print("Keywords 'always_orb_init' and 'always_guess' cannot be used together!")
        sys.exit(1)

    # open template
    with open("PYSCF.template", "r") as f:
        template = f.readlines()

    template_dict = {}
    INTEGERS_KEYS = ["ncas", "nelecas", "roots", "grids-level", "verbose", "max-cycle-macro", "max-cycle-micro", "ah-max-cycle", "ah-start-cycle", "grad-max-cycle", "charge"]
    STRING_KEYS = ["basis", "method", "pdft-functional", "soc-hamiltonian", "df-auxbasis", "dipole-origin"]
    FLOAT_KEYS = ["conv-tol", "conv-tol-grad", "max-stepsize", "ah-start-tol", "ah-level-shift", "ah-conv-tol", "ah-lindep", "fix-spin-shift"]
    BOOL_KEYS = ["density-fit"]

    template_dict["roots"] = [0 for _ in range(8)]
    template_dict["charge"] = 0

    template_dict["method"] = "casscf"
    template_dict["verbose"] = 3


    template_dict["fix-spin-shift"] = 0.2

    template_dict["pdft-functional"] = "tpbe"
    template_dict["grids-level"] = 4
    template_dict["soc-hamiltonian"] = "DKH"
    template_dict["density-fit"] = False
    template_dict["df-auxbasis"] = None
    template_dict["dipole-origin"] = "coord-center"

    # CASSCF solver defaults
    template_dict["conv-tol"] = 1e-7
    template_dict["conv-tol-grad"] = 1e-4

    template_dict["max-stepsize"] = 0.02
    template_dict["max-cycle-macro"] = 50
    template_dict["max-cycle-micro"] = 4

    template_dict["ah-level-shift"] = 1e-8
    template_dict["ah-conv-tol"] = 1e-12
    template_dict["ah-max-cycle"] = 30
    template_dict["ah-lindep"] = 1e-14
    template_dict["ah-start-tol"] = 2.5
    template_dict["ah-start-cycle"] = 3


    # gradient solver defaults
    template_dict["grad-max-cycle"] = 50

    for line in template:
        orig = re.sub("#.*$", "", line).split(None, 1)
        line = re.sub("#.*$", "", line).lower().split()

        if len(line) == 0:
            continue

        key = line[0]
        line = line[1:]

        if key.startswith("spin"):
            template_dict["roots"][int(line[0]) - 1] = int(line[2])

        elif "roots" in key:
            for i, n in enumerate(line):
                template_dict["roots"][i] = int(n)

        elif key in INTEGERS_KEYS:
            template_dict[key] = int(line[0])

        elif key in STRING_KEYS:
            template_dict[key] = line[0]

        elif key in FLOAT_KEYS:
            template_dict[key] = float(line[0])

        elif key in BOOL_KEYS:
            template_dict[key] = True

    # Roots must be larger or equal to states
    for i, n in enumerate(template_dict["roots"]):
        if i == len(qmin["states"]):
            break

        if not n >= qmin["states"][i]:
            print(
                f"Too few states in state-averaging in multiplicity {i+1}! {qmin['states'][i]} requested, but only {n} given."
            )
            sys.exit(1)

    # condense roots list
    for i in range(len(template_dict["roots"]) - 1, 0, -1):
        if template_dict["roots"][i] == 0:
            template_dict["roots"].pop(i)

        else:
            break

    NECESSARY_KEYS = ["basis", "nelecas", "ncas"]
    for key in NECESSARY_KEYS:
        if key not in template_dict:
            print(f"Key {key} missing in template file!")
            sys.exit(1)

    ALLOWED_METHODS = ["casscf", "l-pdft", "mc-pdft", "cms-pdft"]
    for index, method in enumerate(ALLOWED_METHODS):
        if template_dict["method"].lower() == method.lower():
            qmin["method"] = index
            break

    else:
        print(f"Unknown method {template_dict['method']}")
        sys.exit(1)

    # find functional if pdft (methods: 0=casscf, 1=l-pdft, 2=mc-pdft, 3=cms-pdft)
    if qmin["method"] in [1, 2, 3]:
        ALLOWED_FUNCTIONALS = ["tpbe", "ftpbe"]
        for index, func in enumerate(ALLOWED_FUNCTIONALS):
            if template_dict["pdft-functional"] == func:
                qmin["pdft-functional"] = index
                break

        else:
            print(
                f"Warning! No analytical gradients for L-PDFT with {template_dict['pdft-functional']} given!"
            )
            print(f"Allowed functionals are: {', '.join(ALLOWED_FUNCTIONALS)}")
            sys.exit(1)

        if "nacdr" in qmin:
            print("NACdr not allowed with L-PDFT!")
            sys.exit(1)

    active_multiplicities = [
        mult for mult, roots in enumerate(qmin["states"], 1) if roots > 0
    ]
    if active_multiplicities:
        parity = active_multiplicities[0] % 2
        if any(mult % 2 != parity for mult in active_multiplicities):
            print("All requested multiplicities must have the same electron-number parity.")
            sys.exit(1)
        nelecas = template_dict["nelecas"]
        for mult in active_multiplicities:
            spin = mult - 1
            if nelecas < spin or (nelecas - spin) % 2:
                print(
                    f"Active space with {nelecas} electrons is incompatible "
                    f"with multiplicity {mult}."
                )
                sys.exit(1)

    if qmin["method"] == 3 and len(active_multiplicities) > 1:
        print("CMS-PDFT is not compatible with the state-average-mix spin solver.")
        sys.exit(1)

    ALLOWED_SOC_HAMILTONIANS = ["DKH", "BP"]
    template_dict["soc-hamiltonian"] = template_dict["soc-hamiltonian"].upper()
    if template_dict["soc-hamiltonian"] not in ALLOWED_SOC_HAMILTONIANS:
        print(f"Unknown SOC Hamiltonian '{template_dict['soc-hamiltonian']}'. Allowed: {ALLOWED_SOC_HAMILTONIANS}")
        sys.exit(1)

    allowed_dipole_origins = ["coord-center", "mass-center", "charge-center"]
    template_dict["dipole-origin"] = template_dict["dipole-origin"].lower().replace(
        "_", "-"
    )
    if template_dict["dipole-origin"] not in allowed_dipole_origins:
        print(
            f"Unknown dipole origin '{template_dict['dipole-origin']}'. "
            f"Allowed: {allowed_dipole_origins}"
        )
        sys.exit(1)

    if "soc" in qmin:
        if 3 not in active_multiplicities:
            print("SOC requires at least one triplet state.")
            sys.exit(1)
        if qmin["method"] != 1:
            print("SOC is only implemented for L-PDFT")
            sys.exit(1)

    qmin["template"] = template_dict

    # decide which type of gradients to do..
    # 0 = analytical CASSCF gradients in 1 thread/pyscf object (serially)
    # 1 = analytical CASSCF gradients in separate threads/pyscf objects. Possibly distributed over several CPUs (parallel)
    if "grad" in qmin or "nacdr" in qmin:
        if qmin["ncpu"] > 1:
            qmin["gradmode"] = 1

        else:
            qmin["gradmode"] = 0

    else:
        qmin["gradmode"] = 0

    qmin["ncpu"] = max(1, qmin["ncpu"])

    # check the save directory
    if "samestep" in qmin:
        if not os.path.isfile(os.path.join(qmin["savedir"], "pyscf.chk")):
            print("File 'pyscf.chk' missing in SAVEDIR!")
            sys.exit(1)

        if "overlap" in qmin:
            if not os.path.isfile(os.path.join(qmin["savedir"], "pyscf.old.chk")):
                print("File 'pyscf.old.chk' missing in SAVEDIR!")
                sys.exit(1)

    elif "overlap" in qmin:
        if not os.path.isfile(os.path.join(qmin["savedir"], "pyscf.chk")):
            print("File 'pyscf.chk' missing in SAVEDIR")
            sys.exit(1)

    qmin["version"] = get_version()

    if PRINT:
        print_qmin(qmin)

    return qmin


def generate_joblist(qmin):
    """Split the full job into subtasks, each with a qmin dict, a WORKDIR
    structure: joblist = [{WORKDIR: QMin, ..}, {..}, ..]
    each element of the joblist is a est of jobs, and all jobs from the first
    set need to be completed before the second set can be processed."""
    joblist = []
    if qmin["gradmode"] == 0:
        # Serial case on 1 cpu
        qmin_1 = deepcopy(qmin)
        qmin_1["master"] = []
        qmin_1["ncpu"] = 1
        qmin["nslots_pool"] = [1]
        joblist.append({"master": qmin_1})

    elif qmin["gradmode"] == 1:
        # Analytical gradients for several states on several cpus
        # do wave function and dm, soc, overlap always first
        # afterwards do gradients and nacdr asynchronously
        qmin_1 = deepcopy(qmin)
        qmin_1["master"] = []
        qmin_1["gradmap"] = []
        qmin_1["nacmap"] = []
        qmin["nslots_pool"] = [1]
        joblist.append({"master": qmin_1})

        qmin_2 = deepcopy(qmin)
        remove = [
            "h",
            "soc",
            "dm",
            "always_guess",
            "always_orb_init",
            "comment",
            "ncpu",
            "init",
            "veloc",
            "overlap",
            "ion",
            "molden",
        ]
        for r in remove:
            if r in qmin_2:
                del qmin_2[r]

        qmin_2["gradmode"] = 0
        qmin_2["pargrad"] = []
        qmin_2["samestep"] = []
        ntasks = len(qmin["gradmap"]) + len(qmin["nacmap"])

        # Determine number of slots (processes) and number of cpus for each slot
        # right now, we just do this...
        nslots = qmin["ncpu"]
        cpu_per_run = [1] * ntasks

        joblist.append({})
        icount = 0
        for grad in qmin["gradmap"]:
            qmin_3 = deepcopy(qmin_2)
            qmin_3["gradmap"] = [grad]
            qmin_3["nacmap"] = []
            qmin_3["ncpu"] = cpu_per_run[icount]
            icount += 1
            joblist[-1]["grad_%i_%i" % grad] = qmin_3

        for nac in qmin["nacmap"]:
            qmin_3 = deepcopy(qmin_2)
            qmin_3["nacmap"] = [nac]
            qmin_3["gradmap"] = []
            qmin_3["overlap"] = [
                [j + 1, i + 1] for i in range(qmin["nmstates"]) for j in range(i + 1)
            ]
            qmin_3["overlap_nacs"] = []
            qmin_3["ncpu"] = cpu_per_run[icount]
            icount += 1
            joblist[-1]["nacdr_%i_%i_%i_%i" % nac] = qmin_3

        qmin["nslots_pool"].append(nslots)

    if DEBUG:
        pprint.pprint(joblist, depth=3)

    return qmin, joblist


def move_chk_file(qmin):
    """Moves all relevant chk files in the savedir to old-chk files"""
    source = os.path.join(qmin["savedir"], "pyscf.chk")
    target = os.path.join(qmin["savedir"], "pyscf.old.chk")
    if not os.path.isfile(source):
        print(f"File {source} not found, cannot move to old!")
        sys.exit(1)

    if DEBUG:
        print(f"Copy:\t{source}\t==>\t{target}")

    shutil.copy(source, target)


def setup_workdir(qmin):
    """Make the scratch directory, or clean it if it exists. Copy any necessary files."""
    work_dir = qmin["scratchdir"]
    save_dir = qmin["savedir"]

    if os.path.exists(work_dir):
        if not os.path.isdir(work_dir):
            print(f"{work_dir} exists and is not a directory!")
            sys.exit(1)

        else:
            if DEBUG:
                print(f"Remake\t{work_dir}")
            shutil.rmtree(work_dir)
            os.makedirs(work_dir)

    else:
        if DEBUG:
            print(f"Making\t{work_dir}")
        os.makedirs(work_dir)

    source_chk = None
    if "always_guess" not in qmin:
        if "init" in qmin or "always_orb_init" in qmin:
            source_chk = os.path.join(qmin["pwd"], "pyscf.init.chk")

        elif "samestep" in qmin:
            source_chk = os.path.join(save_dir, "pyscf.chk")

        else:
            source_chk = os.path.join(save_dir, "pyscf.old.chk")

    if source_chk is not None and os.path.isfile(source_chk):
        target_chk = os.path.join(work_dir, "pyscf.old.chk")
        if DEBUG:
            print(f"Copying\t{source_chk}\t==>\t{target_chk}", flush=True)
        shutil.copy(source_chk, target_chk)


def save_chk_file(qmin):
    work_dir = qmin["scratchdir"]
    save_dir = qmin["savedir"]

    source_chk = os.path.join(work_dir, "pyscf.chk.master")
    if os.path.isfile(source_chk):
        target_chk = os.path.join(save_dir, "pyscf.chk")
        if DEBUG:
            print(f"Copying\t{source_chk}\t==>\t{target_chk}", flush=True)
        shutil.copy(source_chk, target_chk)


def build_mol(qmin):
    log_file = f"PySCF_{os.path.basename(qmin['scratchdir'])}.log"
    previous_chk = os.path.join(qmin["scratchdir"], "pyscf.old.chk")
    verbose = qmin["template"]["verbose"]
    active_multiplicities = _get_active_multiplicities(qmin)
    reference_spin = active_multiplicities[0][0] - 1

    if os.path.isfile(previous_chk) and "samestep" in qmin:
        if DEBUG:
            print(f"Loading mol from chkfile {previous_chk}", flush=True)
        mol = lib.chkfile.load_mol(previous_chk)
        mol.output = log_file
        mol.verbose = verbose
        mol.max_memory = qmin["memory"]
        mol.build()

    else:
        mol = gto.Mole(
            atom=qmin["geo"],
            unit="Bohr",
            basis=qmin["template"]["basis"],
            output=log_file,
            verbose=verbose,
            symmetry=False,
            charge=qmin["template"]["charge"],
            spin=reference_spin,
            max_memory=qmin["memory"],
        )
        mol.build()

    return mol


def _get_active_multiplicities(qmin):
    """returns ``(multiplicity, number of roots)`` for requested spin spaces."""
    return [
        (mult, nroots)
        for mult, nroots in enumerate(qmin["states"], 1)
        if nroots > 0
    ]


def _is_multispin(qmin):
    return len(_get_active_multiplicities(qmin)) > 1


def _get_solver_multiplicities(qmin):
    """returns spin spaces in the order used by the PySCF solver."""
    if _is_multispin(qmin):
        return [
            (mult, qmin["template"]["roots"][mult - 1])
            for mult, _ in _get_active_multiplicities(qmin)
        ]
    mult = _get_active_multiplicities(qmin)[0][0]
    return [(mult, qmin["template"]["roots"][mult - 1])]


def _nelecas_for_multiplicity(nelecas, mult):
    """returns alpha/beta active electron counts for a spin multiplicity."""
    nelec = sum(nelecas) if isinstance(nelecas, (tuple, list)) else nelecas
    spin = mult - 1
    if nelec < spin or (nelec - spin) % 2:
        raise ValueError(
            f"Active space with {nelec} electrons is incompatible with multiplicity {mult}"
        )
    return ((nelec + spin) // 2, (nelec - spin) // 2)


def gen_solver(mol, qmin):
    use_density_fit = qmin["template"].get("density-fit")
    df_auxbasis = qmin["template"].get("df-auxbasis")
    if df_auxbasis in (None, "", "none"):
        df_auxbasis = None

    active_multiplicities = _get_active_multiplicities(qmin)
    mf = scf.RHF(mol) if mol.spin == 0 else scf.ROHF(mol)
    if use_density_fit:
        print(f"Using density fitting; auxbasis={df_auxbasis}", flush=True)
        mf = mf.density_fit(auxbasis=df_auxbasis)
    mf.max_cycle = 0
    mf.run()

    ncas = qmin["template"]["ncas"]
    nelecas = qmin["template"]["nelecas"]
    multispin = _is_multispin(qmin)
    if not multispin:
        nelecas = _nelecas_for_multiplicity(nelecas, active_multiplicities[0][0])

    if qmin["method"] == 0:
        solver = mcscf.CASSCF(mf, ncas, nelecas)
    else:
        functional = qmin["template"]["pdft-functional"]
        grids_level = qmin["template"]["grids-level"]
        try:
            from pyscf import mcpdft
            solver = mcpdft.CASSCF(mf, functional, ncas, nelecas, grids_level=grids_level)
        except ImportError as e:
            print("MC-PDFT requested but pyscf-forge not installed")
            raise e

    if use_density_fit:
        solver = mcscf.density_fit(solver, auxbasis=df_auxbasis)

    if multispin:
        try:
            from pyscf.csf_fci import csf_solver
        except ImportError as e:
            print("pyscf-forge with CSF solver required for mix-spin calculations")
            raise e

        spin_spaces = _get_solver_multiplicities(qmin)
        total_roots = sum(nroots for _, nroots in spin_spaces)
        weights = [1.0 / total_roots] * total_roots

        solvers = []
        for mult, nroots in spin_spaces:
            spin_solver = csf_solver(mol, smult=mult)
            spin_solver.nroots = nroots
            spin_solver.spin = mult - 1
            solvers.append(spin_solver)

        if qmin["method"] == 1:
            solver = solver.multi_state_mix(solvers, weights, "lin")
        else:
            solver = mcscf.state_average_mix_(solver, solvers, weights)

    else:
        mult = active_multiplicities[0][0]
        nroots = qmin["template"]["roots"][mult - 1]
        weights = [1.0 / nroots] * nroots

        try:
            from pyscf.csf_fci import csf_solver
            solver.fcisolver = csf_solver(mol, smult=mult)
        except ImportError:
            spin = (mult - 1) / 2
            solver.fix_spin_(
                ss=spin * (spin + 1), shift=qmin["template"]["fix-spin-shift"]
            )

        if qmin["method"] == 1:
            solver = solver.multi_state(weights, method="lin")
        elif qmin["method"] == 3:
            solver = solver.multi_state(weights, method="cms")
        elif qmin["method"] in (2, 0):
            solver = solver.state_average(weights)

    solver.conv_tol = qmin["template"]["conv-tol"]
    solver.conv_tol_grad = qmin["template"]["conv-tol-grad"]

    solver.max_stepsize = qmin["template"]["max-stepsize"]
    solver.max_cycle_macro = qmin["template"]["max-cycle-macro"]
    solver.max_cycle_micro = qmin["template"]["max-cycle-micro"]

    solver.ah_level_shift = qmin["template"]["ah-level-shift"]
    solver.ah_conv_tol = qmin["template"]["ah-conv-tol"]
    solver.ah_max_cycle = qmin["template"]["ah-max-cycle"]
    solver.ah_lindep = qmin["template"]["ah-lindep"]
    solver.ah_start_tol = qmin["template"]["ah-start-tol"]
    solver.ah_start_cycle = qmin["template"]["ah-start-cycle"]

    if "master" in qmin:
        solver.chkfile = os.path.join(qmin["scratchdir"], "pyscf.chk.master")
        solver.chk_ci = True

    old_chk = os.path.join(qmin["scratchdir"], "pyscf.old.chk")
    if os.path.isfile(old_chk):
        print(f"Loading MO guess from chk: {old_chk}", flush=True)
        try:
            mo_guess = lib.chkfile.load(old_chk, "mcscf/mo_coeff")
            try:
                prev_mol = lib.chkfile.load_mol(old_chk)
            except (TypeError, OSError, KeyError, ValueError) as exc:
                print(f"Could not read checkpoint molecule ({exc}); projecting without prev_mol.", flush=True)
                prev_mol = None
            same_checkpoint_molecule = (
                prev_mol is not None
                and gto.same_mol(prev_mol, mol, cmp_basis=True)
            )
            if same_checkpoint_molecule:
                print(
                    "Checkpoint geometry and basis match exactly; "
                    "loading orbitals without projection.",
                    flush=True,
                )
                solver.mo_coeff = mo_guess
            else:
                print(
                    "Projecting checkpoint orbitals without replacing the "
                    "converged core by mean-field core orbitals.",
                    flush=True,
                )
                solver.mo_coeff = mcscf.project_init_guess(
                    solver, mo_guess, prev_mol=prev_mol, use_hf_core=False
                )
            solver.ci = None
        except (TypeError, OSError, KeyError, ValueError) as exc:
            print(f"Could not read MO guess from checkpoint ({exc}); using a fresh initial guess.", flush=True)

    solver.kernel(solver.mo_coeff)
    return solver


def _lpdft_state_idx(mult, state_1indexed, qmin):
    """Converts SHARC state labeling to PySCF's labeling
    (mult, state) to PDFT state index (0 index)"""
    offset = 0
    for solver_mult, nroots in _get_solver_multiplicities(qmin):
        if solver_mult == mult:
            return offset + state_1indexed - 1
        offset += nroots
    raise ValueError(f"Multiplicity {mult} is not present in the PySCF solver")


def _get_dipole_origin(mol, origin):
    normalized = origin.upper().replace("-", "_")
    coords = mol.atom_coords()
    if normalized == "COORD_CENTER":
        return np.zeros(3)
    if normalized == "MASS_CENTER":
        masses = mol.atom_mass_list()
        return masses.dot(coords) / masses.sum()
    if normalized == "CHARGE_CENTER":
        charges = mol.atom_charges()
        return charges.dot(coords) / charges.sum()
    raise ValueError(
        f"Unknown dipole origin '{origin}'; use coord-center, mass-center, or charge-center"
    )


def get_dipole_elements(solver, qmin):
    mol = solver.mol
    mo_core = solver.mo_coeff[:, : solver.ncore]
    mo_cas = solver.mo_coeff[:, solver.ncore : solver.ncore + solver.ncas]
    ncas = solver.ncas
    nelecas = solver.nelecas
    nmstates = qmin["nmstates"]

    gauge_center = _get_dipole_origin(
        mol, qmin["template"].get("dipole-origin", "coord-center")
    )

    dm_core = 2 * mo_core @ mo_core.conj().T

    charges = mol.atom_charges()
    coords = mol.atom_coords()
    coords -= gauge_center
    nucl_term = charges.dot(coords)

    with mol.with_common_origin(gauge_center):
        dipole_ints = mol.intor("int1e_r")

    dip_matrix = np.zeros(shape=(3, nmstates, nmstates))

    def _base_fci_call(fci_solver, method, *args, **kwargs):
        base_class = getattr(fci_solver, "_base_class", None)
        if base_class is not None and hasattr(fci_solver, "weights"):
            return getattr(base_class, method)(fci_solver, *args, **kwargs)
        return getattr(fci_solver, method)(*args, **kwargs)

    def _state_rdm1(ci_I, fci_solver, nelec):
        return _base_fci_call(fci_solver, "make_rdm1", ci_I, ncas, nelec)

    def _trans_rdm1(ci_bra, ci_ket, fci_solver, nelec):
        return _base_fci_call(fci_solver, "trans_rdm1", ci_bra, ci_ket, ncas, nelec)

    spin_spaces = _get_solver_multiplicities(qmin)
    if _is_multispin(qmin):
        fci_solvers = solver.fcisolver.fcisolvers
    else:
        fci_solvers = [solver.fcisolver]

    sharc_indices = {
        tuple(state): index - 1 for index, state in qmin["statemap"].items()
    }
    ci_offset = 0
    requested_roots = dict(_get_active_multiplicities(qmin))
    for (mult, solver_nroots), fci_solver in zip(spin_spaces, fci_solvers):
        nroots = requested_roots.get(mult, 0)
        nelec = _nelecas_for_multiplicity(nelecas, mult)
        spatial_dipoles = np.zeros((3, nroots, nroots))

        for state in range(nroots):
            casdm1 = _state_rdm1(solver.ci[ci_offset + state], fci_solver, nelec)
            dm1 = dm_core + mo_cas @ casdm1 @ mo_cas.conj().T
            spatial_dipoles[:, state, state] = nucl_term - np.einsum(
                "xij,ji->x", dipole_ints, dm1
            )

        for bra in range(nroots):
            for ket in range(bra + 1, nroots):
                t_dm = _trans_rdm1(
                    solver.ci[ci_offset + bra],
                    solver.ci[ci_offset + ket],
                    fci_solver,
                    nelec,
                )
                t_dm = mo_cas @ t_dm @ mo_cas.conj().T
                t_dip = -np.einsum("xij,ji->x", dipole_ints, t_dm)
                spatial_dipoles[:, bra, ket] = t_dip
                spatial_dipoles[:, ket, bra] = t_dip

        # The dipole is evaluated in the spin-orbit-free (MCH) basis.  It is
        # spin independent, so only states with equal multiplicity and Ms can
        # couple; all other matrix elements remain zero.
        ms_values = sorted(
            {ms for state_mult, _, ms in sharc_indices if state_mult == mult}
        )
        for ms in ms_values:
            for bra in range(nroots):
                for ket in range(nroots):
                    sharc_bra = sharc_indices[(mult, bra + 1, ms)]
                    sharc_ket = sharc_indices[(mult, ket + 1, ms)]
                    dip_matrix[:, sharc_bra, sharc_ket] = spatial_dipoles[
                        :, bra, ket
                    ]
        ci_offset += solver_nroots

    return dip_matrix


def get_soc_hamiltonian(solver, qmin):
    """Compute spin-orbit Hamiltonian
    off-diagonal are the SOC matrix elements in MCH representation
    diagonal elements are spin-free L-PDFT energies
    """
    from pyscf.siso import SISO

    nmstates = qmin["nmstates"]
    soc_ham_type = qmin["template"].get("soc-hamiltonian", "DKH")

    spin_spaces = _get_active_multiplicities(qmin)
    modelspace = [(nroots, mult) for mult, nroots in spin_spaces]

    ci_orig = list(solver.ci) if isinstance(solver.ci, list) else solver.ci
    e_states_orig = list(solver.e_states)

    try:
        siso = SISO(solver,modelspace,amf=False,mmf=True,ham=soc_ham_type)
        siso.build_imds()
        h_siso = siso.compute_hamiltonian()
    finally:
        solver.ci = ci_orig
        solver.e_states = e_states_orig

    expected_siso_size = sum(mult * nroots for mult, nroots in spin_spaces)
    if h_siso.shape != (expected_siso_size, expected_siso_size):
        raise RuntimeError(
            f"SOC Hamiltonian has shape {h_siso.shape}, "
            f"expected {(expected_siso_size, expected_siso_size)}"
        )

    # PySCF groups Ms components by spatial root; SHARC groups roots by Ms.
    # The SISO model space contains the roots requested in QM.in.
    requested_roots = dict(_get_active_multiplicities(qmin))
    perm = []
    offset = 0
    for mult, solver_nroots in spin_spaces:
        for ms_idx in range(mult):
            for state_idx in range(requested_roots.get(mult, 0)):
                perm.append(offset + mult * state_idx + ms_idx)
        offset += mult * solver_nroots

    h_sharc = h_siso[np.ix_(perm, perm)]
    if h_sharc.shape != (nmstates, nmstates):
        raise RuntimeError(
            f"Reordered SOC Hamiltonian has shape {h_sharc.shape}, "
            f"expected {(nmstates, nmstates)}"
        )

    if not np.allclose(h_sharc, h_sharc.conj().T, atol=1e-8):
        max_dev = np.max(np.abs(h_sharc - h_sharc.conj().T))
        raise RuntimeError(f"SOC Hamiltonian is not Hermitian after "f"SHARC reordering: {max_dev}")

    expected_diag = _build_energy_vector(solver, qmin)
    np.fill_diagonal(h_sharc, expected_diag)

    return h_sharc

def get_grad(solver, qmin):
    nmstates = qmin["nmstates"]
    natom = qmin["natom"]
    grad = np.zeros((nmstates, natom, 3))
    err = 0
    computed = {}

    solver_grad = solver.nuc_grad_method()
    solver_grad.max_cycle = qmin["template"]["grad-max-cycle"]

    for i in sorted(qmin["statemap"]):
        mult, state, _ = tuple(qmin["statemap"][i])
        if (mult, state) in qmin["gradmap"]:
            if (mult, state) not in computed:
                lpdft_idx = _lpdft_state_idx(mult, state, qmin)
                de = solver_grad.kernel(state=lpdft_idx)
                if not solver_grad.converged:
                    print(f"Gradient failed to converge: {qmin['statemap'][i]}", flush=True)
                    err = 1
                computed[(mult, state)] = de
            grad[i - 1] = computed[(mult, state)]

    return grad, err


def get_nac(solver, qmin):
    nmstates = qmin["nmstates"]
    nac = np.zeros(shape=(nmstates, nmstates, qmin["natom"], 3))
    err = 0

    solver_nac = solver.nac_method()

    for i in sorted(qmin["statemap"]):
        for j in sorted(qmin["statemap"]):
            m1, s1, ms1 = tuple(qmin["statemap"][i])
            m2, s2, ms2 = tuple(qmin["statemap"][j])
            if m1 != m2:
                continue

            if ms1 != ms2:
                continue

            if s1 == s2:
                continue

            if (m1, s1, m2, s2) in qmin["nacmap"]:
                bra = i - 1
                ket = j - 1
                nacdr = solver_nac.kernel(state=(bra, ket))
                nac[bra][ket] = nacdr
                nac[ket][bra] = -nacdr
                if not solver_nac.converged:
                    print(
                        f"NACDR failed to converge: {qmin['statemap'][i]}, {qmin['statemap'][j]}"
                    )
                    err = 1
    return nac, err


def _build_energy_vector(solver, qmin):
    """Expand PySCF spatial-state energies over SHARC Ms components."""
    e_spatial = list(solver.e_states)
    requested_roots = dict(_get_active_multiplicities(qmin))
    energies = []
    offset = 0
    for mult, solver_nroots in _get_solver_multiplicities(qmin):
        nroots = requested_roots.get(mult, 0)
        for _ in range(mult):
            energies.extend(e_spatial[offset : offset + nroots])
        offset += solver_nroots

    return np.array(energies)


def run_calc(qmin):
    err = 0
    result = {}


    setup_workdir(qmin)
    mol = build_mol(qmin)

    solver = gen_solver(mol, qmin)

    result = {}
    if not solver.converged:
        print("Calculator failed to converge!", flush=True)
        err += 1

    if "h" in qmin:
        # Spin-free energies: repeat each spatial energy for each ms substate
        result["energies"] = _build_energy_vector(solver, qmin)

    if "soc" in qmin:
        # Full spin-orbit Hamiltonian (complex, nmstates x nmstates)
        result["soc_hamiltonian"] = get_soc_hamiltonian(solver, qmin)

    if "dm" in qmin:
        result["dipole"] = get_dipole_elements(solver, qmin)

    if "molden" in qmin:
        from pyscf.tools import molden

        molden.from_mcscf(solver, os.path.join(qmin["savedir"], "pyscf.molden"))

    if qmin["gradmap"]:
        result["grad"], e = get_grad(solver, qmin)
        err += e

    if qmin["nacmap"]:
        result["nacdr"], e = get_nac(solver, qmin)
        err += e

    save_chk_file(qmin)

    return err, result


def run_jobs(joblist, qmin):
    """Runs all of the jobs specified in joblist"""
    if "newstep" in qmin:
        move_chk_file(qmin)

    lib.param.TMPDIR = qmin["scratchdir"]
    lib.param.MAX_MEMORY = qmin["memory"]
    os.environ['OMP_NUM_THREADS'] = str(qmin["ncpu"])
    os.environ['MKL_NUM_THREADS'] = str(qmin["ncpu"])
    os.environ['OPENBLAS_NUM_THREADS'] = str(qmin["ncpu"])
    lib.num_threads(qmin["ncpu"])

    print(">>>>>>>>>>>>> Starting the job execution")

    error_codes = {}
    result = {}
    outputs = {}
    for idx, jobset in enumerate(joblist):
        if not jobset:
            continue

        pool = Pool(processes=qmin["nslots_pool"][idx])
        for job in jobset:
            qmin_1 = jobset[job]

            qmin_1["scratchdir"] = os.path.join(qmin_1["scratchdir"], job)
            outputs[job] = pool.apply_async(run_calc, [qmin_1])

            time.sleep(qmin["delay"])

        pool.close()
        pool.join()

        print("")

    for i in outputs:
        error_codes[i], result[i] = outputs[i].get()

    if PRINT:
        string = "  " + "=" * 40 + "\n"
        string += "||" + " " * 40 + "||\n"
        string += "||" + " " * 10 + "All Tasks completed!" + " " * 10 + "||\n"
        string += "||" + " " * 40 + "||\n"
        string += "  " + "=" * 40 + "\n"
        print(string)
        j = 0
        string = "Error Codes:\n\n"
        for i in error_codes:
            string += "\t%s\t%i" % (i + " " * (10 - len(i)), error_codes[i])
            j += 1
            if j == 4:
                j = 0
                string += "\n"
        print(string)

    if any((i != 0 for i in error_codes.values())):
        print("Some subprocesses did not finish successfully!")
        sys.exit(1)

    return result


def combine_result(qmin, result):
    output = {}
    nmstates = qmin["nmstates"]
    natom = qmin["natom"]

    if "h" in qmin:
        output["energies"] = result["master"]["energies"]
    if "soc" in qmin:
        output["soc_hamiltonian"] = result["master"]["soc_hamiltonian"]
    if "dm" in qmin:
        output["dipole"] = result["master"]["dipole"]

    if "grad" in qmin:
        output["grad"] = np.zeros(shape=(nmstates, natom, 3))
        for job in result:
            if "grad" in result[job]:
                output["grad"] += result[job]["grad"]

    if "nacdr" in qmin:
        output["nacdr"] = np.zeros(shape=(nmstates, nmstates, natom, 3))
        for job in result:
            if "nacdr" in result[job]:
                output["nacdr"] += result[job]["nacdr"]

    return output


def write_ham(qmin, result):
    """If SOC was computed, write the full SOC Hamiltonian
    write a diagonal matrix with spin-free energies if not
    """
    nmstates = qmin["nmstates"]
    string = f"! 1 Hamiltonian Matrix ({nmstates}x{nmstates}, complex)\n"
    string += f"{nmstates} {nmstates}\n"

    if "soc_hamiltonian" in result:
        h = result["soc_hamiltonian"]
        for i in range(nmstates):
            for j in range(nmstates):
                string += f"{eformat(h[i, j].real, 9, 3)} {eformat(h[i, j].imag, 9, 3)} "
            string += "\n"
    else:
        for i in range(nmstates):
            for j in range(nmstates):
                if i != j:
                    string += f"{eformat(0.0, 9, 3)} {eformat(0.0, 9, 3)} "
                else:
                    e = result["energies"][i]
                    string += f"{eformat(e.real, 9, 3)} {eformat(e.imag, 9, 3)} "
            string += "\n"

    string += "\n"
    return string


def write_dm(qmin, result):
    nmstates = qmin["nmstates"]
    string = f"! 2 Dipole Moment Matrices (3x{nmstates}x{nmstates}, complex)\n"
    for dipole_xyz in result["dipole"]:
        string += f"{nmstates} {nmstates}\n"
        for bra in dipole_xyz[:nmstates, :nmstates]:
            for element in bra:
                string += f"{eformat(element.real, 9, 3)} {eformat(element.imag, 9, 3)} "
            string += "\n"

    return string


def write_grad(qmin, result):
    states = qmin["states"]
    nmstates = qmin["nmstates"]
    natom = qmin["natom"]
    string = f"! 3 Gradient Vectors ({nmstates}x{natom}x3, real)\n"
    for idx, (imult, istate, ims) in enumerate(itnmstates(states)):
        string += f"{natom} 3 ! {imult} {istate} {ims}\n"
        for atom in result["grad"][idx]:
            for coord in atom:
                string += f"{eformat(coord, 9, 3)} "
            string += "\n"

    return string


def write_nac(qmin, result):
    states = qmin["states"]
    nmstates = qmin["nmstates"]
    natom = qmin["natom"]
    string = f"! 5 Nonadiabatic couplings (ddr) ({nmstates}x{nmstates}x{natom}x3)\n"
    i = 0
    for imult, istate, ims in itnmstates(states):
        j = 0
        for jmult, jstate, jms in itnmstates(states):
            string += f"{natom} {3} ! {imult} {istate} {ims} {jmult} {jstate} {jms}\n"
            for atom in result["nacdr"][i][j]:
                for coord in atom:
                    string += f"{eformat(coord, 12, 3)} "
                string += "\n"

            j += 1
        i += 1

    return string


def write_qmout_time(runtime):
    return f"! 8 Runtime\n{eformat(runtime, 9, 3)}\n"


def write_qmout(qmin, result, qmin_filename):
    """Writes the requested quantities to the file which SHARC reads in. The filename is qmin_filename with everything after the first dot replaced by 'out'."""
    if "." in qmin_filename:
        idx = qmin_filename.find(".")
        outfilename = qmin_filename[:idx] + ".out"

    else:
        outfilename = qmin_filename + ".out"

    if PRINT:
        print(f"===> Writing output to file {outfilename} in SHARC Format\n")


    # add header info
    string = '! 0 Basic information\nstates '
    for i in qmin['states']:
        string += '%i ' % i
    string += '\nnmstates %i\n' % qmin['nmstates']
    string += 'natom %i\n' % qmin['natom']
    string += 'npc 0\n'
    string += 'charges '
    for i in qmin['states']:
        string += '%i ' % 0
    string += '\n\n'

    # add data
    if "h" in qmin or "soc" in qmin:
        string += write_ham(qmin, result)
        string += '\n'

    if "dm" in qmin:
        string += write_dm(qmin, result)
        string += '\n'

    if "grad" in qmin:
        string += write_grad(qmin, result)
        string += '\n'

    if "nacdr" in qmin:
        string += write_nac(qmin, result)
        string += '\n'

    string += write_qmout_time(result["runtime"])
    with open(os.path.join(qmin["pwd"], outfilename), "w") as f:
        f.write(string)

    return


def cleanup_directory(dir):
    if PRINT:
        print(f"===> Removing directory {dir}\n")

    try:
        shutil.rmtree(dir)

    except OSError:
        print("fCould not remove directory {dir}")


def main():
    try:
        env_print = os.getenv("SH2CAS_PRINT")
        if env_print and env_print.lower() == "false":
            global PRINT
            PRINT = False

        env_debug = os.getenv("SH2CAS_DEBUG")
        if env_debug and env_debug.lower() == "true":
            global DEBUG
            DEBUG = True

    except ValueError:
        print(
            "SH2CAS_PRINT or SH2CAS_DEBUG environment variables do no evaluate to booleans!"
        )

    if len(sys.argv) != 2:
        print(
            f"""Usage:
./SHARC_PYSCF.py <qmin>
version: {_version_}
date: {_versiondate_}
changelog: {_change_log_}"""
        )
        sys.exit(1)

    qmin_filename = sys.argv[1]

    print_header()
    qmin = readqmin(qmin_filename)

    qmin, joblist = generate_joblist(qmin)

    result = run_jobs(joblist, qmin)
    result = combine_result(qmin, result)

    runtime = measure_time()
    result["runtime"] = runtime

    write_qmout(qmin, result, qmin_filename)

    if not DEBUG:
        cleanup_directory(qmin["scratchdir"])
        if "cleanup" in qmin:
            cleanup_directory(qmin["savedir"])

    if PRINT or DEBUG:
        print("#================ END ================#")


if __name__ == "__main__":
    main()
