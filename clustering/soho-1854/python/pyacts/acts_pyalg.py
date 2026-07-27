#!/usr/bin/env python3
# This file is part of the ACTS project
#
# Copyright (C) 2016 CERN for the benefit of the ACTS project
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import os
from pathlib import Path

os.environ["ACTS_SEQUENCER_DISABLE_FPEMON"] = "1"

import acts
import acts.examples
from acts import UnitConstants as u



class PythonTrackFinder(acts.examples.IAlgorithm):
    def __init__(self, name, level):
        acts.examples.IAlgorithm.__init__(self, name, level)

    def execute(self, context):
        return acts.examples.ProcessCode.SUCCESS

if __name__ == "__main__":
    srcdir = Path(__file__).resolve().parent.parent.parent.parent

    detector = acts.examples.GenericDetector(acts.examples.GenericDetector.Config())
    trackingGeometry = detector.trackingGeometry()
    decorators = detector.contextDecorators()



