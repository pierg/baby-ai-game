#!/usr/bin/env python3

import levels
import levels.instr_gen
import levels.env_gen
import levels.verifier
import agents

# NOTE: please make sure that tests are always deterministic

print('Testing environment generation')
levels.env_gen.test()

print('Testing instruction generation')
levels.instr_gen.test()

# TODO: verifier tests
