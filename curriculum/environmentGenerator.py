# -*- coding: utf-8 -*-


def generateEnv(mission):
    '''
    create an environment that is suited to the mission given
    '''
    numberOfUnitaryMissions=len(mission.sequenceOfUnitaryMissions)
    return('This is a gym env for {} unitary missions'.format(numberOfUnitaryMissions))