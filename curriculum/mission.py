# -*- coding: utf-8 -*-
import phrasing 
import environmentGenerator
import vocab






class UnitaryMission(object):
    '''
    A unitary mission is the building block of a mission, it is parametrized according to parameters={action, adjective, noun, location}
    parameter is a dictionnary (not a tuple)
    '''
    def __init__(self, parameters):
        assert parameters['action'] in vocab.ACTIONS, 'mission {} not recognized'.format(parameters['action'])
        assert parameters['adjective'] in vocab.ADJECTIVES, 'adjective {} not recognized'.format(parameters['adjective'])
        assert parameters['noun'] in vocab.NOUNS, 'noun {} not recognized'.format(parameters['noun'])
        assert parameters['location'] in vocab.LOCATIONS, 'location {} not recognized'.format(parameters['location'])
        
        self.parameters=parameters
        self.expression=phrasing.param2Seq(parameters)
        

class Mission(object):
    '''
    In the curriculum framework, a mission is repressented as a sequence of unitary missions linked by logic words such as 'AND'
    Each unitary mission is parametrized according to (Action, Adj, Noun, Location)
    We
    '''
    def __init__(self, 
                 sequenceOfUnitaryMissions,
                 explorationDifficulty=False,
                 diversityDifficulty=False):
        self.sequenceOfUnitaryMissions=sequenceOfUnitaryMissions
        self.expression=phrasing.mission2seq(sequenceOfUnitaryMissions)
        self.suitedEnv=environmentGenerator.generateEnv(self)
        
    

def main():
    #tests
    
    parameters={'action':'pick up',
                'adjective':'blue',
                'noun':'key',
                'location':'north'}
    
    unitaryMissionTest1=UnitaryMission(parameters)
    print('expression 1 : ', unitaryMissionTest1.expression)
    
   
    missionTest1=Mission([unitaryMissionTest1, unitaryMissionTest1])
    print('full expression : ', missionTest1.expression)


    print('env suited :', missionTest1.suitedEnv)



    print('actions allowed', vocab.ACTIONS)
    print('adjectives allowed', vocab.ADJECTIVES)
    print('nouns allowed', vocab.NOUNS)

if __name__ == "__main__":
    main()