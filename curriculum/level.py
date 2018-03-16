# -*- coding: utf-8 -*-
import vocab
import mission

class Level(object):
    def __init__(self, missions=[]):
        self.missions=missions
        
        
    def addMission(self, mission):
        self.missions.append(mission)
        
        
    def learnPickUp(self):
        '''
        define a level that teaches a certain concept
        the concept is either an action, a adjective or a noun
        we leave the learning of a location to a latter time
        
        To do this, we consider all the variations in the missions that keep the concept in it
        namely, to learn the concept of 'door' we will ask the agent to open all kinds of doors with different colors...
        '''
        for color in vocab.ADJECTIVES:
            for obj in ['ball', 'key']:
                parameters={'action':'pick up',
                            'adjective':color,
                            'noun':obj,
                            'location':None}
                localUnitaryMission=mission.UnitaryMission(parameters)
                localMission=mission.Mission([localUnitaryMission])
                self.addMission(localMission)
        
        
        print('we will learn how to pick up with the following missions')
        for miss in self.missions:
            print(miss.expression)
        

def main():
    level=Level()
    level.learnPickUp()
    

if __name__=='__main__':
    main()
    