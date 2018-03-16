# -*- coding: utf-8 -*-

def param2Seq(param):
    
    '''
    converts the parameters of a unitary mission into a string 
    TO BE IMPLEMENTED
    '''
    
    #check for an existing adjective
    if param['adjective'] is not None:
        adj=param['adjective'] +  ' '
    else:
        adj='' 
        
        
    #check for an existing location
    if param['location'] is not None:
        loc=' which is at the ' + param['location']
    else:
        loc='' 
        
    
    
    output=param['action'] + ' the ' + adj + param['noun'] + loc
   
    return(output)
    
    
def mission2seq(unitaryMissionList):
    '''
    converts a list of unitary missions into a string
    TO BE IMPLEMENTED
    '''
    output=''
    numberOfUnitaryMissions=len(unitaryMissionList)
    
    for index in range(numberOfUnitaryMissions):
        output+=param2Seq(unitaryMissionList[index].parameters)
        
        #check if this is the last unitary mission to process
        if index<numberOfUnitaryMissions-1:
            output+=' and then '
        else:
            return(output)
        
    return(output)