import torch
import numpy as np
import os,sys,inspect


class PreProcessor(object):
    def __init__(self,
                 maxSizeOfMissions):
        
        
        print(sys.path[0])

        currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
        parentdir = os.path.dirname(currentdir)+'/model'
        sys.path.insert(0,parentdir) 
        print(sys.path[0])
        #import sentenceEmbedder
        #self.languageModel=sentenceEmbedder.Sentence2Vec()
        sys.path=sys.path[1:]            
        print('language model loaded')
        
        #used for the simple sentence embedding 
        self.Dico={'continue':['continue',"go forward", 'go on', 'keep the same direction!', 'okay continue this way','advance one case'],
                   'left':['go left','turn left', 'now go to the left', 'go to the left!...The other left!!!', 'go on your left'],
                   'right':['go right', 'turn right', 'now go to the right', 'tous à Tribor !!', 'go on your right','good job! now go to the right'],
                   'turn back': ['turn back','go backward','turn yourself', 'make a U turn', 'look behind you...' ],             
                   'key':['take the key!', 'pick up the key', 'pick it up', 'take it', 'key !!!', 'catch the key','toggle the key'   ],
                   'door':['open the door', 'toggle the door', 'open it', 'tire la chevillette et la bobinette cherra...']}

        self.ActionReference={'continue':0,'left':1,'right':2, 'turn back':3, 'key':4, 'door':5,'other':6}
        self.simpleDimensionEmbedding=7
        self.maxSizeOfMissions=maxSizeOfMissions
        
    def simpleSentenceEmbedding(self, strings):
        '''
        implements an easy language model, the different senteces are embedded in a 1-hot vector
        '''
        listOfOneHotVectors=False
        
        originalMission='other'
        #print('missions from the Teacher', strings)
        for mission in strings:
            tmp=torch.zeros(self.simpleDimensionEmbedding)
            #print('mission from teacher :', mission,':')

            for key, value in self.Dico.items():    # for name, age in list.items():  (for Python 3.x)
                if mission in value:
                    originalMission=key
                    #print('mission parent', originalMission)
            index=self.ActionReference[originalMission]
            tmp[index]=1
            #print('sentence embedding', tmp)
            if listOfOneHotVectors is not False:
                listOfOneHotVectors=torch.cat([listOfOneHotVectors,tmp.unsqueeze(0)],dim=0)
            else:
                listOfOneHotVectors=tmp.unsqueeze(0)
            #print('selected keyword', originalMission)    

        return(listOfOneHotVectors)
        
        
        
        
    def preProcessImage(self,img):
        '''
        The image has to be swapped in order to match the pytorch setting
        
        For now we use a env wrapper to swap the dimensions but once we will get rid of it,
        this dimension swapping should be implemented here
        '''
        return(img)
        
        #return(img.transpose(2, 0, 1))
    
    def Code2String(self, listOfAsciiCodes):
        '''
        This function is used when you choosed to encode your string messages in ASCII format
        And you want to convert it to a torch variable using your language model
        '''
        
        #back to original string using the decoder method 
        originalMissions=[]
        for i in range ( listOfAsciiCodes.size()[0]):
            originalMissions+=[self.stringDecoder(listOfAsciiCodes[i]) ]
        
        #using the specified language model
        return(originalMissions)
      
        
    def stringEncoder_LanguageModel(self,string):
        ''' 
        encode a string or a sequence of strings using the ASCII encoding
        '''    
        return(self.languageModel.encodeSent(string))
    
    
    
    def stringEncoder(self,string):
        ''' 
        encode a string using the ASCII encoding
        '''
        code=np.zeros(self.maxSizeOfMissions)
        for i in range(len(string)):
            code[i]=ord(string[i])
        return(torch.from_numpy(code))
            
    def stringDecoder(self,code):
        '''
        decode a pytorch Tensor containing the ASCII codes of the original string
        '''
        string=''
        for x in code:
            if int(x)==0:
                return(string)
            else:
                string+=chr(int(x))
        return(string)