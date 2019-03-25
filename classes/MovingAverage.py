# encoding: utf-8

from collections import deque

class MovingAverage:
    def __init__(self,N):
        '''Returns moving average of last N elements added. Removes older elements.'''
        self.N=N
        self.sum = 0
        self.queue = deque()
        
    def append(self,value):
        self.queue.append(value)
        self.sum+=value
        if len(self.queue)>self.N:
            old_value = self.queue.popleft()
            self.sum-=old_value
            
    @property
    def value(self):
        return self.sum/len(self.queue)
        

if __name__=='__main__':
    '''Demo use:'''
        
    ma=MovingAverage(2)
    ma.append(10)
    print(ma.value)
    ma.append(20)
    print(ma.value)
    ma.append(30)
    print(ma.value)
    ma.append(40)
    print(ma.value)
    ma.append(50)
    print(ma.value)

    print(ma.queue)

