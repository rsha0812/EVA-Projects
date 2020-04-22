
import matplotlib.pyplot as plt

def OneCyclePolicy(LRmax, step, iterations):
    LRmin = LRmax/10;
    LRvalues = []
    for x in range(0,iterations+1):
        cycle = int(1+(x/(2*step)))
        a = abs((x/step)-(2*cycle)+1)
        LRt = LRmin + ((LRmax - LRmin)*(1-a))
        LRvalues.append(LRt)
    return LRvalues
print (OneCyclePolicy(1,5,30))