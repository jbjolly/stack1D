from __future__ import division
import numpy as np


class Image():
    def __init__(   self,
                    fminIm=0,
                    fmaxIm=1,
                    numberOfChannelsIm=1,
                    numberOfLines=1,
                    spectrum=[],
                    lines=[]):
        self.fminIm=fminIm
        self.fmaxIm=fmaxIm
        self.numberOfChannelsIm=numberOfChannelsIm
        self.bandwidth=self.fmaxIm-self.fminIm
        self.chanwidth=self.bandwidth/self.numberOfChannelsIm
        self.numberOfLines=numberOfLines
        self.spectrum=spectrum
        self.lines=lines
    def setLines(self, newLines):
        self.lines=newLines

def Spectrum(frequencies, lines=[], noise=False, sigmaNoise=0):
    spectrum=np.zeros(len(frequencies))
    for line in lines:
        spectrum+=Gauss(f=frequencies,
                        f0=line.fToday,
                        sPeak=line.amp,
                        dF=line.linewidth,
                        noise=noise,
                        sigmaNoise=sigmaNoise)
    return spectrum

class Line():
    def __init__(   self,
                    linewidthRandom=False,
                    linewidthWidth=0.1,
                    linewidth=1,
                    amp=1,
                    randomAmp=False,
                    randomAmpWidth=0,
                    z=0.,
                    fEm=1):
        if linewidthRandom:
            self.linewidth=np.random.normal(linewidth, linewidthWidth)
        else:
            self.linewidth=float(linewidth)
        if randomAmp:
            self.amp=np.random.normal(float(amp),randomAmpWidth)
        else:
            self.amp=float(amp)
        self.z=float(z)
        self.fEm=float(fEm)
        if dz!=0:
            self.observedZ=np.random.normal(self.z, dz)
        else:
            self.observedZ=self.z
        self.fToday=self.fEm/(1+self.z)
        self.observedFToday=self.fEm/(1+self.observedZ)

    def setZ(self, newZ):
        self.observedZ=newZ

def Gauss(  f=[], #this should be a numpy array
            f0=0, #center of your gaussian
            dV='default', #velocity, used to calculate frequency width, set to 'default' to avoid errors
            emittedF=0, #emitted frequency, used to calculate frequency width
            sPeak=1., #amplitude
            dF=1,
            noise=False,
            sigmaNoise=0 ): #frequency width

    if not dF: #if dF is not given by the user it is calculated
        dF=float(dV)*emittedF/c
    sigma=dF/2.3548
    if noise:
        return sPeak*np.exp(-((f-f0)*(f-f0))/(2*sigma*sigma))+np.random.normal(0, sigmaNoise, len(f))
    else:
        return sPeak*np.exp(-((f-f0)*(f-f0))/(2*sigma*sigma))
'''
/&\ STACKSIZE HAS TO BE AN ODD NUMBER
'''
def Stack(images, method='mean', stackSize=0, visu=0):
    if stackSize/2.==int(stackSize/2.):
        raise Exception('STACKSIZE IS AN EVEN NUMBER')
    toStack=np.zeros((len(images), stackSize))
    for i in range(len(images)):
        thisImage=images[i]
        thisImageLines=thisImage.lines
        freqMidLine=np.zeros(len(thisImageLines))
        freqMidLineIndex=np.zeros(len(thisImageLines))#.astype(int)
        for j in range(len(thisImageLines)):
            thisLine=thisImageLines[j]
            freqMidLine[j]=thisLine.observedFToday
            freqMidLineIndex[j]=int((freqMidLine[j]-thisImage.fminIm)/thisImage.chanwidth)
            toStack[i]+=thisImage.spectrum[freqMidLineIndex[j]-int(stackSize/2):freqMidLineIndex[j]+int(stackSize/2)+1]
    if method=='mean':
        return np.average(toStack,0)
    elif method=='median':
        '''
        check median fct
        '''
        return np.median(toStack,0)


def MCMC_once(oldSummedStack=0, oldImages=[], deltaZ=0, sumWidth=0, method='mean'):
    #newZ=np.zeros((numberOfImages, numberOfLinesPerImage))
    newImages=([0 for i in range(numberOfImages)])
    for i,image in enumerate(oldImages):
        tempLines=[]
        for j, line in enumerate(image.lines):
            newZ=np.random.normal(line.observedZ, deltaZ)
            tempLine=line
            tempLine.setZ(newZ)
            tempLines.append(tempLine)
        newImages[i]=image
        newImages[i].setLines(tempLines)

    newSummedStack=np.mean(Stack(newImages, method, stackSize=sumWidth))

    if newSummedStack>oldSummedStack:
        return newSummedStack, newImages
    else:
        return oldSummedStack, oldImages

def MCMC(nRandom=1, oldStack=0, oldImages=[], deltaZ=0, sumWidth=0, method='mean', saveResult=True):
    theseImages=oldImages
    thisSummedStack=oldSummedStack
    if saveResult:
        result=([ 0 for i in range(nRandom) ])
        for i in range(nRandom):
            print 'mcmc', i
            result[i]=MCMC_once(oldSummedStack=thisSummedStack, oldImages=theseImages, deltaZ=deltaZ, sumWidth=sumWidth, method=method)
            thisStack=result[i][0]
            theseImages=result[i][1]
    else:
        for i in range(nRandom):
            print 'mcmc', i
            result=MCMC_once(oldSummedStack=thisSummedStack, oldImages=theseImages, deltaZ=deltaZ, sumWidth=sumWidth, method=method)
            thisStack=result[0]
            theseImages=result[1]

    return result

''' ================ D E F I N I T I O N S ================== '''

c=1e5#in km
numberOfImages=10
dz=0.0001
emittedFrequency=1420
myFmin=100
myFmax=300
numberOfChansIm=200
numberOfChansStack=21
imChanWidth=(myFmax-myFmin)/numberOfChansIm
fullFrequencies=np.arange(myFmin, myFmax, imChanWidth)
numberOfLinesPerImage=1
myLinewidth=3
lineMeanAmp=1
panWidth=(myFmax-myFmin)/10 #in Hz
zRange=[(emittedFrequency/myFmax)-1,(emittedFrequency/myFmin)-1]
freqMaxLim=myFmax-2*panWidth
freqMinLim=myFmin+2*panWidth
zRange_pan=[(emittedFrequency/freqMaxLim)-1,(emittedFrequency/freqMinLim)-1]
myNoiseAmp=lineMeanAmp/2
myNoise=False

''' ======= I M A G E S    G E N E R A T I O N   ======= '''

print 'generating images'
allImages=[]
for i in range(numberOfImages):
    print i
    thisImageLines=[]
    for j in range(numberOfLinesPerImage):
        randomZ=np.random.uniform(zRange_pan[0],zRange_pan[1])
        if i==0:
            tempAmp=lineMeanAmp/10
        else:
            tempAmp=lineMeanAmp
        thisImageLines.append(Line( linewidth=myLinewidth,
                                    amp=tempAmp, z=randomZ,
                                    fEm=emittedFrequency))
    allImages.append(Image( fminIm=myFmin,
                            fmaxIm=myFmax,
                            numberOfChannelsIm=numberOfChansIm,
                            numberOfLines=numberOfLinesPerImage,
                            spectrum=Spectrum( fullFrequencies,
                                                lines=thisImageLines,
                                                noise=myNoise,
                                                sigmaNoise=myNoiseAmp),#spectrum=Spectrum(fullFrequencies, lines=thisImageLines),
                            lines=thisImageLines))

print 'done'
''' =========== S T A C K ============= '''
print 'starting to stack'
stacked=Stack(allImages, stackSize=numberOfChansStack)

print 'done'
''' ============= M C M C =========== '''
print 'starting to do some sweet MCMC'


sumWidth=int(numberOfChansStack/2)+1
firstStackMCMC=Stack(allImages, stackSize=sumWidth)

nRandom=100
oldSummedStack=np.mean(firstStackMCMC)
oldImages=allImages
deltaZ=0.01
method='mean'
saveResult=True

allMCMCResults=MCMC(    nRandom=nRandom,
                        oldStack=oldSummedStack,
                        oldImages=oldImages,
                        deltaZ=deltaZ,
                        sumWidth=sumWidth,
                        saveResult=saveResult)
print 'done'



import matplotlib.pyplot as plt
import math

fig=plt.figure()
ax=([fig.add_subplot(3,math.ceil(numberOfImages/3),i+1) for i in range(len(allMCMCResults[-1][1])) ])

for i in range(len(allMCMCResults[-1][1])):
    print i

for i in range(len(allMCMCResults[-1][1])):
    print i
    '''/&\ IL FAUT MODIF lines[0] par une for loop lines[n]'''

    ax[i].plot(([allMCMCResults[p][1][i].lines[0].z for p in range(len(allMCMCResults))]))

    ax[i].plot( ([ allMCMCResults[j][1][i].lines[0].observedZ for j in range(len(allMCMCResults)) ]) )

fig.show()

''' ======== B O O T S T R A P ========
stackedBoot=([0 for i in range(numberOfImages) ])

stackedBoot1=([0 for i in range(numberOfImages) ])

#newImages=([ ([0 for i in range(numberOfImages-1) ]) for j in range(numberOfImages)])
newImages=([0 for i in range(numberOfImages-1) ])
newImages1=([0 for i in range(numberOfImages+1) ])
#imageNotStacked=0
for i in range(numberOfImages):
    k=0
    k1=0
    for j in range(numberOfImages):
        if j!=i:

            newImages[k]=allImages[j]
            newImages1[k1]=allImages[j]
            k+=1
            k1+=1
        else:
            newImages1[k1]=allImages[j]
            newImages1[k1+1]=allImages[j]
            k1+=2
    stackedBoot[i]=Stack(newImages, stackSize=numberOfChansStack)
    stackedBoot1[i]=Stack(newImages1, stackSize=numberOfChansStack)


leVecteurDeLaDiff=([stackedBoot[i]-stacked for i in range(len(stackedBoot))])
leVecteurDeLaDiff1=([stackedBoot1[i]-stacked for i in range(len(stackedBoot))])

import matplotlib.pyplot as plt
import math

#fig3=plt.figure()
#ax=([fig3.add_subplot(3,math.ceil(numberOfImages/3),i+1) for i in range(len(leVecteurDeLaDiff1)) ])
toplot=([ [np.mean(leVecteurDeLaDiff[i][9:12]), np.mean(leVecteurDeLaDiff1[i][9:12])] for i in range(len(leVecteurDeLaDiff)) ])

for i in range(len(toplot)):
        ax[i].plot(toplot[i])
        ax[i].set_title(str(i))
plt.savefig('letest.pdf')
fig3.show()

for i in range(len(toplot)):
    if i==0:
        plt.plot(toplot[i], linewidth=10)
    else:
        plt.plot(toplot[i])
plt.show()
print '+1'
print np.mean(leVecteurDeLaDiff1,1)
print ''
print np.mean(leVecteurDeLaDiff1,0)
'''
