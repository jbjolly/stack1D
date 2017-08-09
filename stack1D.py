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

''' ================ D E F I N I T I O N S ================== '''

c=1e5#in km
numberOfImages=9
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
myNoise=True
''' ======= I M A G E S    G E N E R A T I O N   ======= '''


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


''' =========== S T A C K ============= '''

stacked=Stack(allImages, stackSize=numberOfChansStack)




import matplotlib.pyplot as plt
fig=plt.figure()
ax1=fig.add_subplot(2,1,1)
ax2=fig.add_subplot(2,1,2)
for i in range(numberOfImages):
    ax1.plot(allImages[i].spectrum)

ax2.plot(stacked)
fig.show()


''' ======== B O O T S T R A P ======== '''
stackedBoot=([0 for i in range(numberOfImages+1) ])
stackedBoot[0]=stacked

stackedBoot1=([0 for i in range(numberOfImages+1) ])
stackedBoot1[0]=stacked

#newImages=([ ([0 for i in range(numberOfImages-1) ]) for j in range(numberOfImages)])
newImages=([0 for i in range(numberOfImages-1) ])
newImages1=([0 for i in range(numberOfImages+1) ])
#imageNotStacked=0
for i in range(numberOfImages):
    k=0
    for j in range(numberOfImages):
        if j!=i:

            newImages[k]=allImages[j]
            newImages1[k]=allImages[j]
            k+=1
        else:
            newImages1[-1]=allImages[j]
            newImages1[-2]=allImages[j]
    stackedBoot[i+1]=Stack(newImages, stackSize=numberOfChansStack)
    stackedBoot1[i+1]=Stack(newImages1, stackSize=numberOfChansStack)

fig1=plt.figure()
ax=fig1.add_subplot(1,1,1)
for i in range(numberOfImages+1):
    if i==0:
        ax.plot(stackedBoot[i],'ro-',linewidth=15)
    else:
        ax.plot(stackedBoot[i])
fig1.show()


leVecteurDeLaDiff=([stackedBoot[i]-stackedBoot[0] for i in range(1,len(stackedBoot))])
leVecteurDeLaDiff1=([stackedBoot1[i]-stackedBoot1[0] for i in range(1,len(stackedBoot))])

import math
fig2=plt.figure()
ax=([fig2.add_subplot(3,math.ceil(numberOfImages/3),i) for i in range(len(leVecteurDeLaDiff)-1) ])

for i in range(len(leVecteurDeLaDiff)):
    if i!=0:
        ax[i-1].plot(leVecteurDeLaDiff[i])
fig2.show()

print np.mean(leVecteurDeLaDiff,1)
print ''
print np.mean(leVecteurDeLaDiff,0)



fig3=plt.figure()
ax=([fig3.add_subplot(3,math.ceil(numberOfImages/3),i) for i in range(len(leVecteurDeLaDiff1)-1) ])

for i in range(len(leVecteurDeLaDiff1)):
    if i!=0:
        ax[i-1].plot(leVecteurDeLaDiff1[i])
fig3.show()
print '+1'
print np.mean(leVecteurDeLaDiff1,1)
print ''
print np.mean(leVecteurDeLaDiff1,0)
