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
        freqMidLineIndex=np.zeros(len(thisImageLines))
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
numberOfImages=100
dz=0
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

''' ============== '''


allImages=[]
for i in range(numberOfImages):
    print i
    thisImageLines=[]
    for j in range(numberOfLinesPerImage):
        randomZ=np.random.uniform(zRange_pan[0],zRange_pan[1])
        thisImageLines.append(Line( linewidth=myLinewidth,
                                    amp=lineMeanAmp, z=randomZ,
                                    fEm=emittedFrequency))
    allImages.append(Image( fminIm=myFmin,
                            fmaxIm=myFmax,
                            numberOfChannelsIm=numberOfChansIm,
                            numberOfLines=numberOfLinesPerImage,
                            spectrum=Spectrum( fullFrequencies,
                                                lines=thisImageLines,
                                                noise=True,
                                                sigmaNoise=lineMeanAmp),#spectrum=Spectrum(fullFrequencies, lines=thisImageLines),
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
