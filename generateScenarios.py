import numpy as np
import matplotlib.pyplot as plt


def getSinFunction(t, offset, period, scale):

    p = scale * np.sin(2.0*np.pi*(t+offset)/period)
    return p


def getAbsSinFunction(t, offset, period, scale):
    return np.absolute(getSinFunction(t, offset, 2.0*period, scale))


def getLinearEvolution(t, start, growth):
    return start + t * growth


def makeScenario1(t, scale, T, offset):

    p1 = getAbsSinFunction(t, offset, T, scale)
    #This is the signal
    signal = (p1).astype(int)
    #This is the random contribution
    err = np.random.poisson(np.abs(signal))
    return err


def makeScenario2(t, scale, T, offset, scaleS, TS, offsetS):


    p1 = getAbsSinFunction(t, offset, T, scale)
    #This is the seasonal signal
    p2 = getSinFunction(t, offsetS, TS, scaleS)
    #This is the signal
    signal = (p1+p2).astype(int)
    #This is the random contribution
    err = np.random.poisson(np.abs(signal))
    return err


def makeScenario3(t, scale, T, offset, scaleS, TS, offsetS, growthRate):

    p1 = getAbsSinFunction(t, offset, T, scale)
    #This is the seasonal signal
    p2 = getSinFunction(t, offsetS, TS, scaleS)
    #This is the function according to the company evolution
    p3 = getLinearEvolution(t, 0.0, growthRate)   
    #This is the signal
    signal = (p1+p2+p3).astype(int)
    #This is the random contribution
    err = np.random.poisson(np.abs(signal))
    return err


if __name__ == "__main__":

    #This is the number of years to be simulated
    years = 5
    t = np.asarray(range(0, years * 12))

    #Let's make scenario number 1
    #This scenario includes a periodic stock behavior + stocastic term given by a poisson distribution
    #This assumption assumes that there is a periodic component in the production and need of stock 
    #The parameters to provide are:
    #scale: Scale of the variation
    #T: period in months
    #offset: The offset in which measurements are taken
    # data1 = makeScenario1(t=t, scale=1000.0, T=12.0, offset=0.3)
    data1 = makeScenario1(t=t, scale=1000.0, T=12.0, offset=0.3)
    # print(data1)

    #Let's make scenario number 2
    #This scenario includes a periodic stock behavior + seasonal behaviour + stocastic term given by a poisson distribution
    #This assumption assumes that there is a periodic component in the production and need of stock, plus a seasonal variation 
    #The parameters to provide are:
    #scale: Scale of the periodic variation
    #T: period in months of the periodic variation
    #offset: The offset in which measurements are taken
    #scaleS: Scale of the seasonal variation
    #TS: period in months of the seasonal variation
    #offsetS: The offset of the seasonal variation
    data2 = makeScenario2(t=t, scale=1000.0, T=12.0, offset=1, scaleS=200.0, TS = 6.0, offsetS=0.5)
    # data2 = makeScenario2(t=t, scale=1000.0, T=6.0, offset=1, scaleS=1000.0, TS = 4.0, offsetS=0.8)


    #Let's make scenario number 3
    #This scenario includes a periodic stock behavior + seasonal behaviour + yearly evolution + stocastic term given by a poisson distribution
    #This assumption assumes that there is a periodic component in the production and need of stock, plus a seasonal variation, plus a variation in the stock of the company 
    #The parameters to provide are:
    #scale: Scale of the periodic variation
    #T: period in months of the periodic variation
    #offset: The offset in which measurements are taken
    #scaleS: Scale of the seasonal variation
    #TS: period in months of the seasonal variation
    #offsetS: The offset of the seasonal variation
    #growthRate: rate of growth of the company
    # data3 = makeScenario3(t=t, scale=1000.0, T=6.0, offset=1, scaleS=200.0, TS=7.0, offsetS=0.5, growthRate=-15)
    data3 = makeScenario3(t=t, scale=1000.0, T=6.0, offset=1, scaleS=300.0, TS=8.0, offsetS=1.5, growthRate=15)

    fig = plt.figure(figsize=(15,10))
    plt.subplot(3, 1, 1)
    plt.plot(t, data1)
    plt.subplot(3, 1, 2)
    plt.plot(t, data2)
    plt.subplot(3, 1, 3)
    plt.plot(t, data3)
    
    plt.show()

# main()