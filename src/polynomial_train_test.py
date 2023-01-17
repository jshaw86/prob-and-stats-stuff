import numpy
import matplotlib.pyplot as plt

numpy.random.seed(2)

pageSpeeds = numpy.random.normal(3.0,1.0, 100)
purchaseAmount = numpy.random.normal(50.0,30.0,100) / pageSpeeds


trainX = pageSpeeds[:80]
testX = pageSpeeds[80:]

trainY = purchaseAmount[:80]
testY = purchaseAmount[80:]

x = numpy.array(trainX)
y = numpy.array(trainY)

p = numpy.poly1d(numpy.polyfit(x,y,2))

xp = numpy.linspace(0, 7, 100)
axes = plt.axes()
plt.scatter(x, y)
plt.plot(xp, p(xp))
plt.show()

testx = numpy.array(testX)
testy = numpy.array(testY)

p = numpy.poly1d(numpy.polyfit(x,y,2))

xp = numpy.linspace(0, 7, 100)
axes = plt.axes()
plt.scatter(testx, testy)
plt.plot(xp, p(xp))
plt.show()


