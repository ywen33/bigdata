#Yilin Wen Homework 1
#Part A
pagination = 'Page 1 of 12'
index12=pagination.index('12')
newVariable=int(pagination[index12:])
print(newVariable)

#Part B
runner1 = 300.25 #Finish time in Minutes
runner2 = 260.75
runner3 = 315.75
runner4 = 245.25
average = (runner1+runner2+runner3+runner4)/4
print('Average:', average)
variance = ((runner1-average)**2+(runner2-average)**2+(runner3-average)**2+(runner4-average)**2)/4
print('Variance:', variance)

#Part C
P=input('Please enter principle:\n')
r=input('Please enter annual interest rate (example 5.2 for 5.2%):\n')
t=input('Please enter the term in years:\n')
n=input('Please enter number of times the interest will compound per year:\n')
A=float(P)*(1+float(r)/100/float(n))**(float(t)*float(n))
print('In '+t+' years, at the interest rate of '+r+'% compounded '+n+' times per year, the initial amount of'+'${0:,.2f}'.format(float(P))+' will be worth '+'${0:,.2f}'.format(A)+'. '+'${0:,.2f}'.format(A-float(P))+' will be paid in interest.')