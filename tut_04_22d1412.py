"""=================================================== Assignment 4 ===================================================

Some instructions:
    * You can write seperate function for gradient and hessian computations.
    * You can also write any extra function as per need.
    * Use in-build functions for the computation of inverse, norm, etc. 

"""


""" Import the required libraries"""
# Start your code here
import matplotlib.pyplot as plt
import numpy as np

# End your code here

def func(x_input):
  """
  --------------------------------------------------------
  Write your logic to evaluate the function value. 

  Input parameters:
    x: input column vector (a numpy array of n dimension)

  Returns:
    y : Value of the function given in the problem at x.
    
  --------------------------------------------------------
  """
  
  # Start your code here
  #y=(x_input[0]-1)**2+(x_input[1]-1)**2- x_input[0]*x_input[1]
  #y=x_input[0]**2+x_input[1]**2+(0.5*x_input[0]+x_input[1])**2+(0.5*x_input[0]+x_input[1])**4
  #y=-0.0001*(abs(np.sin(x_input[0])*np.sin(x_input[1])*np.exp(abs(100-np.sqrt((x_input[0]**2+x_input[1]**2))/np.pi))))**0.1
  y=(x_input[0]+2*x_input[1]-7)**2+(2*x_input[0]+x_input[1]-5)**2
  return y

def gradient(func, x_input):
  """
  --------------------------------------------------------------------------------------------------
  Write your logic for gradient computation in this function. Use the code from assignment 2.

  Input parameters:  
    func : function to be evaluated
    x_input: input column vector (numpy array of n dimension)

  Returns: 
    delF : gradient as a column vector (numpy array)
  --------------------------------------------------------------------------------------------------
  """
    # Start your code here
  h=0.001
  delF=np.zeros((x_input.size,1))
  for i in range(x_input.size):
      h = 0.001
      ei=np.zeros((x_input.size,1))
      ei[i]=1
      ei = h*ei
      delF[i]=(func(x_input+ei)-func(x_input-ei))/(2*h)
  return delF
    # End your code here

def hessian(func, x_input):
  """
  --------------------------------------------------------------------------------------------------
  Write your logic for hessian computation in this function. Use the code from assignment 2.

  Input parameters:  
    func : function to be evaluated
    x_input: input column vector (numpy array)

  Returns: 
    del2F : hessian as a 2-D numpy array
  --------------------------------------------------------------------------------------------------
  """
  # Start your code here
  delx=np.repeat(0.001,x_input.size)
  del2F=np.zeros([x_input.size,x_input.size])
  for i in range(x_input.size):
      for j in range(x_input.size):
          h = 0.001
          ei=np.zeros((x_input.size,1))
          ei[i]=1
          ei = h*ei 
          ej=np.zeros((x_input.size,1))
          ej[j]=1
          ej = h*ej
          if(i==j):
              del2F[i][j]=(1/(delx[i])**2)*(func(x_input+ei)-2*func(x_input)+func(x_input-ei))
          else:
              A=func(x_input+ei+ej)
              B=func(x_input-ei-ej)
              C=func(x_input-ei+ej)
              D=func(x_input+ei-ej)
              del2F[i][j]=(1/(4*delx[i]*delx[j]))*(A+B-C-D)    
  return del2F
     # End your code here        
        
def steepest_descent(func, x_initial):
    """
    -----------------------------------------------------------------------------------------------------------------------------
    Write your logic for steepest descent using in-exact line search. 

    Input parameters:  
        func : input function to be evaluated
        x_initial: initial value of x, a column vector (numpy array)

    Returns:
        x_output : converged x value, a column vector (numpy array)
        f_output : value of f at x_output
        grad_output : value of gradient at x_output, a column vector(numpy array)
    -----------------------------------------------------------------------------------------------------------------------------
    """
    global x_itr_SD, num_itr_SD, f_itr_SD
    # Start your code here
    x_old= x_initial
    x_iterations=np.zeros((1,2))
    x_iterations[0,0]=x_initial[0]
    x_iterations[0,1]=x_initial[1]
    f_values=func(x_initial)
    rho=0.8
    c=0.1
    alpha_bar=5
    flag=False
    for i in range(2,15000):
            if (np.linalg.norm(gradient(func,x_old)))**2<1e-6 :
                flag=True
                break
            else:
                p=-gradient(func,x_old)
                alpha=alpha_bar
                while (func(x_old+alpha*p)>(func(x_old)+c*alpha*np.dot(gradient(func,x_old).T,p))) :
                  alpha=rho*alpha
                x_new = x_old + alpha*p  #p=-gradient(func,x_old)
                x_iterations=np.concatenate((x_iterations,x_new.T),axis=0)
                f_values=np.append(f_values,[func(x_new)])
                x_old=x_new
    if(flag==False):
     print("“Maximum iterations reached but convergence did not happen")
    x_output=x_new
    f_output=func(x_new)
    grad_output=gradient(func,x_new)
    x_itr_SD=x_iterations
    num_itr_SD=i-1
    f_itr_SD=f_values

    # End your code here
    
    return x_output, f_output, grad_output
    
def newton_method(func, x_initial):
  """
  -----------------------------------------------------------------------------------------------------------------------------
  Write your logic for newton method in this function. 

  Input parameters:  
    func : input function to be evaluated
    x_initial: initial value of x, a column vector (numpy array)

  Returns:
    x_output : converged x value, a column vector (numpy array)
    f_output : value of f at x_output
    grad_output : value of gradient at x_output
    num_iterations : no. of iterations taken to converge (integer)
    x_iterations : values of x at each iterations, a (num_interations x n) numpy array where, n is the dimension of x_input
    f_values : function values at each iteration (numpy array of size (num_iterations x 1))
  -----------------------------------------------------------------------------------------------------------------------------
  """
  # Write code here
  global x_itr_NM, num_itr_NM, f_itr_NM
  x_old= x_initial
  x_iterations=np.zeros((1,2))
  x_iterations[0,0]=x_initial[0]
  x_iterations[0,1]=x_initial[1]
  f_values=func(x_initial)
  rho=0.8
  c=0.1
  alpha_bar=5
  flag=False
  for i in range(2,15000):
          if (np.linalg.norm(gradient(func,x_old)))**2<1e-6 :
              flag=True
              break
          else:
              alpha=alpha_bar
              p=-np.dot(np.linalg.inv(hessian(func,x_old)),(gradient(func,x_old)))
              while (func(x_old+alpha*p)>(func(x_old)+c*alpha*np.dot(gradient(func,x_old).T,p))) :
                  alpha=rho*alpha
              x_new = x_old + alpha*p
              x_iterations=np.concatenate((x_iterations,x_new.T),axis=0)
              f_values=np.append(f_values,[func(x_new)])
              x_old=x_new
  if(flag==False):
     print("“Maximum iterations reached but convergence did not happen")
  x_output=x_new
  f_output=func(x_new)
  grad_output=gradient(func,x_new)
  x_itr_NM=x_iterations
  num_itr_NM=i-1
  f_itr_NM=f_values
  # End your code here
  return x_output, f_output, grad_output 

def quasi_newton_method(func, x_initial):
    """
    -----------------------------------------------------------------------------------------------------------------------------
    Write your logic for quasi-newton method with in-exact line search. 

    Input parameters:  
        func : input function to be evaluated
        x_initial: initial value of x, a column vector (numpy array)

    Returns:
        x_output : converged x value, a column vector (numpy array)
        f_output : value of f at x_output
        grad_output : value of gradient at x_output, a column vector (numpy array)
    -----------------------------------------------------------------------------------------------------------------------------
    """
    
    # Start your code here
    global x_itr_QN, num_itr_QN, f_itr_QN
    x_old= x_initial
    x_iterations=np.zeros((1,2))
    x_iterations[0,0]=x_initial[0]
    x_iterations[0,1]=x_initial[1]
    f_values=func(x_initial)
    I=np.eye(x_initial.size)
    C=I
    rho=0.8
    c=0.1
    alpha_bar=5
    flag=False
    for i in range(2,15000):
        if(np.linalg.norm(gradient(func,x_old))**2<1e-6):
            flag=True
            break
        else:
            alpha=alpha_bar
            p=-np.dot(C,gradient(func,x_old))
            while(func(x_old+alpha*p)>func(x_old)+c*alpha*np.dot(gradient(func,x_old).T,p)):
              alpha=alpha*rho
            x_new=x_old+alpha*p
            s=x_new-x_old
            y=gradient(func,x_new)-gradient(func,x_old)
            term1 = (I-np.dot(s,y.T)/np.dot(y.T,s))
            term2 = (I-np.dot(y,s.T)/np.dot(y.T,s))
            term3 = np.dot(s,s.T)/np.dot(y.T,s)
            C= term1.dot(C).dot(term2) + term3
            x_iterations=np.concatenate((x_iterations,x_new.T),axis=0)
            f_values=np.append(f_values,[func(x_new)])
            x_old=x_new
    if(flag==False):
      print("“Maximum iterations reached but convergence did not happen")
    x_output=x_new
    f_output=func(x_output)
    grad_output=gradient(func,x_output)
    x_itr_QN=x_iterations
    num_itr_QN=i-1
    f_itr_QN=f_values
    # End your code here
    
    return x_output, f_output, grad_output

#
#
def iterative_methods(func, x_initial):
    """
     A function to call your steepest descent, newton method and quasi-newton method.
    """
    x_SD, f_SD, grad_SD = steepest_descent(func, x_initial)
    x_NM, f_NM, grad_NM = newton_method(func, x_initial)
    x_QN, f_QN, grad_QN = quasi_newton_method(func, x_initial)

    return x_SD, f_SD, grad_SD, x_NM, f_NM, grad_NM, x_QN, f_QN, grad_QN 
    
    
    
    
"""--------------- Main code: Below code is used to test the correctness of your code ---------------

    func : function to evaluate the function value. 
    x_initial: initial value of x, a column vector, numpy array
    
"""

# Define x_initial here
x_int=np.zeros((2,1))
x_int[0]=1.5
x_int[1]=1.5
x_int=np.array([1.5, 1.5])

x_SD, f_SD, grad_SD, x_NM, f_NM, grad_NM, x_QN, f_QN, grad_QN = iterative_methods(func, x_int)
print("x_SD, f_SD, grad_SD, itr_SD :\n", x_SD, f_SD, grad_SD, num_itr_SD, "\n","x_NM, f_NM, grad_NM, itr_NM :\n", x_NM, f_NM, grad_NM, num_itr_NM,"\n", "x_QN, f_QN, grad_QN, itr_QN :\n", x_QN, f_QN, grad_QN, num_itr_QN)

plt.figure
plt.subplot(211)
plt.plot(x_itr_SD[:,0], ls='-', marker='.')
plt.plot(x_itr_NM[:,0], ls='-', marker='.')
plt.plot(x_itr_QN[:,0], ls='-', marker='.')
plt.title("X1 and X2 vs Iteration Number")
plt.ylabel("X1")
plt.xlabel("Iteration Number")
plt.legend(["Steepest Descent","Newton's Method","Quasi Newton Method"])
plt.subplot(212)
plt.plot(x_itr_SD[:,1],ls='-', marker='.')
plt.plot(x_itr_NM[:,1], ls='-', marker='.')
plt.plot(x_itr_QN[:,1], ls='-', marker='.')
plt.ylabel("X2")
plt.xlabel("Iteration Number")
plt.legend(["Steepest Descent","Newton's Method","Quasi Newton Method"])
plt.show()

plt.figure
plt.plot(f_itr_SD, ls='-', marker='.')
plt.plot(f_itr_NM, ls='-', marker='.')
plt.plot(f_itr_QN, ls='-', marker='.')
plt.title("Function vs Iteration Number")
plt.ylabel("func")
plt.xlabel("Iteration Number")
plt.legend(["Steepest Descent","Newton's Method","Quasi Newton Method"])
plt.show()

