"""============================================ Assignment 3: Newton Method ============================================"""
""" Import the required libraries"""
# Start you code here
import numpy as np
import matplotlib.pyplot as plt
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
  x_old= x_initial
  x_iterations=np.zeros((1,2))
  x_iterations[0,0]=x_initial[0]
  x_iterations[0,1]=x_initial[1]
  f_values=func(x_initial)
  flag=False
  for i in range(2,15000):
          if (np.linalg.norm(gradient(func,x_old)))**2<1e-6 :
              flag=True
              break
          else:
              x_new = x_old - np.dot(np.linalg.inv(hessian(func,x_old)),(gradient(func,x_old)))
              x_iterations=np.concatenate((x_iterations,x_new.T),axis=0)
              f_values=np.append(f_values,[func(x_new)])
              x_old=x_new
  if(flag==False):
     print("“Maximum iterations reached but convergence did not happen")
  x_output=x_new
  f_output=func(x_new)
  grad_output=gradient(func,x_new)
  num_iterations=i-1
  # End your code here
  return x_output, f_output, grad_output, num_iterations, x_iterations, f_values  

def plot_x_iterations(NM_iter, NM_x):
  """
  -----------------------------------------------------------------------------------------------------------------------------
  Write your logic for plotting x_input versus iteration number i.e,
  x1 with iteration number and x2 with iteration number in same figure but as separate subplots. 

  Input parameters:  
    NM_iter : no. of iterations taken to converge (integer)
    NM_x: values of x at each iterations, a (num_interations X n) numpy array where, n is the dimension of x_input

  Output the plot.
  -----------------------------------------------------------------------------------------------------------------------------
  """
  # Start your code here
  plt.figure
  plt.subplot(121)
  plt.plot(range(1,NM_iter+1),NM_x[:,0])
  plt.xlabel("iterations")
  plt.xticks(range(1,NM_iter+1))
  plt.ylabel("x1")
  plt.title("x1 vs iteration number")
  plt.subplot(122)
  plt.plot(range(1,NM_iter+1),NM_x[:,1])
  plt.xlabel("iterations")
  plt.xticks(range(1,NM_iter+1))
  plt.ylabel("x2")
  plt.title("x2 vs iteration number")
  plt.show()
  # End your code here

def plot_func_iterations(NM_iter, NM_f):
  """
  ------------------------------------------------------------------------------------------------
  Write your logic to generate a plot which shows the value of f(x) versus iteration number.

  Input parameters:  
    NM_iter : no. of iterations taken to converge (integer)
    NM_f: function values at each iteration (numpy array of size (num_iterations x 1))

  Output the plot.
  -------------------------------------------------------------------------------------------------
  """
  # Start your code here
  plt.figure
  plt.plot(range(1,NM_iter+1),NM_f)
  plt.xlabel("iterations")
  plt.xticks(range(1,NM_iter+1))
  plt.ylabel("f")
  plt.title("f vs iteration number")
  plt.show()

  # End your code here




"""--------------- Main code: Below code is used to test the correctness of your code ---------------"""

x_initial = np.array([[1.5, 1.5]]).T

x_output, f_output, grad_output, num_iterations, x_iterations, f_values = newton_method(func, x_initial)

print("\nFunction converged at x = \n",x_output)
print("\nFunction value at converged point = \n",f_output)
print("\nGradient value at converged point = \n",grad_output)

plot_x_iterations(num_iterations, x_iterations)

plot_func_iterations(num_iterations, f_values)