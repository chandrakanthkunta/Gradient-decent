import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# Use classic theme
plt.style.use('classic')

# Function to compute gradient and gradient descent steps
def gradient_descent(function, derivative, start, learning_rate, iterations):
    x_values = [start]
    for i in range(iterations):
        start = start - learning_rate * derivative(start)
        x_values.append(start)
    return x_values

# Function definitions and their derivatives
def function_x_squared(x):
    return x ** 2

def derivative_x_squared(x):
    return 2 * x

def function_sin(x):
    return np.sin(x)

def derivative_sin(x):
    return np.cos(x)

def function_x(x):
    return x

def derivative_x(x):
    return 1

def function_x_cubed(x):
    return x ** 3

def derivative_x_cubed(x):
    return 3 * x ** 2

def function_exp(x):
    return np.exp(x)

def derivative_exp(x):
    return np.exp(x)

def function_abs(x):
    return np.abs(x)

def derivative_abs(x):
    return np.where(x > 0, 1, -1)

# Streamlit app
st.title("Gradient Descent Visualizer")

# Sidebar options
st.sidebar.header("Select a function to plot")
functions = {
    "x^2": (function_x_squared, derivative_x_squared),
    "sin(x)": (function_sin, derivative_sin),
    "x": (function_x, derivative_x),
    "x^3": (function_x_cubed, derivative_x_cubed),
    "e^x": (function_exp, derivative_exp),
    "|x|": (function_abs, derivative_abs),
}

selected_function_name = st.sidebar.selectbox("Function", list(functions.keys()))
selected_function, selected_derivative = functions[selected_function_name]

start = st.sidebar.slider("Select Starting Point", -10.0, 10.0, 5.0)
learning_rate = st.sidebar.slider("Select Learning Rate", 0.01, 1.0, 0.1)
iterations = st.sidebar.slider("Select Number of Iterations", 1, 50, 10)

# Compute gradient descent steps
x_values = gradient_descent(selected_function, selected_derivative, start, learning_rate, iterations)
y_values = [selected_function(x) for x in x_values]

# Plotting with enhanced styles
fig, ax = plt.subplots()
x_plot = np.linspace(-10, 10, 400)
y_plot = selected_function(x_plot)

# Enhanced plot styles
ax.plot(x_plot, y_plot, label='f(x)', color='navy', linestyle='-', linewidth=2)
ax.plot(x_values, y_values, marker='o', linestyle='--', color='red', label='Gradient Descent', markersize=8)

# Plotting tangents
for i, (x, y) in enumerate(zip(x_values, y_values)):
    slope = selected_derivative(x)
    tangent_line = slope * (x_plot - x) + y
    ax.plot(x_plot, tangent_line, linestyle='-.', color='gray', alpha=0.5)

# Enhance plot aesthetics
ax.set_xlabel('x', fontsize=12)
ax.set_ylabel('f(x)', fontsize=12)
ax.set_title(f'Gradient Descent with Learning Rate {learning_rate}', fontsize=14)
ax.legend()
ax.grid(True)

# Customize the spines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_position(('outward', 5))
ax.spines['bottom'].set_position(('outward', 5))

st.pyplot(fig)

# Display values
st.write("### Gradient Descent Steps")
st.write(f"Function: {selected_function_name}")
st.write(f"Starting Point: {start}")
st.write(f"Learning Rate: {learning_rate}")
st.write(f"Number of Iterations: {iterations}")
st.write(f"Final Point: {x_values[-1]}")
st.write(f"Function value at Final Point: {selected_function(x_values[-1])}")
