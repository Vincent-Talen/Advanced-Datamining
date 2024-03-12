# vlearning

A Python package for building and training models and neural networks.

---
## About
This package was created as part of the assignment for a bioinformatics course on
neural networks and deep learning. The way the neural networks are structured as layers
is done to follow the design specifications of the course, and the implementation is
based on the mathematical concepts taught in the course.

---
## Differences between the usage of the `vlearning` package and the `model.py` module
The main difference is that importing is different in the beginning of the notebooks and the accessing of the classes/functions available, which is explained in the first section.
But there have also been some more specific changes to the classes and functions themselves, which will be discussed in the second section.

---
### Importing and accessing
Previously when importing it was only `import model, data` but now the following code should be used:
```python
import vlearning
from vlearning import data, __version__
from vlearning import activation_functions, loss_functions, layers
print(f"Using vlearning version {__version__}")
```

Thus, normally in the notebooks, everything is accessed through the use of `model.`, but this obviously does not work anymore.
Listed below are the changes that need to be made to the notebook code to make it work with the `vlearning` package:

- Replacing the `model.` prefix with `vlearning.` for...
    - the Perceptron() class: `vlearning.Perceptron()`
    - the LinearRegression() class: `vlearning.LinearRegression()`
    - the Neuron() class: `vlearning.Neuron()`
    - the derivative() function: `vlearning.derivative()`

- Replacing the `model.` prefix with `activation_functions.` or `loss_functions.`, because these are all now available through their respective modules.
    - an activation function: `activation_functions.sign`
    - a loss function: `loss_functions.mean_squared_error`

- Replacing the `model.` prefix with `layers.` for the neural network layer classes.
    - access the layers like: `layers.Layer()`, `layers.InputLayer()`, etc.

---
### More specific changes:
To make the way the neural networks are implemented through the layer classes some changes have been made to the names of attributes.
Some of these changes are not as important, only noticeable when printing the network layers. 
Others are more significant and changed the API, requiring changes to the code when initialising some of the classes.

- **Renamed the `inputs` and `outputs` attributes to include the `num_` prefix.**  
  _This was done to avoid confusion as to what these attributes really are, not the actual list of inputs or outputs, but the number of them._


- **In the Layer class the `next` attribute was renamed to `next_layer` and all the method parameters also with the `next` name to `new_layer`.**  
  _These were renamed to avoid conflicts with the already existing name for the built-in `next()` function/keyword of Python._


- **The `layercounter` Layer class attribute was made hidden and renamed to `_counter`.**  
  _Unintended usage should be prevented this way by making it a hidden attribute and renaming it solved the bad naming practise of concatenating two words._


- **Made the `set_inputs` method hidden and thus renamed it to `_set_inputs`.**
  _It is only ever used internally by the `add()` method, setting the number of outputs of the previous layer as the number of inputs for the current layer._


- **For the DenseLayer class the `bias` class attribute was renamed to `biases`.**  
  _Indicating that the attribute is in fact, plural, and that it contains a list with multiple values._
