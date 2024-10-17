# Perceptron
In this project, we will learn the perceptron neurons' function in data classification.

## 1. **Introduction**

Neural networks, inspired by the functioning of the human brain, have been a subject of interest since their inception. Unlike traditional digital computers, which are based on linear, sequential processing, the brain operates as a nonlinear, parallel information-processing system. With the ability to organize its structural components—neurons—it can perform tasks such as pattern recognition, perception, and motor control at speeds unmatched by even the fastest computers today.

For instance, the human visual system is an incredible information-processing tool that creates a detailed representation of our environment. This allows us to recognize objects, including familiar faces in complex scenes, in mere milliseconds. Despite the complexity of such tasks, digital systems still struggle to achieve comparable efficiency.

Another example is the bat’s echolocation system, a highly evolved form of sonar. Through its neural system, the bat can extract detailed information about a target’s distance, velocity, size, and position, with astonishing speed and precision. The bat's brain, despite its small size, can process this information faster and more effectively than even the most advanced radar systems.

This kind of complex neural computation underscores the remarkable plasticity of biological brains. At birth, the human brain is already structured but continues to develop as it experiences its environment. This adaptability, referred to as "plasticity," is fundamental not only to biological brains but also to artificial neural networks. These artificial networks aim to replicate the brain’s ability to learn and adapt through experiential knowledge.

Neural networks, like their biological counterparts, consist of interconnected neurons that work together in parallel to process information. By adjusting the synaptic weights—connections between neurons—artificial networks learn from their environment. This learning is facilitated by algorithms that modify these weights in an orderly fashion to meet design objectives. The potential for neural networks to modify their topology, similar to how biological neurons form new connections, offers further avenues for enhancing their capabilities.

## 2. **Biological Neural Networks**

Artificial neural networks (ANNs) are inspired by biological neural networks, though the extent of this inspiration varies. For some researchers, the biological accuracy of the model is essential, while for others, the focus is more on the network’s ability to perform computational tasks.

A biological neuron has three key components: dendrites, the soma, and the axon. Dendrites receive electrical impulses from other neurons through synapses, where signals are modified before being passed on. The soma processes these incoming signals, and if the combined input exceeds a threshold, the neuron "fires," transmitting the signal down the axon to other neurons. This process is akin to the functioning of artificial neurons, where input signals are weighted, summed, and passed through an activation function that determines the neuron’s output.

Biological neurons also possess the remarkable capability of fault tolerance. Even when neurons are lost due to damage or aging, the brain can adapt by forming new connections, ensuring continued functionality. Similarly, ANNs can be designed to handle minor damage, such as lost data or connections, without significant loss of performance. This robustness makes ANNs suitable for complex real-world tasks where input data may be noisy or incomplete.

![Neuron](https://github.com/user-attachments/assets/6d73775b-d4bf-4ad9-9593-898fd1b898ed)

## 3. **Neuron Model**

In an ANN, the neuron is the fundamental computational unit. The mathematical model of a neuron is described by:

1. **Weighted Input Summation**: Each input \( x_j \) is multiplied by its corresponding synaptic weight \( w_{kj} \). The sum of all weighted inputs, along with the bias \( b_k \), forms the net input \( u_k \):

![image](https://github.com/user-attachments/assets/1a5e1313-8dc3-40a9-a506-b6f2ad9b14d4)

2. **Activation Function**: The net input is passed through an activation function to produce the output \( y_k \):

![image](https://github.com/user-attachments/assets/9b020406-2e66-43b4-9e5b-fadf8965a693)

![Perceptron](https://user-images.githubusercontent.com/118474020/202779325-65d108e4-6e10-49c7-9ec8-9c98bcff53c5.png)

   For example, in a perceptron, the activation function is typically a step function or a sign function, which produces binary outputs (either -1 or 1).

   In artificial neural networks, the activation function plays a crucial role in introducing non-linearity into the model, enabling the network to solve complex problems. A variety of activation functions are used, depending on the desired behavior of the neuron. Here are several common activation functions:

1. **Sigmoid Activation Function**:
   - The **sigmoid** function is continuous and smoothly transitions from 0 to 1. It is often used when the network is required to output probabilities.
   
![image](https://github.com/user-attachments/assets/6d343a10-fe6e-412e-89cd-44e0d57f234e)

     
     The output ranges between 0 and 1, making it useful in scenarios where binary-like outputs are needed but in a continuous form.

2. **Hyperbolic Tangent (Tanh) Activation Function**:
   - The **tanh** function is similar to the sigmoid function but has an output range between -1 and 1. It provides stronger gradients for optimization and allows negative output values.
   
     ![image](https://github.com/user-attachments/assets/98a8aea0-5cc6-4836-ab14-02ea9862bf8a)


   This function is used when the network needs to differentiate between positive and negative inputs or outputs.

3. **Rectified Linear Unit (ReLU) Activation Function**:
   - The **ReLU** function is widely used in deep learning because it mitigates the vanishing gradient problem and allows networks to converge faster. It outputs the input directly if it is positive and zero otherwise.

    ![image](https://github.com/user-attachments/assets/a3d40b50-be69-412a-89b5-7f7f0f6a314b)

   
     ReLU is not bounded, meaning its output can go from 0 to infinity, making it ideal for models that require large positive output ranges.

4. **Leaky ReLU Activation Function**:
   - To address the "dying ReLU" problem where neurons may stop activating entirely, **Leaky ReLU** introduces a small slope for negative values.

   ![image](https://github.com/user-attachments/assets/a8acf132-5c4d-4c44-b3fd-b1a0bec2f4d4)
   
     Where \( \alpha \) is a small positive constant (e.g., 0.01).

5. **Softmax Activation Function**:
   - **Softmax** is typically used in the output layer of classification networks where multiple classes are predicted. It converts a vector of values into probabilities, making it useful in multi-class classification.

   ![image](https://github.com/user-attachments/assets/7e61c585-a360-4d27-9b19-9c6d1ed981e2)
   
     The output of the softmax function is a probability distribution over multiple classes, where the sum of all outputs equals 1.

6. **Swish Activation Function**:
   - **Swish** is a newer activation function that has been found to perform better than ReLU in some networks. It is defined as a combination of the input and a sigmoid function:

   ![image](https://github.com/user-attachments/assets/4698d67a-6dee-42d1-b78c-63d13e15bc29)
   
     This smooth, non-monotonic function can enhance the performance of deep networks.


## 4. **Perceptron Learning Algorithm**

The perceptron is the simplest form of a neural network that classifies linearly separable patterns. The perceptron learning algorithm updates the synaptic weights iteratively based on the difference between the actual output \( y \) and the desired output \( d \).

The perceptron model can be mathematically expressed as:

1. **Net Input**: The net input to the perceptron is the weighted sum of inputs plus a bias term:

![image](https://github.com/user-attachments/assets/8ab0fc13-bd20-4397-a086-25dcf89a0aa1)


2. **Output**: The output of the perceptron is a binary value determined by a step function (sign function):

![image](https://github.com/user-attachments/assets/38511a1f-28f7-4161-bbc3-a6b74cea4d17)

   Where:

![image](https://github.com/user-attachments/assets/c4bf2056-e8ce-4795-8462-8fca6f57c0e8)

3. **Weight Update Rule**: The synaptic weights are updated using the following rule:

![image](https://github.com/user-attachments/assets/44943ae0-359e-4491-9dff-c3eed2d8d9c1)

   Where \( t \) is the iteration number, and \( \eta \) is the learning rate.

4. **Bias Update Rule**: The bias term is also updated similarly:

![image](https://github.com/user-attachments/assets/bcf5a0a4-7816-4969-994f-0598f3cb0f6b)

This algorithm ensures that the perceptron converges if the training data is linearly separable.

### 5. **Perceptron Convergence**

The perceptron convergence theorem states that if the training data is linearly separable, the perceptron algorithm will find a set of weights that correctly classify all the training examples. This means that after a finite number of iterations, the decision boundary will be positioned to separate the two classes of data. The decision boundary for a perceptron is a hyperplane defined by:

![image](https://github.com/user-attachments/assets/4f7574f8-fc78-43c0-8b30-593d8e6f95c3)

This hyperplane separates the input space into two regions, one for each class.

In summary, the perceptron learning algorithm adjusts the synaptic weights iteratively to reduce classification errors. It forms the foundation for more complex neural networks, and its behavior can be mathematically described through the update rules for weights and biases.
