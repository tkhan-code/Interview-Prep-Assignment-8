cat > data/corpus/deep_learning_basics.md << 'EOF'
# Deep Learning Basics

## What is Deep Learning?
Deep learning is a subset of machine learning that uses neural networks with multiple layers (deep neural networks) to learn hierarchical representations of data. These networks can automatically discover patterns and features without manual feature engineering.

## Backpropagation Explained
Backpropagation is the core algorithm that makes deep learning work. It calculates how much each weight in the network contributes to the overall error. The algorithm works in two phases:
1. Forward pass: Input data flows through the network to produce a prediction
2. Backward pass: The error is propagated backwards, and gradients are computed using the chain rule
3. Weight update: Network weights are adjusted to minimize the error

## Activation Functions
Activation functions introduce non-linearity, allowing neural networks to learn complex patterns:
- ReLU (Rectified Linear Unit): f(x) = max(0, x) - Fast and prevents vanishing gradients
- Sigmoid: f(x) = 1/(1+e^-x) - Good for binary classification outputs
- Tanh: f(x) = (e^x - e^-x)/(e^x + e^-x) - Zero-centered, better than sigmoid for hidden layers
- Softmax: Used for multi-class classification outputs

## Gradient Descent Optimization
Gradient descent is the optimization algorithm used to train neural networks. It works by:
- Computing the gradient of the loss function
- Taking steps in the direction of steepest descent
- The learning rate controls the step size
- Common variants include SGD with momentum, Adam, and RMSprop
EOF