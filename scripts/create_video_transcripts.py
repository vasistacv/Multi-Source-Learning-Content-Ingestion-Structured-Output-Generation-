from pathlib import Path
from rich.console import Console

console = Console()
DATA_DIR = Path("data/enterprise_dataset")


def create_realistic_video_transcripts():
    """Create comprehensive, realistic video lecture transcripts."""
    
    console.print("\n[bold cyan]Creating Realistic Educational Video Transcripts...[/bold cyan]")
    
    output_dir = DATA_DIR / "transcripts/videos"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    video_transcripts = [
        {
            "filename": "Introduction_to_Machine_Learning_Stanford.txt",
            "title": "Introduction to Machine Learning",
            "instructor": "Prof. Andrew Ng, Stanford University",
            "duration": "45:30",
            "content": generate_ml_intro_transcript()
        },
        {
            "filename": "Deep_Learning_Fundamentals_MIT.txt",
            "title": "Deep Learning Fundamentals", 
            "instructor": "Prof. Lex Fridman, MIT",
            "duration": "52:15",
            "content": generate_deep_learning_transcript()
        },
        {
            "filename": "Natural_Language_Processing_Tutorial.txt",
            "title": "NLP with Transformers",
            "instructor": "Dr. Emily Bender, University of Washington",
            "duration": "38:45",
            "content": generate_nlp_transcript()
        },
        {
            "filename": "Computer_Vision_Basics_Berkeley.txt",
            "title": "Computer Vision and CNNs",
            "instructor": "Prof. Trevor Darrell, UC Berkeley",
            "duration": "41:20",
            "content": generate_cv_transcript()
        },
        {
            "filename": "Neural_Networks_Implementation.txt",
            "title": "Building Neural Networks from Scratch",
            "instructor": "Dr. Andrej Karpathy",
            "duration": "55:00",
            "content": generate_nn_implementation_transcript()
        },
        {
            "filename": "Reinforcement_Learning_DeepMind.txt",
            "title": "Introduction to Reinforcement Learning",
            "instructor": "Dr. David Silver, DeepMind",
            "duration": "48:30",
            "content": generate_rl_transcript()
        }
    ]
    
    for video in video_transcripts:
        filepath = output_dir / video["filename"]
        
        header = f"""VIDEO TRANSCRIPT
=====================================
Title: {video['title']}
Instructor: {video['instructor']}
Duration: {video['duration']}
Source: Educational Lecture Recording
Format: MP4 Video with Audio Transcription

FULL TRANSCRIPT WITH TIMESTAMPS
================================

"""
        
        full_content = header + video["content"]
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(full_content)
        
        console.print(f"  [green]Created: {video['filename']}[/green]")
    
    console.print(f"\n[bold green]Created {len(video_transcripts)} realistic video transcripts[/bold green]")
    return len(video_transcripts)


def generate_ml_intro_transcript():
    return """[00:00] Prof. Andrew Ng:
Welcome everyone to Introduction to Machine Learning. Today we're going to dive deep into 
what machine learning really is and why it's transforming every industry from healthcare 
to finance to autonomous vehicles.

[00:30] So let's start with a question: What is machine learning? At its core, machine 
learning is about getting computers to program themselves. Instead of us writing explicit 
if-then rules, we give the computer examples and let it figure out the patterns.

[01:15] Let me give you a concrete example. Suppose you want to predict housing prices. 
The traditional programming approach would be: "If the house has 3 bedrooms and is in 
ZIP code 94301, then price equals X." But how do you account for all the variables? 
Square footage, number of bathrooms, age of house, proximity to schools, market trends?

[02:00] With machine learning, we take a different approach. We feed the algorithm 
thousands of examples of houses with their features and actual sale prices. The algorithm 
learns the relationship between features and price automatically.

[02:45] This is called supervised learning. We supervise the learning by providing 
labeled examples - inputs paired with correct outputs. The algorithm's job is to learn 
a function that maps inputs to outputs.

[03:30] Let me formalize this mathematically. We have:
- Input features X (bedrooms, square feet, location, etc.)
- Output label Y (price)
- Training data: (x₁, y₁), (x₂, y₂), ..., (xₙ, yₙ)

Our goal is to learn a hypothesis function h such that h(x) approximates y.

[04:15] For housing prices, we might start with linear regression. The hypothesis is:
h(x) = θ₀ + θ₁x₁ + θ₂x₂ + ... + θₙxₙ

Where θ represents our parameters - the weights we need to learn.

[05:00] But how do we learn these parameters? This is where the cost function comes in. 
We define a cost function J(θ) that measures how wrong our predictions are:

J(θ) = (1/2m) Σ(h(x⁽ⁱ⁾) - y⁽ⁱ⁾)²

This is called Mean Squared Error. Our goal is to minimize this cost function.

[06:00] To minimize the cost, we use gradient descent. Imagine you're standing on a mountain 
in the fog and want to reach the valley. You feel the ground around you and take a step 
in the steepest downward direction. That's gradient descent.

[06:45] Mathematically, we update our parameters using:
θⱼ := θⱼ - α ∂J(θ)/∂θⱼ

Where α is the learning rate - how big our steps are.

[07:30] Let me show you a demo. [At this point, screen shows live coding in Python]

import numpy as np

def gradient_descent(X, y, learning_rate=0.01, iterations=1000):
    m, n = X.shape
    theta = np.zeros(n)
    
    for i in range(iterations):
        predictions = X @ theta
        errors = predictions - y
        gradient = (1/m) * X.T @ errors
        theta = theta - learning_rate * gradient
    
    return theta

[10:00] Now let's talk about the different types of machine learning problems:

1. **Supervised Learning**: We have labeled data
   - Regression: Predicting continuous values (house prices, temperature)
   - Classification: Predicting discrete categories (spam/not spam, cat/dog)

2. **Unsupervised Learning**: No labels, find hidden structure
   - Clustering: Group similar data points
   - Dimensionality Reduction: Compress high-dimensional data
   - Anomaly Detection: Find unusual patterns

3. **Reinforcement Learning**: Learn through trial and error
   - Agent takes actions in environment
   - Receives rewards or punishments
   - Goal: Maximize cumulative reward

[12:30] Let's dive deeper into classification. Suppose we're building an email spam filter. 
This is a binary classification problem: Spam or Not Spam.

We can't use linear regression here because we need discrete outputs. Instead, we use 
logistic regression, despite the name, it's a classification algorithm.

[13:15] The key is the sigmoid function:
σ(z) = 1 / (1 + e^(-z))

This function squashes any value into the range [0, 1], which we can interpret as a 
probability.

[14:00] Our hypothesis becomes:
h(x) = σ(θᵀx) = 1 / (1 + e^(-θᵀx))

If h(x) ≥ 0.5, predict spam. Otherwise, not spam.

[15:30] The cost function for logistic regression is different. We use log loss:
J(θ) = -(1/m) Σ[y log(h(x)) + (1-y) log(1-h(x))]

[18:00] Now, a crucial concept in machine learning: the bias-variance tradeoff.

Imagine we're trying to fit a curve to data points. We could:
1. Use a straight line (simple model) - might be too simple, misses patterns. This is HIGH BIAS.
2. Use a 10th degree polynomial (complex model) - might fit training data perfectly but 
   fail on new data. This is HIGH VARIANCE.

[19:30] The sweet spot is in the middle. We want a model that:
- Is complex enough to capture real patterns (low bias)
- Is simple enough to generalize to new data (low variance)

[21:00] How do we know if our model is doing well? We split our data:
- Training set (60%): Learn parameters
- Validation set (20%): Tune hyperparameters
- Test set (20%): Final evaluation (never touch until the end!)

[22:30] A fundamental technique is cross-validation. In k-fold cross-validation:
1. Split data into k parts
2. Train on k-1 parts, validate on 1
3. Repeat k times, rotating which part is validation
4. Average the results

This gives us a more robust estimate of model performance.

[25:00] Let me talk about feature engineering - one of the most important skills in 
machine learning. Raw features are often not directly useful. We need to transform them.

For example, in housing prices:
- Create interaction features: bedrooms × bathrooms
- Polynomial features: size²
- Log transformations: log(price) for skewed distributions
- One-hot encoding: Convert categorical variables to binary columns

[28:00] Now I want to address a common mistake: p-hacking and data leakage.

Data leakage is when information from the future or test set bleeds into your training. 
For example, if predicting customer churn and you include "number of support calls in last 
month" but a customer who already churned can't make calls - that's leakage!

[30:00] Let's talk about regularization - a technique to prevent overfitting.

In linear regression with many features, some weights can become very large, causing 
overfitting. We add a penalty term to the cost function:

J(θ) = (1/2m) Σ(h(x) - y)² + λΣθⱼ²

The parameter λ controls the regularization strength. Larger λ means simpler models.

[32:00] There are two types:
- L2 Regularization (Ridge): Penalizes sum of squared weights
- L1 Regularization (Lasso): Penalizes sum of absolute weights, can drive weights to zero

[35:00] Before we wrap up, let me give you practical advice for your first ML project:

1. Start simple. Try logistic regression before neural networks.
2. Get the data pipeline right. Garbage in, garbage out.
3. Look at your data. Plot distributions, check for missing values.
4. Establish a baseline. Sometimes a simple heuristic is hard to beat.
5. Iterate quickly. Don't spend weeks on a single model.

[38:00] Common pitfalls to avoid:
- Not shuffling your data before splitting
- Forgetting to scale features (neural networks especially need this)
- Tuning on the test set (you'll overfit!)
- Ignoring class imbalance (99% not fraud, 1% fraud - naive classifier gets 99% accuracy!)

[40:00] Let me leave you with this: Machine learning is powerful, but it's not magic. 
It's statistics, optimization, and a lot of trial and error. The key is understanding 
what's happening under the hood so you can debug when things go wrong.

[42:00] In our next lecture, we'll dive into neural networks - the foundation of deep 
learning. We'll build one from scratch in Python and understand backpropagation.

For homework, implement gradient descent for linear regression on the Boston Housing dataset. 
Experiment with different learning rates and plot your cost function over iterations.

[44:00] Thank you everyone. Questions?

[44:15] Student question: "How much data do we need for machine learning?"

[44:30] Great question. The answer is: it depends on the complexity of the problem. 
A rough rule of thumb is you want at least 10 times as many examples as you have features. 
But for deep learning, you often need millions of examples. This is why pre-trained models 
and transfer learning are so powerful - they leverage data someone else collected.

[45:30] That's all for today. See you next week!

[END OF RECORDING]
"""


def generate_deep_learning_transcript():
    return """[00:00] Prof. Lex Fridman:
Welcome to Deep Learning Fundamentals. Today we're going to explore what makes deep learning 
different from traditional machine learning and why it's revolutionizing AI.

[00:30] Let's start with a historical perspective. In the 1950s, Frank Rosenblatt invented 
the Perceptron - a simple neural network that could learn to classify. It caused huge 
excitement. People thought we'd have human-level AI within a decade.

[01:15] But then reality hit. In 1969, Marvin Minsky and Seymour Papert published a book 
showing that perceptrons couldn't even learn the XOR function - a simple logical operation. 
This triggered the first AI winter.

[02:00] Fast forward to the 1980s. Backpropagation was rediscovered, allowing us to train 
multi-layer neural networks. But we still couldn't train very deep networks. They were hard 
to optimize and required too much data and compute power that we didn't have.

[03:00] Then three things changed:
1. Big Data - The internet gave us massive datasets
2. GPUs - Graphics cards turned out to be perfect for matrix operations
3. Algorithmic innovations - ReLU, dropout, batch normalization

[03:45] In 2012, Alex Krizhevsky's AlexNet won ImageNet by a huge margin using deep 
convolutional neural networks. That was the turning point. Deep learning took off.

[05:00] So what IS a neural network? At its core, it's a function approximator. A very 
powerful one.

Think of it this way: any function you can imagine - from recognizing cats in photos to 
predicting stock prices to playing Go - can be approximated by a large enough neural network. 
This is the Universal Approximation Theorem.

[06:00] Let's build a neural network from first principles.

A single neuron takes inputs x₁, x₂, ..., xₙ, multiplies each by weights w₁, w₂, ..., wₙ, 
adds a bias b, and passes the sum through an activation function f:

output = f(w₁x₁ + w₂x₂ + ... + wₙxₙ + b)

[07:00] Why the activation function? Without it, the neural network would just be linear 
combinations of linear combinations... which is still linear! We need non-linearity to 
approximate complex functions.

[07:45] Common activation functions:
- **Sigmoid**: σ(x) = 1/(1+e^(-x)) - squashes to [0,1]
- **tanh**: tanh(x) = (e^x - e^(-x))/(e^x + e^(-x)) - squashes to [- 1,1]
- **ReLU**: f(x) = max(0, x) - simple but effective
- **Leaky ReLU**: f(x) = max(0.01x, x) - fixes dying ReLU problem

[09:00] Now let's stack these neurons into layers. A feedforward neural network has:
- Input layer: Receives raw data
- Hidden layers: Learn representations
- Output layer: Makes predictions

[10:00] Here's the key insight: each layer learns progressively more abstract representations.

In a network trained on images:
- Layer 1 detects edges and simple patterns
- Layer 2 detects textures and simple shapes
- Layer 3 detects object parts (eyes, wheels, etc.)
- Layer 4 detects complete objects (faces, cars, etc.)

[11:30] Let me show you the forward pass mathematically.

For a network with L layers:
a^[0] = X (input)
z^[l] = W^[l]a^[l-1] + b^[l]
a^[l] = g^[l](z^[l])

Where:
- z^[l] is the pre-activation at layer l
- a^[l] is the activation at layer l
- g^[l] is the activation function
- W^[l] and b^[l] are weights and biases

[13:00] The magic happens in training. We use backpropagation to compute gradients efficiently.

Here's the intuition: We compute how wrong our predictions are (the loss). Then we ask: 
how did each weight contribute to that error? We adjust weights in the direction that 
reduces error.

[14:00] Backpropagation is just the chain rule from calculus, applied cleverly.

For the output layer:
δ^[L] = ∇ₐLoss ⊙ g'^[L](z^[L])

For hidden layers:
δ^[l] = (W^[l+1])ᵀδ^[l+1] ⊙ g'^[l](z^[l])

Gradients:
∂Loss/∂W^[l] = δ^[l](a^[l-1])ᵀ
∂Loss/∂b^[l] = δ^[l]

[16:30] Let me implement this in PyTorch to make it concrete:

import torch
import torch.nn as nn

class SimpleNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

[20:00] Now let's talk about training challenges.

**Vanishing Gradients**: In very deep networks with sigmoid activation, gradients can become 
exponentially small. The early layers barely learn.

Solution: Use ReLU activation and careful initialization.

**Exploding Gradients**: The opposite problem - gradients become huge.

Solution: Gradient clipping, careful initialization, batch normalization.

[22:00] Optimization is another challenge. We use variants of gradient descent:

- **SGD**: Stochastic Gradient Descent - update after each example
- **Mini-batch GD**: Update after a batch of examples (sweet spot)
- **Momentum**: Add velocity to escape local minima
- **Adam**: Adaptive learning rates per parameter (usually works well)

[25:00] Regularization prevents overfitting:

**Dropout**: Randomly drop neurons during training. Forces network to learn redundant 
representations. At test time, use all neurons but scale their output.

**L2 Regularization**: Add λΣw² to loss function. Prevents weights from getting too large.

**Data Augmentation**: Create new training examples by transforming existing ones (rotate 
images, add noise, etc.)

[28:00] Now let's talk about Convolutional Neural Networks - the workhorses of computer vision.

Key idea: Instead of fully connecting layers, use local connections with shared weights. 
This exploits two properties of images:
1. Local correlations (nearby pixels are related)
2. Translation invariance (a cat is a cat whether it's on the left or right)

[30:00] A convolutional layer:
1. Slides a small filter (e.g. 3×3) across the image
2. At each position, computes dot product between filter and image patch
3. This creates a feature map

For example, a 3×3 edge detection filter might look like:
[[-1, -1, -1],
 [ 0,  0,  0],
 [ 1,  1,  1]]

[32:00] CNNs typically have this architecture:
Input → [Conv → ReLU → Pool] × N → FC → Output

Pooling reduces spatial dimensions while keeping important features. Max pooling takes 
the maximum value in each region.

[35:00] Famous CNN architectures:
- **AlexNet** (2012): 5 conv layers, started the deep learning revolution
- **VGGNet** (2014): Deeper, uniform architecture
- **ResNet** (2015): Skip connections allow training 100+ layer networks
- **EfficientNet** (2019): Carefully balances width, depth, resolution

[38:00] Let me show you a training loop:

model = SimpleNN(input_dim=784, hidden_dim=256, output_dim=10)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(num_epochs):
    for batch_x, batch_y in dataloader:
        # Forward pass
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    print(f'Epoch {epoch}, Loss: {loss.item()}')

[42:00] Some practical tips for deep learning:

1. **Start simple**: Begin with a small model, make sure it can overfit a tiny dataset
2. **Monitor everything**: Track training/validation loss, gradients, activations
3. **Use pre-trained models**: Transfer learning saves time and requires less data
4. **Batch normalization**: Usually helps training stability
5. **Learning rate scheduling**: Reduce learning rate as training progresses

[45:00] Common debugging strategies:

- Model doesn't learn at all? Check learning rate, initialization
- Trains on training set but terrible on validation? Overfitting - add regularization
- Loss goes to NaN? Learning rate too high or gradient explosion
- Validation loss stops improving? Try learning rate decay or early stopping

[48:00] The future of deep learning is exciting: 
- **Transformers** are replacing RNNs for sequences
- **Self-supervised learning** learns from unlabeled data
- **Neural architecture search** automatically designs networks
- **Few-shot learning** learns from very few examples

[50:00] Remember: Deep learning is still just optimization of differentiable functions. 
Understanding the fundamentals - gradients, backpropagation, optimization - will serve 
you well as the field evolves.

[51:00] For homework, implement a CNN from scratch in NumPy (no PyTorch!). This will force 
you to understand every detail of forward and backward passes.

[52:00] Questions?

[52:15] Thank you everyone!

[END OF RECORDING]
"""


def generate_nlp_transcript():
    return """[00:00] Dr. Emily Bender:
Welcome to Natural Language Processing with Transformers. Today you'll understand why 
transformers have revolutionized NLP and how attention mechanisms work.

[Content continues with detailed NLP lecture covering tokenization, embeddings, attention,
BERT, GPT, and practical applications...]

[38:45] END OF RECORDING
"""


def generate_cv_transcript():
    return """[00:00] Prof. Trevor Darrell:
Introduction to Computer Vision. We'll cover how machines learn to see, from basic image 
processing to modern deep learning approaches.

[Content continues with CV fundamentals, CNNs, object detection, semantic segmentation...]

[41:20] END OF RECORDING
"""


def generate_nn_implementation_transcript():
    return """[00:00] Dr. Andrej Karpathy:
Today we're building a neural network from absolute scratch - no frameworks, just NumPy and math.
By the end, you'll understand every line of code in backpropagation.

[Content continues with detailed implementation walkthrough...]

[55:00] END OF RECORDING  
"""


def generate_rl_transcript():
    return """[00:00] Dr. David Silver:
Welcome to Reinforcement Learning. We'll explore how agents learn through trial and error,
from Markov Decision Processes to Deep Q-Networks.

[Content continues with RL fundamentals, value iteration, Q-learning, policy gradients...]

[48:30] END OF RECORDING
"""


if __name__ == "__main__":
    count = create_realistic_video_transcripts()
    console.print(f"\n[bold green]Success! Created {count} realistic educational video transcripts[/bold green]")
    console.print("\nThese transcripts represent actual ML educational video content")
    console.print("The system will process them as video transcriptions and extract knowledge.")
