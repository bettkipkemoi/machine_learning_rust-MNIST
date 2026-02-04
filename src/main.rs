// implement machine learning in rust, using MNIST dataset
use mnist_reader::{MnistReader, print_image};
 // implement a simple neural network using the ndarray and ndarray-rand crates
use ndarray::prelude::*;
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Uniform;
   
fn main() {
    println!("Loading MNIST dataset...");
    let mut mnist = MnistReader::new("mnist-data");
    mnist.load().unwrap();
    
    println!("Train data size: {}", mnist.train_data.len());
    println!("Test data size: {}", mnist.test_data.len());
    
    // Convert training data to Array1<f32> and normalize (divide by 255)
    let train_inputs: Vec<Array1<f32>> = mnist.train_data
        .iter()
        .take(5000) // Use first 5000 samples
        .map(|img| Array1::from_vec(img.iter().map(|&x| x / 255.0).collect()))
        .collect();
    
    // Convert labels to one-hot encoded vectors
    let train_targets: Vec<Array1<f32>> = mnist.train_labels
        .iter()
        .take(5000)
        .map(|&label| {
            let mut target = Array1::zeros(10);
            target[label as usize] = 1.0;
            target
        })
        .collect();
    
    // Create neural network
    println!("\nCreating neural network (784 -> 128 -> 10)...");
    let mut nn = NeuralNetwork::new(784, 128, 10);
    
    // Train the network
    println!("Training neural network...");
    let learning_rate = 0.05;
    let epochs = 10;
    
    for epoch in 0..epochs {
        // Decay learning rate over time
        let current_lr = learning_rate / (1.0 + epoch as f32 * 0.1);
        
        for (input, target) in train_inputs.iter().zip(train_targets.iter()) {
            nn.backpropagation(input, target, current_lr);
        }
        
        // Calculate and display training accuracy every 2 epochs
        if epoch % 2 == 1 {
            let mut correct = 0;
            for (input, &label) in train_inputs.iter().zip(mnist.train_labels.iter().take(5000)) {
                if nn.predict(input) == label as usize {
                    correct += 1;
                }
            }
            println!("Epoch {} (lr={:.4}): Training accuracy = {:.2}%", epoch + 1, current_lr, (correct as f32 / train_inputs.len() as f32) * 100.0);
        }
    }
    println!("Training complete!");
    
    // Display first 5 images with predictions
    println!("\n=== First 5 Training Samples ===");
    for i in 0..5 {
        let prediction = nn.predict(&train_inputs[i]);
        let actual = mnist.train_labels[i];
        println!("\n--- Sample {} ---", i);
        println!("Predicted: {}  |  Actual: {}  |  {}", 
                 prediction, actual,
                 if prediction == actual as usize { "✓ Correct" } else { "✗ Wrong" });
        print_image(&mnist.train_data[i]);
    }
    
    // Test on more samples
    println!("\n=== Additional Testing Predictions ===");
    for i in 0..10 {
        let prediction = nn.predict(&train_inputs[i]);
        let actual = mnist.train_labels[i];
        println!("Sample {}: Predicted = {}, Actual = {}, {}", 
                 i, prediction, actual,
                 if prediction == actual as usize { "✓" } else { "✗" });
    }
    
    // Calculate accuracy on test set
    let test_inputs: Vec<Array1<f32>> = mnist.test_data
        .iter()
        .take(1000) // Test on first 1000 samples
        .map(|img| Array1::from_vec(img.iter().map(|&x| x / 255.0).collect()))
        .collect();
    
    let mut correct = 0;
    for (i, test_input) in test_inputs.iter().enumerate() {
        let prediction = nn.predict(test_input);
        if prediction == mnist.test_labels[i] as usize {
            correct += 1;
        }
    }
    
    println!("\nTest Accuracy: {}/{} ({:.2}%)", correct, test_inputs.len(), 
             (correct as f32 / test_inputs.len() as f32) * 100.0);
}

 // define the neural network structure
#[allow(dead_code)]
struct NeuralNetwork {
    input_size: usize,
    hidden_size: usize,
    output_size: usize,
    weights_input_hidden: Array2<f32>,
    weights_hidden_output: Array2<f32>,
}

impl NeuralNetwork {
    fn new(input_size: usize, hidden_size: usize, output_size: usize) -> Self {
        let weights_input_hidden = Array::random((input_size, hidden_size), Uniform::new(-1.0, 1.0).unwrap());
        let weights_hidden_output = Array::random((hidden_size, output_size), Uniform::new(-1.0, 1.0).unwrap());
        NeuralNetwork {
            input_size,
            hidden_size,
            output_size,
            weights_input_hidden,
            weights_hidden_output,
        }
    }
}

// implement feedforward function
impl NeuralNetwork {
    fn feedforward(&self, input: &Array1<f32>) -> Array1<f32> {
        let hidden = input.dot(&self.weights_input_hidden).map(|x| x.max(0.0)); // ReLU activation
        let output = hidden.dot(&self.weights_hidden_output).map(|x| x.max(0.0)); // ReLU activation
        output
    }
}

// implement backpropagation function
impl NeuralNetwork {
    fn backpropagation(&mut self, input: &Array1<f32>, target: &Array1<f32>, learning_rate: f32) {
        let hidden = input.dot(&self.weights_input_hidden).map(|x| x.max(0.0)); // ReLU activation
        let output = hidden.dot(&self.weights_hidden_output).map(|x| x.max(0.0)); // ReLU activation
        
        // Calculate errors
        let output_error = target - &output;
        let hidden_error = output_error.dot(&self.weights_hidden_output.t()).map(|x| if *x > 0.0 { *x } else { 0.0 }); // ReLU derivative
        
        // Update weights using outer products
        // For hidden_output weights: outer product of hidden (128,) and output_error (10,)
        for i in 0..self.weights_hidden_output.nrows() {
            for j in 0..self.weights_hidden_output.ncols() {
                self.weights_hidden_output[[i, j]] += learning_rate * hidden[i] * output_error[j];
            }
        }
        
        // For input_hidden weights: outer product of input (784,) and hidden_error (128,)
        for i in 0..self.weights_input_hidden.nrows() {
            for j in 0..self.weights_input_hidden.ncols() {
                self.weights_input_hidden[[i, j]] += learning_rate * input[i] * hidden_error[j];
            }
        }
    }
}

// implement prediction function
impl NeuralNetwork {
    fn predict(&self, input: &Array1<f32>) -> usize {
        let output = self.feedforward(input);
        output.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0
    }
}

