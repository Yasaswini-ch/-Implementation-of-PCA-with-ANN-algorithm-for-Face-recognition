
import { Step } from './types';

export const TRAINING_STEPS: Step[] = [
  {
    id: 'train-1',
    title: '1. Generate Face Database',
    description: 'Each face image is represented as a column vector. If we have p images of size m*n, the face database is a matrix of size (m*n) x p.',
    hasVisual: true,
    visualType: 'dataset',
  },
  {
    id: 'train-2',
    title: '2. Mean Calculation',
    description: 'Calculate the mean of all face vectors to get an "average face". This mean vector will have the dimension (m*n) x 1.',
    hasVisual: true,
    visualType: 'mean',
  },
  {
    id: 'train-3',
    title: '3. Do Mean Zero',
    description: 'Subtract the mean face from each individual face image. This centers the data around the origin.',
    hasVisual: true,
    visualType: 'mean-zero',
  },
  {
    id: 'train-4',
    title: '4. Calculate Co-Variance',
    description: 'Calculate the surrogate covariance matrix (p x p) from the mean-aligned faces. This is computationally easier than the full (m*n) x (m*n) matrix.',
    hasVisual: false,
  },
  {
    id: 'train-5',
    title: '5. Eigenvalue Decomposition',
    description: 'Find the eigenvectors and eigenvalues of the surrogate covariance matrix. These represent the principal components of the face data.',
    hasVisual: false,
  },
  {
    id: 'train-6',
    title: '6. Find Best Direction',
    description: 'Sort the eigenvalues in descending order and select the top k eigenvectors. This k-dimensional space is the "face space".',
    hasVisual: false,
  },
  {
    id: 'train-7',
    title: '7. Generating Eigenfaces',
    description: 'Project the mean-aligned faces onto the feature vector (the top k eigenvectors) to create the Eigenfaces. These are the basis vectors of the face space.',
    hasVisual: true,
    visualType: 'eigenfaces',
  },
  {
    id: 'train-8',
    title: '8. Generate Signature of Each Face',
    description: 'Project each mean-aligned face onto the Eigenfaces to get a set of weights (a signature vector) for each face. This vector is a compressed representation of the face.',
    hasVisual: false,
  },
  {
    id: 'train-9',
    title: '9. Apply ANN for Training',
    description: 'Use the generated signature vectors as input to train a back-propagation neural network to classify the faces.',
    hasVisual: false,
  },
];

export const TESTING_STEPS: Step[] = [
  {
    id: 'test-1',
    title: '1. Input & Normalize Test Image',
    description: 'Take a new image for testing and convert it into a column vector, same as the training images.',
    hasVisual: true,
    visualType: 'interactive',
  },
  {
    id: 'test-2',
    title: '2. Do Mean Zero',
    description: 'Subtract the *same mean face* calculated during training from the new test face vector.',
    hasVisual: true,
    visualType: 'interactive',
  },
  {
    id: 'test-3',
    title: '3. Project to Eigenfaces',
    description: 'Project the mean-aligned test face onto the Eigenfaces to get its signature (weight vector).',
    hasVisual: true,
    visualType: 'interactive',
  },
  {
    id: 'test-4',
    title: '4. Predict with ANN',
    description: 'Feed the signature of the test face to the trained ANN model to predict which known face it is, or if it is an unknown "imposter".',
    hasVisual: true,
    visualType: 'interactive',
  },
];
