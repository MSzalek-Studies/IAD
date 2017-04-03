package models;

import activator.Activator;
import org.la4j.Matrix;
import org.la4j.matrix.dense.Basic2DMatrix;
import org.la4j.matrix.functor.MatrixFunction;
import utils.MatrixUtils;

/**
 * Created by szale_000 on 2017-03-07.
 */
public class Layer {

    protected final boolean hasBias;
    //f-number of features
    //n-number of neurons
    //m-number of training examples
    protected Matrix weights; // f x n
    protected Matrix activationValues; // m x n (+1 if bias)
    protected Matrix errorOnValues; // m x n (+1 if bias)
    protected Matrix errorOnWeights; // f x n
    protected Matrix previousWeightChange; // w poprzedniej epoce
    protected int numFeatures;
    protected int numNeurons; //without bias
    protected double momentum;
    protected double learningRate;

    protected Activator activator;

    public Layer(int numFeatures, int numNeurons, boolean hasBias, Activator activator, double learningRate, double momentum) {
        this.hasBias = hasBias;
        this.numFeatures = numFeatures;
        this.numNeurons = numNeurons;
        this.activator = activator;
        this.learningRate = learningRate;
        this.momentum = momentum;

        weights = MatrixUtils.randomlyInitWeights(numFeatures, numNeurons);
        //activationValues = new Basic2DMatrix(numExamples, numNeurons); niepotrzebne
        previousWeightChange = new Basic2DMatrix(numFeatures, numNeurons);
    }

    public Layer(int numFeatures, int numNeurons, Activator activator, double learningRate, double momentum) {
        this(numFeatures, numNeurons, false, activator, learningRate, momentum);
    }

    public void calculateErrors(Layer nextLayer) {
        Matrix nextWeights = nextLayer.getWeights();
        Matrix nextLayerErrors = nextLayer.getErrorOnValues();
        if (nextLayer.hasBias()) {
            nextLayerErrors = nextLayerErrors.removeFirstColumn();
        }
        Matrix temp = nextLayerErrors.multiply(nextWeights.transpose());
        errorOnValues = temp.transform(new MatrixFunction() {
            @Override
            public double evaluate(int i, int j, double value) {
                return value * activationValues.get(i, j) * (1 - activationValues.get(i, j));
            }
        });//pewnie tu sie popsuje
    }

    public void propagateBackward(Matrix previousActivationValues) {
        if (hasBias()) {
            errorOnValues = errorOnValues.removeFirstColumn();
        }
        errorOnWeights = previousActivationValues.transpose().multiply(errorOnValues);
        //Matrix regularization = weights.multiply(LEARNING_RATE/numExamples);
        //errorOnWeights = errorOnWeights.add(regularization);
    }

    public void gradientDescent() {
        //Matrix difference = errorOnWeights.multiply(LEARNING_RATE/numExamples);
        Matrix difference = errorOnWeights.multiply(learningRate).add(previousWeightChange.multiply(momentum));
        previousWeightChange = difference;
        weights = weights.subtract(difference);
    }

    /**
     * @param inputValues is vector f x m where f is number of features of previous (L-1) layer
     */
    public void forwardPropagate(Matrix inputValues) {
        Matrix values = inputValues.multiply(weights);
        activationValues = activator.activate(values);
        if (hasBias) {
            activationValues = MatrixUtils.addBiasColumn(activationValues);
        }
    }

    public void forwardPropagate(Layer previousLayer) {
        forwardPropagate(previousLayer.getActivationValues());
    }

    //============= GETTERS ===================

    public Matrix getActivationValues() {
        return activationValues;
    }

    public Matrix getErrorOnValues() {
        return errorOnValues;
    }

    public Matrix getWeights() {
        return weights;
    }

    public boolean hasBias() {
        return hasBias;
    }
}
