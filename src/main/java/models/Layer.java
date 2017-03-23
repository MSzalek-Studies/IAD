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

    protected static final double LEARNING_RATE = 0.003;
    protected static final double MOMENTUM = 0.9;
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
    protected int numExamples;

    protected Activator activator;

    public Layer(int numFeatures, int numNeurons, int numExamples, boolean hasBias, Activator activator) {
        this.hasBias = hasBias;
        this.numFeatures = numFeatures;
        this.numNeurons = numNeurons;
        this.numExamples = numExamples;
        this.activator = activator;

        weights = MatrixUtils.randomlyInitWeights(numFeatures, numNeurons);
        activationValues = new Basic2DMatrix(numExamples, numNeurons);
        previousWeightChange = new Basic2DMatrix(numFeatures, numNeurons);
    }

    public Layer(int numFeatures, int numNeurons, int numExamples, Activator activator) {
        this(numFeatures, numNeurons, numExamples, false, activator);
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
        });//pewnie tu sie spierdoli
    }

    public void propagateBackward(Matrix previousActivationValues) {
        if (hasBias()) {
            errorOnValues = errorOnValues.removeFirstColumn();
        }
        errorOnWeights = previousActivationValues.transpose().multiply(errorOnValues);
        errorOnWeights = errorOnWeights.add(weights.multiply(LEARNING_RATE / numExamples));
    }

    public void gradientDescent() {
        //Matrix difference = errorOnWeights.multiply(LEARNING_RATE);
        Matrix difference = errorOnWeights.multiply(LEARNING_RATE).add(previousWeightChange.multiply(MOMENTUM));
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
