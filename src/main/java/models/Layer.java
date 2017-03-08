package models;

import org.la4j.Matrix;
import org.la4j.matrix.dense.Basic2DMatrix;
import org.la4j.matrix.functor.MatrixFunction;
import utils.MatrixUtils;

/**
 * Created by szale_000 on 2017-03-07.
 */
public class Layer {

    protected static final double LEARNING_RATE = 0.5;
    protected final boolean hasBias;
    //f-number of features
    //n-number of neurons
    //m-number of training examples
    protected Matrix weights; // f x n
    protected Matrix activationValues; // m x n (+1 if bias)
    protected Matrix errorOnValues; // m x n (+1 if bias)
    protected Matrix errorOnWeights; // f x n
    protected int numFeatures;
    protected int numNeurons;
    protected int numExamples;

    public Layer(int numFeatures, int numNeurons, int numExamples, boolean hasBias) {
        this.hasBias = hasBias;
        this.numFeatures = numFeatures;
        this.numNeurons = numNeurons;
        this.numExamples = numExamples;

        weights = MatrixUtils.randomlyInitWeights(numFeatures, numNeurons);
        activationValues = new Basic2DMatrix(numExamples, numNeurons);
        errorOnValues = new Basic2DMatrix(numNeurons, numExamples);
        errorOnWeights = new Basic2DMatrix(numFeatures, numNeurons);
    }

    public Layer(int numFeatures, int numNeurons, int numExamples) {
        this(numFeatures, numNeurons, numExamples, false);
    }

    public void calculateErrors(Layer nextLayer) {
        Matrix nextWeights = nextLayer.getWeights();
        if (nextLayer.hasBias()) {
            nextWeights = nextWeights.removeFirstColumn();
        }
        Matrix nextLayerErrors = nextLayer.getErrorOnValues();
        errorOnValues = nextWeights.multiply(nextLayerErrors).transform(new MatrixFunction() {
            @Override
            public double evaluate(int i, int j, double value) {
                Matrix sigmoided = MatrixUtils.sigmoid(nextLayer.activationValues);
                return value * sigmoided.get(i, j) * (1 - sigmoided.get(i, j));
            }
        }).transpose();//pewnie tu sie spierdoli
    }

    public void propagateBackward(Matrix previousActivationValues) {
        errorOnWeights = previousActivationValues.transpose().multiply(errorOnValues);
        if (hasBias()) {
            errorOnWeights = errorOnWeights.removeFirstColumn();
        }
    }

    public void gradientDescent() {
        weights = weights.subtract(errorOnWeights.multiply(LEARNING_RATE / numExamples));
    }

    /**
     * @param inputValues is vector f x m where f is number of features of previous (L-1) layer
     */
    public void forwardPropagate(Matrix inputValues) {
        Matrix values = inputValues.multiply(weights);
        activationValues = MatrixUtils.sigmoid(values);
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
