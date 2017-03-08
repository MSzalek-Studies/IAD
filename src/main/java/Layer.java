import org.apache.commons.math3.linear.*;

import java.util.Random;

/**
 * Created by szale_000 on 2017-03-07.
 */
public class Layer {

    protected static final double LEARNING_RATE = 0.25;

    protected RealMatrix weights;//nxk where n is number of features and k i number of neurons in layer
    protected RealVector values;
    protected RealVector activationValues;
    protected RealVector errorOnValues;
    protected RealMatrix errorOnWeights;

    public Layer(int numFeatures, int numNeurons) {
        weights = initWeights(numFeatures, numNeurons);
        values = MatrixUtils.createRealVector(new double[numNeurons]);
        activationValues = new ArrayRealVector(new double[numNeurons]);
        errorOnValues = new ArrayRealVector(new double[numNeurons]);
        errorOnWeights = MatrixUtils.createRealMatrix(numFeatures, numNeurons);
    }

    public RealVector getActivationValues() {
        return activationValues;
    }

    public RealVector getErrorOnValues() {
        return errorOnValues;
    }

    public RealMatrix getWeights() {
        return weights;
    }

    public void calculateErrors(Layer nextLayer) {
        RealVector derivative = activationValues.ebeMultiply(activationValues.mapMultiply(-1).mapAdd(1));
        errorOnValues = nextLayer.getWeights().operate(nextLayer.getErrorOnValues()).ebeMultiply(derivative);
    }

    public void propagateBackward(RealVector previousActivationValues) {
        RealMatrix prevActVals = new Array2DRowRealMatrix(previousActivationValues.getDimension(),1);
        prevActVals.setColumnVector(0, previousActivationValues);
        RealMatrix errors = new Array2DRowRealMatrix(errorOnValues.getDimension(),1);
        errors.setColumnVector(0, errorOnValues);

        errorOnWeights = errorOnWeights.add(prevActVals.multiply(errors.transpose()));
    }

    public void gradientDescent(int numberOfExamples) {
        weights = weights.subtract(errorOnWeights.scalarMultiply(LEARNING_RATE/numberOfExamples));
    }
    public void clearExceptWeights() {
        int numNeurons = weights.getColumnDimension();
        int numFeatures = weights.getRowDimension();
        values = MatrixUtils.createRealVector(new double[numNeurons]);
        activationValues = new ArrayRealVector(new double[numNeurons]);
        errorOnValues = new ArrayRealVector(new double[numNeurons]);
        errorOnWeights = MatrixUtils.createRealMatrix(numFeatures, numNeurons);
    }

    /**
     *
     * @param inputValues is vector nx1 where n is number of features
     */
    public void forwardPropagate(RealVector inputValues) {
        values = weights.transpose().operate(inputValues);
        activationValues = sigmoid(values);
    }

    public void forwardPropagate(Layer layer) {
        forwardPropagate(layer.getActivationValues());
    }

    private RealVector sigmoid(RealVector vector) {
        RealVector result = new ArrayRealVector(vector.getDimension());
        for (int i=0; i<vector.getDimension(); i++) {
            double value = vector.getEntry(i);
            value = 1 / (1+Math.exp(-value));
            result.setEntry(i,value);
        }
        return result;
    }

    private RealMatrix initWeights(int numFeatures, int numNeurons) {
        RealMatrix weights = MatrixUtils.createRealMatrix(numFeatures, numNeurons);
        for(int i=0; i<numFeatures; i++) {
            for (int j=0; j<numNeurons; j++) {
                weights.setEntry(i,j,new Random().nextDouble());
            }
        }
        return weights;
    }
}
