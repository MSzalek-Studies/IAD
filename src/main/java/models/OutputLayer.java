package models;

import org.la4j.Matrix;

/**
 * Created by szale_000 on 2017-03-07.
 */
public class OutputLayer extends Layer {

    public OutputLayer(int numFeatures, int numNeurons, int numExamples) {
        super(numFeatures, numNeurons, numExamples, false);
    }

    public double cost(Matrix expectedValues) {
        Matrix costMatrix1 = expectedValues.transform((i, j, value) -> -1 * value * Math.log(activationValues.get(i, j)));
        Matrix costMatrix2 = expectedValues.transform((i, j, value) -> (1 - value) * Math.log(1 - activationValues.get(i, j)));
        Matrix costMatrix = costMatrix1.subtract(costMatrix2);
        double cost = costMatrix.sum() / numExamples;
        return cost;
    }

    public void calculateErrors(Matrix expectedValues) {
        errorOnValues = activationValues.subtract(expectedValues);
    }
}
