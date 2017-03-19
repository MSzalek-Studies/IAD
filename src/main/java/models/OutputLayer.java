package models;

import activator.Activator;
import org.la4j.Matrix;
import org.la4j.matrix.functor.MatrixFunction;

/**
 * Created by szale_000 on 2017-03-07.
 */
public class OutputLayer extends Layer {

    public OutputLayer(int numFeatures, int numNeurons, int numExamples, Activator activator) {
        super(numFeatures, numNeurons, numExamples, false, activator);
    }

    public double cost(Matrix expectedValues) {
        //Matrix costMatrix1 = expectedValues.transform((i, j, value) -> -1 * value * Math.log(activationValues.get(i, j)));
        //Matrix costMatrix2 = expectedValues.transform((i, j, value) -> (1 - value) * Math.log(1 - activationValues.get(i, j)));
        //Matrix costMatrix = costMatrix1.subtract(costMatrix2);
        //double cost = costMatrix.sum() / numExamples;
        double cost = (expectedValues.subtract(activationValues)).transform(new MatrixFunction() {
            @Override
            public double evaluate(int i, int j, double value) {
                return value * value;
            }
        }).sum() / 2;
        return cost;
    }

    public void calculateErrors(Matrix expectedValues) {
        errorOnValues = activationValues.subtract(expectedValues);
    }
}
