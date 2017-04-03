package models;

import activator.Activator;
import org.la4j.Matrix;
import org.la4j.matrix.functor.MatrixFunction;

/**
 * Created by szale_000 on 2017-03-07.
 */
public class OutputLayer extends Layer {

    public OutputLayer(int numFeatures, int numNeurons, Activator activator, double learningRate, double momentum) {
        super(numFeatures, numNeurons, false, activator, learningRate, momentum);
    }

    public double cost(Matrix expectedValues, int numExamples) {
        double cost = (expectedValues.subtract(activationValues)).transform(new MatrixFunction() {
            @Override
            public double evaluate(int i, int j, double value) {
                return (value * value) / (2 * numExamples);
            }
        }).sum(); //TODO: dzielic przez 2?
        return cost;
    }

    public void calculateErrors(Matrix expectedValues) {
        errorOnValues = activationValues.subtract(expectedValues);
    }
}
