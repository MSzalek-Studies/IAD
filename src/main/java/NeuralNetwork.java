import models.InputLayer;
import models.Layer;
import models.OutputLayer;
import org.la4j.Matrix;
import org.la4j.vector.dense.BasicVector;

import java.util.Arrays;

/**
 * Created by szale_000 on 2017-03-07.
 */
public class NeuralNetwork {

    private Matrix inputMatrix;
    private Matrix expectedResults;

    private int numExamples;
    private int inputSize;
    private int hiddenSize;
    private int outputSize;
    private boolean includeBias;

    private InputLayer inputLayer;
    private Layer hiddenLayer;
    private OutputLayer outputLayer;

    public NeuralNetwork(Matrix inputMatrix, Matrix expectedResults,
                         int hiddenLayerSize, boolean includeBias) {
        this.inputMatrix = inputMatrix;
        this.expectedResults = expectedResults;
        numExamples = inputMatrix.rows();
        inputSize = inputMatrix.getRow(0).length();
        outputSize = expectedResults.getRow(0).length();
        hiddenSize = hiddenLayerSize;
        this.includeBias = includeBias;
        inputLayer = new InputLayer(inputSize, numExamples, includeBias);
        hiddenLayer = new Layer(inputSize + (includeBias ? 1 : 0), hiddenSize, numExamples, includeBias);
        outputLayer = new OutputLayer(hiddenSize + (includeBias ? 1 : 0), outputSize, numExamples);
    }

    public void train(int iterations) {
        inputLayer.setInput(inputMatrix);
        for (int it = 0; it < iterations; it++) {
            forwardPropagateNetwork(inputLayer, hiddenLayer, outputLayer);

            outputLayer.calculateErrors(expectedResults);
            hiddenLayer.calculateErrors(outputLayer);

            outputLayer.propagateBackward(hiddenLayer.getActivationValues());
            hiddenLayer.propagateBackward(inputLayer.getActivationValues());

            hiddenLayer.gradientDescent();
            outputLayer.gradientDescent();

            double cost = outputLayer.cost(expectedResults);
            System.out.println(it + "cost: " + cost);
            //TODO: show chart
        }
        showResults(inputLayer, hiddenLayer, outputLayer, numExamples);
    }

    /**
     * @param input mxn matrix
     * @return
     */
    public Matrix predict(Matrix input) {
        inputLayer.setInput(input);
        forwardPropagateNetwork(inputLayer, hiddenLayer, outputLayer);
        return outputLayer.getActivationValues();
    }

    private void showResults(Layer inputLayer, Layer hiddenLayer, Layer outputLayer, int numExamples) {
        forwardPropagateNetwork(inputLayer, hiddenLayer, outputLayer);

        System.out.println("\n\n================================\n\n");
        for (int i = 0; i < numExamples; i++) {
            System.out.println("input: " + Arrays.toString(((BasicVector) inputMatrix.getRow(i)).toArray()));
            System.out.println("expected: " + Arrays.toString(((BasicVector) expectedResults.getRow(i)).toArray()));
            System.out.println("output: " + Arrays.toString(((BasicVector) outputLayer.getActivationValues().getRow(i)).toArray()) + "\n");
        }
    }

    private void forwardPropagateNetwork(Layer inputLayer, Layer hiddenLayer, Layer outputLayer) {
        hiddenLayer.forwardPropagate(inputLayer);
        outputLayer.forwardPropagate(hiddenLayer);
    }

}
